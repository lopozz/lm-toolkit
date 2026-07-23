"""
Lightweight on-disk cache for benchmark results, keyed by model and task.

Unlike `mteb.ResultCache`, this doesn't track dataset revisions or MTEB
versions to decide what to rerun: a cached result is either present (skip) or
absent (run).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name)


def _model_dir(model: str) -> str:
    return model.replace("/", "__")


class ResultCache:
    def __init__(self, root: str | Path = "results", benchmark: str | None = None):
        self.root = Path(root).expanduser()
        self.benchmark = benchmark

    def _base_dir(self) -> Path:
        return self.root / self.benchmark if self.benchmark else self.root

    def has_result(self, model: str, task: str, revision: str | None = None) -> bool:
        return self.result_path(model, task, revision).exists()

    def result_path(self, model: str, task: str, revision: str | None = None) -> Path:
        parts = [self._base_dir(), _model_dir(model)]
        if revision is not None:
            parts.append(revision)
        return Path(*parts) / f"{_safe_filename(str(task))}.json"

    def save_result(
        self,
        result: dict[str, Any],
        model: str,
        task: str,
        revision: str | None = None,
    ) -> Path:
        metadata = result.setdefault("metadata", {})
        metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        metadata.setdefault("model", model)
        metadata.setdefault("task", task)
        if self.benchmark is not None:
            metadata.setdefault("benchmark", self.benchmark)
        if revision is not None:
            metadata.setdefault("revision", revision)

        output_path = self.result_path(model, task, revision)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    def load_results(
        self,
        models: list[str] | None = None,
        tasks: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        model_filter = set(models or [])
        task_filter = set(tasks or [])
        loaded: list[dict[str, Any]] = []

        for path in sorted(self._base_dir().rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                continue

            model = metadata.get("model")
            task = metadata.get("task")
            if model_filter and model not in model_filter:
                continue
            if task_filter and task not in task_filter:
                continue

            payload["_path"] = str(path)
            loaded.append(payload)

        return loaded
