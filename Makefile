.PHONY: quality style
quality:
	ruff check .
	ruff format --check .

style:
	ruff format .
	ruff check --fix .