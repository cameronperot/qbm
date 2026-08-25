.PHONY: help sync install test test-quiet lint lint-fix format format-check typecheck check pre-commit docs clean

help:
	@echo "Available commands:"
	@echo "  make sync          - Sync dependencies with uv locked state"
	@echo "  make install       - Alias for make sync"
	@echo "  make test          - Run tests with coverage"
	@echo "  make test-quiet    - Run tests in quiet mode"
	@echo "  make lint          - Check code style and rules with Ruff"
	@echo "  make lint-fix      - Automatically fix Ruff lint errors"
	@echo "  make format        - Format code with Ruff"
	@echo "  make format-check  - Check code formatting with Ruff"
	@echo "  make typecheck     - Run static type checking with ty"
	@echo "  make check         - Run all checks (lint, format-check, typecheck, test)"
	@echo "  make pre-commit    - Run pre-commit hooks on all files"
	@echo "  make docs-build    - Build MkDocs documentation"
	@echo "  make docs-serve    - Serve MkDocs documentation"
	@echo "  make clean         - Clean cache directories and build artifacts"

sync:
	uv sync --locked

install: sync

test:
	uv run pytest

test-quiet:
	uv run pytest -q

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run ty check

check: lint format-check typecheck test

pre-commit:
	uv run pre-commit run --all-files

docs-build:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve --strict

clean:
	rm -rf .coverage .coverage.* .pytest_cache .ruff_cache .mypy_cache build dist site
	find . -type d -name "__pycache__" -exec rm -rf {} +
