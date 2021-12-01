SHELL=/bin/bash
LINT_PATHS=pyfilter/ examples/ setup.py

venv:
	python3 -m venv .venv
	pip install -e ".[dev]"

lint:
	flake8 ${LINT_PATHS}

format:
	isort ${LINT_PATHS}
	black ${LINT_PATHS}

clean:
	rm -rf .venv pyfilter.egg-info pyfilter/__pycache__ pyfilter/*/__pycache__

.PHONY: venv format clean lint
