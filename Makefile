.PHONY: install test cov lint fmt type baseline gate demo calibrate clean all

PY ?= python3
GOLDEN ?= examples/golden/triage.yaml

install:
	$(PY) -m pip install -e ".[dev]"

test:
	pytest -q

cov:
	pytest --cov=promptgate --cov-report=term-missing

lint:
	ruff check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

type:
	mypy

baseline:
	promptgate baseline $(GOLDEN)

gate:
	promptgate gate $(GOLDEN)

# The headline demo: record a baseline on the good prompt, then run the one
# with an instruction deleted and watch the build go red.
demo:
	@rm -rf .promptgate
	promptgate baseline $(GOLDEN)
	@echo
	-promptgate gate $(GOLDEN) --prompt-file examples/prompts/triage.v2.txt

calibrate:
	promptgate calibrate examples/labels.jsonl

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage .promptgate
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

all: lint type test demo
