PYTHON ?= .venv/bin/python
CLI := $(PYTHON) scripts/ruleshift_benchmark.py

.PHONY: help test validity reaudit integrity evidence-pass notebook-check

help:
	@printf "Available targets:\n"
	@printf "  make test\n"
	@printf "  make validity\n"
	@printf "  make reaudit\n"
	@printf "  make integrity\n"
	@printf "  make evidence-pass\n"
	@printf "  make notebook-check\n"

test:
	$(CLI) test

validity:
	$(CLI) validity

reaudit:
	$(CLI) reaudit

integrity:
	$(CLI) integrity

evidence-pass:
	$(CLI) evidence-pass

notebook-check:
	$(PYTHON) -m pytest tests/test_kbench_notebook.py -v
