PYTHON ?= .venv/bin/python
CLI := $(PYTHON) scripts/ife.py

.PHONY: help test validity reaudit integrity evidence-pass

help:
	@printf "Available targets:\n"
	@printf "  make test\n"
	@printf "  make validity\n"
	@printf "  make reaudit\n"
	@printf "  make integrity\n"
	@printf "  make evidence-pass\n"

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
