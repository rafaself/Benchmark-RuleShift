PYTHON ?= .venv/bin/python
CLI := PYTHONPATH=src $(PYTHON) -m core.cli

.PHONY: help test validity reaudit integrity evidence-pass notebook-check contract-audit compliance-check update-hashes

help:
	@printf "Available targets:\n"
	@printf "  make test\n"
	@printf "  make validity\n"
	@printf "  make reaudit\n"
	@printf "  make integrity\n"
	@printf "  make evidence-pass\n"
	@printf "  make notebook-check\n"
	@printf "  make contract-audit\n"
	@printf "  make compliance-check\n"
	@printf "  make update-hashes\n"

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

contract-audit:
	$(CLI) contract-audit

compliance-check:
	$(PYTHON) scripts/check_public_private_isolation.py
	$(PYTHON) -m pytest tests/test_kbench_notebook.py::TestNotebookEndToEnd -v

update-hashes:
	$(PYTHON) scripts/update_manifest_hashes.py
