VENV_BIN := .venv/bin

PYTHON ?= $(if $(wildcard $(VENV_BIN)/python),$(VENV_BIN)/python,python3)
JUPYTER ?= $(if $(wildcard $(VENV_BIN)/jupyter),$(VENV_BIN)/jupyter,jupyter)

.PHONY: notelab test build-private build-test verify-public verify-private \
        release-check web \
        deploy-dataset deploy-test-dataset deploy-private-dataset deploy-notebook deploy-web deploy-all

notelab:
	$(JUPYTER) lab --no-browser kaggle/notebook/cogflex_notebook_task.ipynb

web:
	cd web && npm run dev

test:
	$(PYTHON) -m unittest discover -s tests -q

build-private:
	$(PYTHON) -m scripts.build_private_cogflex_dataset

build-test:
	$(PYTHON) -m scripts.build_test_cogflex_dataset

verify-public:
	$(PYTHON) -m scripts.verify_cogflex --split public

verify-private:
	$(PYTHON) -m scripts.verify_cogflex --split private

# ── Release check (optional) ────────────────────────────────────────────────
# Rebuild all artifacts, verify both splits, run the test suite.
release-check:
	./scripts/release_check.sh

# ── Publish targets ──────────────────────────────────────────────────────────
deploy-dataset:
	./scripts/deploy_dataset.sh

deploy-test-dataset:
	./scripts/deploy_test_dataset.sh

deploy-private-dataset:
	./scripts/deploy_private_dataset.sh

deploy-notebook:
	./scripts/deploy_notebook.sh

deploy-web:
	./scripts/deploy_web.sh

deploy-all: deploy-dataset deploy-test-dataset deploy-private-dataset deploy-notebook
