PYTHON ?= python3
JUPYTER ?= jupyter

.PHONY: notelab test build-private verify-public verify-private \
        release-check \
        deploy-dataset deploy-private-dataset deploy-notebook deploy-all

notelab:
	$(JUPYTER) lab --no-browser kaggle/notebook/cogflex_notebook_task.ipynb

test:
	$(PYTHON) -m unittest discover -s tests -q

build-private:
	$(PYTHON) -m scripts.build_private_cogflex_dataset

verify-public:
	$(PYTHON) -m scripts.verify_cogflex --split public

verify-private:
	@if [ -z "$(COGFLEX_PRIVATE_BUNDLE_DIR)" ]; then \
		echo "COGFLEX_PRIVATE_BUNDLE_DIR is required for verify-private" >&2; \
		echo "Example: COGFLEX_PRIVATE_BUNDLE_DIR=/abs/path/to/private-bundle make verify-private" >&2; \
		exit 1; \
	fi
	$(PYTHON) -m scripts.verify_cogflex --split private \
		--private-bundle-dir "$(COGFLEX_PRIVATE_BUNDLE_DIR)"

# ── Release gate ────────────────────────────────────────────────────────────
# Rebuild all artifacts, verify both splits, run the test suite.
# Writes .release_ok on success. Deploy targets enforce this gate.
release-check:
	./scripts/release_check.sh

# ── Publish targets (require release-check to have passed) ──────────────────
deploy-dataset:
	./scripts/deploy_dataset.sh

deploy-private-dataset:
	./scripts/deploy_private_dataset.sh

deploy-notebook:
	./scripts/deploy_notebook.sh

deploy-all: deploy-dataset deploy-private-dataset deploy-notebook
