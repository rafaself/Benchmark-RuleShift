.PHONY: notelab verify-public verify-private deploy-dataset deploy-private-dataset deploy-notebook deploy-all

notelab:
	.venv/bin/jupyter lab --no-browser kaggle/notebook/ruleshift_notebook_task.ipynb

verify-public:
	.venv/bin/python scripts/verify_ruleshift.py --split public

verify-private:
	.venv/bin/python scripts/verify_ruleshift.py --split private

deploy-dataset:
	./scripts/deploy_dataset.sh

deploy-private-dataset:
	./scripts/deploy_private_dataset.sh

deploy-notebook:
	./scripts/deploy_notebook.sh

deploy-all: deploy-dataset deploy-private-dataset deploy-notebook
