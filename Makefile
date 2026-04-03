.PHONY: notelab deploy-dataset deploy-notebook

notelab:
	.venv/bin/jupyter lab --no-browser kaggle/ruleshift_notebook_task.ipynb

deploy-dataset:
	./scripts/deploy_dataset.sh

deploy-notebook:
	./scripts/deploy_notebook.sh
