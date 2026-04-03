# RuleShift Benchmark

RuleShift is a Kaggle-first benchmark for rule updating under conflicting evidence. Each episode shows labeled examples, silently flips the governing rule mid-sequence, and asks the model to predict four post-shift probe labels.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
python3 -m pip install -e .
python3 -m pytest -v
python3 scripts/build_kaggle.py
```

## Main Flow

Read the repository in this order:

1. `kaggle/ruleshift_notebook_task.ipynb`
2. `src/tasks/ruleshift_benchmark/__init__.py`
3. `src/tasks/ruleshift_benchmark/runtime.py`

`scripts/build_kaggle.py` copies exactly what Kaggle needs: the notebook plus the `src/` runtime dataset.
