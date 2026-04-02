# RuleShift Benchmark

RuleShift is a Kaggle-first benchmark for rule updating under conflicting evidence. Each episode shows labeled examples, silently flips the governing rule mid-sequence, and asks the model to predict four post-shift probe labels.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
python3 -m pip install -e .
python3 -m pytest tests/test_kbench_notebook.py -v
python3 -m scripts.deploy --skip-publish
```

Use `python3 -m scripts.deploy --release-message "..."` only when you intend to publish the Kaggle dataset and notebook bundle.

## Main Flow

- Official notebook: `packaging/kaggle/ruleshift_notebook_task.ipynb`
- Notebook runtime path: `src/core/kaggle/runner.py`
- Official payload builder and validation: `src/core/kaggle/payload.py`

The notebook loads the frozen leaderboard split, runs the binary task, and emits the validated Kaggle payload.
