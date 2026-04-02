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

- Official notebook: `kaggle/ruleshift_notebook_task.ipynb`
- Runtime protocol + episode schema: `src/tasks/ruleshift_benchmark/protocol.py` and `src/tasks/ruleshift_benchmark/schema.py`
- Split loading, episode generation, prompt rendering + bundle assembly: `src/tasks/ruleshift_benchmark/splits.py`
- Evaluation runner: `src/tasks/ruleshift_benchmark/runner.py`
- Kaggle package build: `scripts/build_kaggle.py`

The notebook loads the frozen leaderboard bundle, runs the binary task, and returns the aggregate `(numerator, denominator)` result.
