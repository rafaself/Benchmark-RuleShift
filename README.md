# RuleShift Benchmark

Build the Kaggle package:

```bash
python3 -m pip install -e .
python3 scripts/build_kaggle.py --output-dir /tmp/ruleshift-kaggle-build
```

Output:

- `/tmp/ruleshift-kaggle-build/kernel`: notebook upload payload
- `/tmp/ruleshift-kaggle-build/dataset`: runtime dataset upload payload
