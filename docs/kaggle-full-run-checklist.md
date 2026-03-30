# Kaggle Full Run Checklist

Use this checklist for the real Phase 6 hosted Kaggle run after the local validation phases pass.

## Before the run

Run the local gate:

```bash
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local ./scripts/pre_deploy_check.sh
```

Confirm the notebook is on the normal path:

- `DEBUG_MODE = False`
- `DEBUG_LIMIT = None`
- no temporary smoke toggles enabled
- final task cell still runs `payload = ruleshift_benchmark_v1_binary(kbench.llm)`
- final selection cell still uses `%choose ruleshift_benchmark_v1_binary`

Rebuild the release artifacts you intend to publish:

```bash
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local python scripts/build_runtime_dataset_package.py --output-dir /tmp/ruleshift-runtime-package
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local python scripts/build_kernel_package.py --output-dir /tmp/ruleshift-kernel-bundle
```

## In Kaggle

1. Publish the runtime dataset artifact you intend to use.
2. Publish or update the notebook bundle built from the same repo state.
3. Open the notebook and verify the attached dataset matches the intended runtime dataset version.
4. Run the notebook normally. Do not enable any extra debug or smoke settings.

## Evidence to capture

Capture these items from the hosted run:

- notebook output showing the canonical payload block
- evidence that execution progressed past setup and into model-backed evaluation
- run duration for the full evaluation
- Kaggle credit/token usage shown by the platform, if available
- final result payload fields:
  - `score`
  - `numerator`
  - `denominator`
  - `total_episodes`
  - `benchmark_version`
  - `split`
  - `manifest_version`

## Success signals

- the notebook completes end-to-end without setup/runtime exceptions
- the final payload matches the canonical contract exactly
- `total_episodes` matches the intended evaluation scope
- runtime/latency looks materially larger than the local stub path and plausible for real model inference
- Kaggle usage/credits increase in a way consistent with real model calls

## Failure signals

- failure before the payload block appears
- immediate failure before evaluation starts
- payload shape drift
- implausibly fast completion with no observable platform usage change
- evidence that the wrong dataset version or notebook code was executed

## Post-run review

Record:

- did the run complete successfully
- were real model calls confirmed
- did latency/usage look plausible
- was any runtime drift observed
- are the results trustworthy enough for normal iteration
