# Local Kaggle-like Validation Environment

This project includes a minimal Docker setup that uses the official Kaggle Python image as the base runtime for local development and debugging.

## Files

- `Dockerfile.kaggle-local`
- `docker-compose.kaggle-local.yml`
- `scripts/bootstrap_kaggle_local.sh`

## Start an interactive shell

Build the image and start a shell in the project root:

```bash
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local
```

The repository is bind-mounted at `/workspace/ruleshift`, and the container starts in that directory.
The first run can take a long time because the Kaggle image is very large.

## Run one-off commands

Print the Python version:

```bash
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local python --version
```

Print the working directory:

```bash
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local python -c "from pathlib import Path; print(Path.cwd())"
```

Confirm core imports:

```bash
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local python -c "import core.kaggle.runner; import core.splits; print(core.kaggle.runner.__file__)"
```

Run a relevant test file:

```bash
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local python -m pytest tests/test_kaggle_execution.py -v
```

Run the local preflight gate:

```bash
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local python scripts/preflight_kaggle.py
```

Run the safe pre-deploy gate:

```bash
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local ./scripts/pre_deploy_check.sh
```

Run a local script from the project root:

```bash
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local python scripts/build_runtime_dataset_package.py --output-dir /tmp/ruleshift-runtime-package
```

## Import behavior

The container bootstrap installs:

- `requirements-dev.txt`
- the repository itself with `pip install -e .`

That keeps imports explicit and container-local without relying on host-side virtual environments or host path configuration.

## Validation Summary

- Base image: `gcr.io/kaggle-images/python:v161`
- Intended workdir: `/workspace/ruleshift`
- Intended import path behavior: editable install inside the container, so `core.*` and `tasks.*` imports resolve cleanly from the mounted repository
- Validation status in this environment: container definition completed, but first-time runtime validation did not finish because pulling the Kaggle base image required several multi-gigabyte layers and exceeded practical turn time
- Verified during setup: Docker resolved the pinned Kaggle base image and began building from `gcr.io/kaggle-images/python:v161`
- Not yet verified end-to-end here: `python --version`, `Path.cwd()`, `import core.kaggle.runner`, and `python -m pytest tests/test_kaggle_execution.py -v` after the full image pull completes

## Safe Deploy Workflow

Run the local gate before any Kaggle publish step:

```bash
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local ./scripts/pre_deploy_check.sh
```

Only after that passes, run the actual packaging or deploy action:

```bash
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local python scripts/build_runtime_dataset_package.py --output-dir /tmp/ruleshift-runtime-package
docker compose -f docker-compose.kaggle-local.yml run --rm kaggle-local python scripts/build_kernel_package.py --output-dir /tmp/ruleshift-kernel-bundle
```

The deploy gate reduces runtime drift risk by rebuilding both public artifacts locally before release. Those build scripts already verify canonical metadata and notebook/split-manifest hash consistency against `packaging/kaggle/frozen_artifacts_manifest.json`.

## Known Limits

- This mirrors the Kaggle Python runtime image, but not the full hosted notebook environment.
- The first pull is very heavy because the upstream Kaggle image includes the notebook stack, not just Python.
- Kaggle-specific mounts such as `/kaggle/input` and `/kaggle/working` are not reproduced yet.
- GPU, internet policy, notebook orchestration, and Kaggle UI behavior are intentionally out of scope for this phase.
