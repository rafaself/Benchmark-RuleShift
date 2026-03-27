# Kaggle Deployment Runbook

## Required Secrets

| Secret | Scope | Purpose |
|---|---|---|
| `KAGGLE_API_TOKEN` | Repository secret | Authenticates all `kaggle` CLI calls |

Set at: **GitHub → Settings → Secrets and variables → Actions → Repository secrets**

---

## Canonical Values

These values are locked. Do not change them without a coordinated update to both the workflow and the matching build script constant.

| Artifact | Canonical id |
|---|---|
| Runtime dataset | `raptorengineer/ruleshift-runtime` |
| Notebook kernel | `raptorengineer/ruleshift-notebook-task` |

---

## Deploy Dataset Only

**Workflow:** `deploy-kaggle-dataset.yml`
**Trigger:** Actions → Deploy — Kaggle Runtime Dataset → Run workflow

| Input | Value |
|---|---|
| `environment` | `staging` or `production` |
| `dataset_id` | `raptorengineer/ruleshift-runtime` |
| `dataset_title` | `RuleShift Runtime` |
| `release_message` | Describe the change (e.g. `R16 split manifests`) |

**Behavior:**
1. Runs validation gates (isolation check, packaging tests, CD build tests).
2. Builds the runtime package via `scripts/cd/build_runtime_dataset_package.py`.
3. Checks whether the dataset exists on Kaggle.
   - First time: `kaggle datasets create`
   - Subsequent: `kaggle datasets version -m <release_message>`

---

## Deploy Notebook Only

**Prerequisite:** The runtime dataset (`raptorengineer/ruleshift-runtime`) must already be published.

**Workflow:** `deploy-kaggle-notebook.yml`
**Trigger:** Actions → Deploy — Kaggle Notebook → Run workflow

| Input | Value |
|---|---|
| `environment` | `staging` or `production` |
| `kernel_id` | `raptorengineer/ruleshift-notebook-task` |
| `kernel_title` | `RuleShift Notebook Task — Cognitive Flexibility Benchmark` |
| `runtime_dataset_slug` | `raptorengineer/ruleshift-runtime` |

**Behavior:**
1. Runs validation gates (isolation check, packaging tests, CD build tests, notebook smoke tests).
2. Builds the kernel bundle via `scripts/cd/build_kernel_package.py --runtime-dataset-slug`.
3. Runs `kaggle kernels push` (always an upsert — creates on first push, updates on subsequent).

---

## Deploy Both (Dataset then Notebook)

**Workflow:** `deploy-kaggle-orchestrator.yml`
**Trigger:** Actions → Deploy — Kaggle Orchestrator → Run workflow

| Input | Value |
|---|---|
| `target` | `all` |
| `environment` | `staging` or `production` |

No other inputs. All canonical values are hardcoded in the orchestrator.

**Behavior:**
1. Runs dataset deployment to completion.
2. On success, runs notebook deployment.
3. If dataset deployment fails, notebook deployment does not start.

---

## Deploying Individual Targets via Orchestrator

Set `target` to `dataset` or `notebook` to deploy only that artifact.
`target=notebook` does **not** trigger dataset deployment first.

---

## Expected Workflow Behavior

| Condition | Result |
|---|---|
| `dataset_id` or `kernel_id` input differs from canonical value | Job aborts before build, no artifact is produced |
| Private artifact detected in `src/` during build | Build script aborts, deploy job fails |
| Validation gates fail | Deploy job never starts (`needs: validate`) |
| Dataset does not yet exist on Kaggle | `kaggle datasets create` is used automatically |
| Kaggle credentials file left on runner after failure | Removed by `Revoke Kaggle credentials` step (`if: always()`) |
| Two orchestrated deploys triggered for the same environment | Second run waits; it is not cancelled |
| Direct trigger of a dedicated workflow while orchestrator is running | Not blocked — concurrency groups are independent |
