"""RuleShift's Kaggle runtime surface.

Read the runtime in this order:
1. `kaggle/ruleshift_notebook_task.ipynb`
2. `tasks.ruleshift_benchmark` for the package-level entrypoints used there
3. `tasks.ruleshift_benchmark.splits` for frozen data loading and prompt assembly
4. `tasks.ruleshift_benchmark.runner` for response normalization and scoring

`protocol.py` and `schema.py` hold the frozen episode contract that the runtime
builds and scores.
"""

from tasks.ruleshift_benchmark.runner import (
    BinaryResponse,
    KaggleExecutionError,
    normalize_binary_response,
    run_binary_task,
    score_episode,
)
from tasks.ruleshift_benchmark.splits import (
    MANIFEST_VERSION,
    PRIVATE_DATASET_ROOT_ENV_VAR,
    TASK_NAME,
    build_benchmark_bundle,
    build_leaderboard_rows,
    discover_private_dataset_root,
    render_binary_prompt,
    resolve_private_dataset_root,
)

__all__ = [
    "BinaryResponse",
    "KaggleExecutionError",
    "MANIFEST_VERSION",
    "PRIVATE_DATASET_ROOT_ENV_VAR",
    "TASK_NAME",
    "build_benchmark_bundle",
    "build_leaderboard_rows",
    "discover_private_dataset_root",
    "normalize_binary_response",
    "render_binary_prompt",
    "resolve_private_dataset_root",
    "run_binary_task",
    "score_episode",
]
