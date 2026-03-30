from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RUN_ID_ENV_VAR = "RULESHIFT_RUN_ID"
RUN_OUTPUT_DIR_ENV_VAR = "RULESHIFT_RUN_OUTPUT_DIR"


@dataclass(frozen=True, slots=True)
class BenchmarkRunContext:
    run_id: str
    output_dir: Path
    provider: str
    model: str


def build_run_context(
    *,
    repo_root: Path | str | None = None,
    llm: object | None = None,
    run_id: str | None = None,
    output_dir: Path | str | None = None,
) -> BenchmarkRunContext:
    resolved_run_id = _resolve_run_id(run_id)
    resolved_output_dir = _resolve_output_dir(
        repo_root=repo_root,
        run_id=resolved_run_id,
        output_dir=output_dir,
    )
    provider, model = _resolve_provider_model(llm)
    return BenchmarkRunContext(
        run_id=resolved_run_id,
        output_dir=resolved_output_dir,
        provider=provider,
        model=model,
    )


def _resolve_run_id(run_id: str | None) -> str:
    if run_id is not None and run_id.strip():
        return run_id.strip()
    env_run_id = os.getenv(RUN_ID_ENV_VAR, "").strip()
    if env_run_id:
        return env_run_id
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"ruleshift-{timestamp}-{uuid.uuid4().hex[:8]}"


def _resolve_output_dir(
    *,
    repo_root: Path | str | None,
    run_id: str,
    output_dir: Path | str | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()

    env_output_dir = os.getenv(RUN_OUTPUT_DIR_ENV_VAR, "").strip()
    if env_output_dir:
        return Path(env_output_dir).resolve()

    kaggle_working_dir = Path("/kaggle/working")
    if kaggle_working_dir.exists():
        return kaggle_working_dir / run_id

    if repo_root is None:
        repo_root = Path.cwd()

    return Path(repo_root).resolve() / "reports" / "local" / run_id


def _resolve_provider_model(llm: object | None) -> tuple[str, str]:
    provider = _resolve_identity_field(
        llm,
        "provider",
        "provider_name",
        env_var="RULESHIFT_PROVIDER",
    )
    model = _resolve_identity_field(
        llm,
        "model",
        "model_name",
        env_var="RULESHIFT_MODEL",
    )
    return provider, model


def _resolve_identity_field(
    llm: object | None,
    *attribute_names: str,
    env_var: str,
) -> str:
    for attribute_name in attribute_names:
        try:
            value = getattr(llm, attribute_name)
        except Exception:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()

    env_value = os.getenv(env_var, "").strip()
    if env_value:
        return env_value

    return "unknown"
