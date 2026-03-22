from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

from core.gemini_panel import (
    default_gemini_first_panel_report_path,
    run_gemini_first_panel,
)
from core.model_execution import ModelMode, ModelRawResult, ModelRequest, ModelRunConfig
from generator import generate_episode
from protocol import InteractionLabel, Split
from render import render_binary_prompt, render_narrative_prompt


def _labels_text(labels: tuple[InteractionLabel, ...]) -> str:
    return ", ".join(label.value for label in labels)


def _narrative_text(labels: tuple[InteractionLabel, ...]) -> str:
    return "Reasoning.\n" + _labels_text(labels)


def _wrong_labels(episode) -> tuple[InteractionLabel, ...]:
    first_label = episode.probe_targets[0]
    flipped_first_label = (
        InteractionLabel.REPEL
        if first_label is InteractionLabel.ATTRACT
        else InteractionLabel.ATTRACT
    )
    return (flipped_first_label, *episode.probe_targets[1:])


@dataclass
class FakeGeminiAdapter:
    responses: dict[tuple[ModelMode, str], ModelRawResult]

    def generate(self, request: ModelRequest, config: ModelRunConfig) -> ModelRawResult:
        del config
        return self.responses[(request.mode, request.prompt_text)]


def test_run_gemini_first_panel_writes_paired_artifact_and_report(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "core.gemini_panel.current_report_timestamp",
        lambda: "20260322_201900",
    )
    episodes_by_split = {
        "dev": (
            generate_episode(0, split=Split.DEV),
            generate_episode(1, split=Split.DEV),
        ),
        "public_leaderboard": (
            generate_episode(100, split=Split.PUBLIC),
            generate_episode(101, split=Split.PUBLIC),
        ),
        "private_leaderboard": (
            generate_episode(200, split=Split.PRIVATE),
            generate_episode(201, split=Split.PRIVATE),
        ),
    }
    responses: dict[tuple[ModelMode, str], ModelRawResult] = {}
    for split_name, episodes in episodes_by_split.items():
        for index, episode in enumerate(episodes):
            binary_prompt = render_binary_prompt(episode)
            binary_request = ModelRequest(
                provider_name="gemini",
                model_name="fake-model",
                prompt_text=binary_prompt,
                mode=ModelMode.BINARY,
            )
            responses[(ModelMode.BINARY, binary_prompt)] = ModelRawResult.from_request(
                binary_request,
                response_text=_labels_text(episode.probe_targets),
            )

            narrative_prompt = render_narrative_prompt(episode)
            narrative_request = ModelRequest(
                provider_name="gemini",
                model_name="fake-model",
                prompt_text=narrative_prompt,
                mode=ModelMode.NARRATIVE,
            )
            if split_name == "dev" and index == 1:
                responses[(ModelMode.NARRATIVE, narrative_prompt)] = (
                    ModelRawResult.from_request(
                        narrative_request,
                        error_type="TimeoutError",
                        error_message="timed out",
                    )
                )
                continue
            elif split_name == "public_leaderboard" and index == 0:
                response_text = _narrative_text(_wrong_labels(episode))
            elif split_name == "private_leaderboard" and index == 1:
                response_text = "Reasoning without a valid final line."
            else:
                response_text = _narrative_text(episode.probe_targets)
            responses[(ModelMode.NARRATIVE, narrative_prompt)] = (
                ModelRawResult.from_request(
                    narrative_request,
                    response_text=response_text,
                )
            )

    monkeypatch.setattr(
        "core.gemini_panel.load_frozen_split",
        lambda split_name: tuple(
            SimpleNamespace(episode=episode) for episode in episodes_by_split[split_name]
        ),
    )

    report_path = tmp_path / "m1_binary_vs_narrative_robustness_report.md"
    artifacts = run_gemini_first_panel(
        model_name="fake-model",
        report_path=report_path,
        modes=(ModelMode.BINARY, ModelMode.NARRATIVE),
        adapter=FakeGeminiAdapter(responses),
    )

    assert artifacts.report_path == report_path
    assert artifacts.artifact_path == report_path.with_suffix(".json")
    assert artifacts.snapshot_report_path == tmp_path / "m1_binary_vs_narrative_robustness_report__20260322_201900.md"
    assert artifacts.snapshot_artifact_path == tmp_path / "m1_binary_vs_narrative_robustness_report__20260322_201900.json"
    assert artifacts.report_path.read_text(encoding="utf-8") == artifacts.report_markdown
    assert (
        artifacts.snapshot_report_path.read_text(encoding="utf-8")
        == artifacts.report_markdown
    )

    payload = json.loads(artifacts.artifact_path.read_text(encoding="utf-8"))
    assert payload["prompt_modes"] == ["binary", "narrative"]
    assert [split["split_name"] for split in payload["splits"]] == [
        "dev",
        "public_leaderboard",
        "private_leaderboard",
    ]
    for split_payload in payload["splits"]:
        assert split_payload["pairing_checks"] == {
            "same_episode_order": True,
            "same_probe_targets": True,
        }
        for row in split_payload["rows"]:
            assert sorted(row["modes"]) == ["binary", "narrative"]

    narrative_overall = next(
        row
        for row in payload["failure_taxonomy"]
        if row["scope"] == "overall" and row["mode"] == "Narrative"
    )
    assert narrative_overall["runtime_error_rate"] > 0.0
    assert narrative_overall["parse_failure_rate"] > 0.0
    assert narrative_overall["adaptation_failure_rate"] > 0.0

    first_dev_row = payload["splits"][0]["rows"][0]
    assert first_dev_row["modes"]["binary"]["finish_reason"] is None

    assert "## Paired Robustness" in artifacts.report_markdown
    assert "## Failure Taxonomy" in artifacts.report_markdown
    assert "| Scope | Mode | Provider/runtime error rate |" in artifacts.report_markdown
    assert "Provider/runtime failures were observed in the live run." in artifacts.report_markdown
    assert "Binary-only headline metric: fake-model Binary =" in artifacts.report_markdown
    assert "Binary vs Narrative comparison is unavailable" not in artifacts.report_markdown


def test_default_gemini_first_panel_report_paths_are_grouped_by_target():
    binary_path = default_gemini_first_panel_report_path(include_narrative=False)
    paired_path = default_gemini_first_panel_report_path(include_narrative=True)

    assert binary_path.as_posix().endswith(
        "reports/live/gemini-first-panel/binary-only/latest/report.md"
    )
    assert paired_path.as_posix().endswith(
        "reports/live/gemini-first-panel/binary-vs-narrative/latest/report.md"
    )
