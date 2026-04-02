from __future__ import annotations

from tasks.ruleshift_benchmark.generator import generate_episode
from tasks.ruleshift_benchmark.presentation import BINARY_PRESENTATIONS
from tasks.ruleshift_benchmark.protocol import TemplateFamily
from tasks.ruleshift_benchmark.render import render_binary_prompt


def test_binary_presentations_define_all_template_families() -> None:
    assert set(BINARY_PRESENTATIONS) == set(TemplateFamily)


def test_render_binary_prompt_preserves_canonical_surface_output() -> None:
    assert render_binary_prompt(generate_episode(0)) == (
        "You are given labeled records for two markers.\n"
        "Each labeled line shows r1, r2, and the observed state.\n"
        "Use the full sequence to infer which sign combinations were revised by the later evidence, "
        "then answer the final unlabeled cases.\n"
        "\n"
        "Labeled examples:\n"
        "1. r1=-2, r2=+2 -> zark\n"
        "2. r1=+2, r2=+2 -> blim\n"
        "3. r1=-3, r2=+2 -> blim\n"
        "4. r1=-1, r2=+2 -> blim\n"
        "5. r1=+3, r2=+3 -> zark\n"
        "\n"
        "Probes:\n"
        "6. r1=-1, r2=-1 -> ?\n"
        "7. r1=+3, r2=-1 -> ?\n"
        "8. r1=+3, r2=+2 -> ?\n"
        "9. r1=-1, r2=+1 -> ?\n"
        "\n"
        "Return exactly 4 outputs in order, one per probe. "
        "Use only type_a or type_b. Map zark to type_a and blim to type_b."
    )


def test_render_binary_prompt_preserves_observation_log_surface_output() -> None:
    assert render_binary_prompt(generate_episode(9)) == (
        "Review the observation log for two markers.\n"
        "Each entry records r1, r2, and the observed state.\n"
        "Use the full log to infer which sign combinations were revised later, then answer the unlabeled probe entries.\n"
        "\n"
        "Resolved log entries:\n"
        "[01] r1=-3 | r2=+3 | observed=zark\n"
        "[02] r1=-2 | r2=+1 | observed=zark\n"
        "[03] r1=-2 | r2=+3 | observed=blim\n"
        "[04] r1=-3 | r2=-1 | observed=zark\n"
        "[05] r1=-3 | r2=+1 | observed=blim\n"
        "\n"
        "Unresolved probe entries:\n"
        "[06] r1=-2 | r2=-1 | observed=?\n"
        "[07] r1=-1 | r2=+3 | observed=?\n"
        "[08] r1=+1 | r2=+1 | observed=?\n"
        "[09] r1=+1 | r2=-2 | observed=?\n"
        "\n"
        "Return exactly 4 outputs in order, one per probe. "
        "Use only type_a or type_b. Map zark to type_a and blim to type_b."
    )


def test_render_binary_prompt_preserves_case_ledger_surface_output() -> None:
    assert render_binary_prompt(generate_episode(18)) == (
        "Review the case ledger for two markers.\n"
        "Each row records r1, r2, and the observed state.\n"
        "Use the full ledger to infer which sign combinations were revised by the later evidence, "
        "then complete the pending rows.\n"
        "\n"
        "Confirmed ledger rows:\n"
        "row 01 | r1=-1 | r2=-2 | state=blim\n"
        "row 02 | r1=+2 | r2=-1 | state=zark\n"
        "row 03 | r1=-2 | r2=+3 | state=blim\n"
        "row 04 | r1=-1 | r2=-3 | state=zark\n"
        "row 05 | r1=-2 | r2=-1 | state=zark\n"
        "\n"
        "Pending ledger rows:\n"
        "row 06 | r1=+3 | r2=-3 | state=?\n"
        "row 07 | r1=+1 | r2=+2 | state=?\n"
        "row 08 | r1=-1 | r2=+1 | state=?\n"
        "row 09 | r1=-1 | r2=-1 | state=?\n"
        "\n"
        "Return exactly 4 outputs in order, one per probe. "
        "Use only type_a or type_b. Map zark to type_a and blim to type_b."
    )
