# ch-executive-functions-1

This repository defines **Iron Find Electric**, a documentation-first benchmark proposal for the Executive Functions track of the Measuring Progress Toward AGI challenge. The benchmark uses short two-charge episodes to test whether a model can revise an inferred binary interaction rule after contradictory evidence, without turning the task into a general physics benchmark.

Current status: this repo contains specifications and design notes only. No benchmark implementation or dataset artifacts are included yet.

Current next milestone: build a deterministic local prototype that generates valid episodes, assigns deterministic difficulty tiers, renders Binary and Narrative prompts, parses outputs, computes `Post-shift Probe Accuracy`, runs shortcut baselines, and validates frozen schema and seed/version contracts before split freeze.

The v1 benchmark package is defined as:

- **Adaptive Rule Updating — Binary**: the only leaderboard-primary task
- **Adaptive Rule Updating — Narrative**: a required non-leaderboard robustness companion task built from the same underlying episodes

Document map:

- [iron_find_electric_implementation_spec.md](./iron_find_electric_implementation_spec.md): authoritative implementation spec and source of truth for v1 behavior
- [iron_find_electric_improved_plan.md](./iron_find_electric_improved_plan.md): aligned project plan, scope, roadmap, and validity strategy
- [benchmark_design_section_cognitive_flexibility.md](./benchmark_design_section_cognitive_flexibility.md): literature-informed framing and explicit v1 limitations

If a planning or framing document conflicts with the implementation spec, treat the implementation spec as authoritative.
