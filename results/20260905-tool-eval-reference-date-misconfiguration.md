# Superseded tool-eval setup

The `20260905-final-tool-eval-{nvfp4-ostrich,fp8-kiwi}-c6.json`
runs used `--reference-date 2026-09-05`. Both completed and were persisted
by the benchmark CLI to SQLite and full-trace Markdown. Their rounded score
was 86; these are **not standard-suite qualification scores**.

Inspection of the benchmark source found that the option changes the system
reference date but the standard scenario evaluators remain hardcoded to
March dates: TC-05 expects 2026-03-23 and TC-08 expects 2026-03-21. The model
correctly scheduled September dates given the overridden system prompt,
which those evaluators rejected. TC-17 is similarly date-sensitive.

The in-progress serial continuations with the same override were interrupted
on 2026-09-05 at approximately 14:36 UTC. Completed scores and traces are
retained unchanged; no scoring or benchmark source was modified.

The replacement runs use the benchmark's standard reference date,
2026-03-20, with no `--reference-date` override. Their artifact names include
`standard-tool-eval`. Do not silently add points to the superseded runs or
compare them with the initial standard-default FP8 run.
