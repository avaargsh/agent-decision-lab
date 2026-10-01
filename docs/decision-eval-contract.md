# Decision Eval Contract

The release-facing unit is a content-addressed `decision-eval/v1` artifact, not a
free-form benchmark report.

A comparable evaluation must carry:

- exact test dataset digest and case IDs;
- adapter and model identity;
- calibration digest/provenance when calibration is used;
- base accuracy/calibration/latency/token metrics;
- selected operating point and risk budget when selective automation is used;
- measured System-2 fallback evidence when fallback claims are made;
- a content digest that covers the complete artifact.

## Cross-adapter matrix

`decision_lab.contract_matrix.build_contract_matrix()` accepts multiple verified
artifacts and refuses comparison unless all artifacts use:

1. the same `decision-eval` schema version;
2. the same bounded decision type;
3. the exact same final test dataset digest and ordered case IDs.

Calibration inputs may differ between adapters, but their provenance remains
visible in every matrix row.

This creates one comparison boundary for:

- frozen candidate logits;
- autoregressive Structured Output;
- Decision LoRA;
- dedicated decision heads;
- encoder baselines.

New model families should add adapters and produce the same artifact rather than
adding model-specific benchmark schemas.

## What the matrix does not prove

A shared artifact contract makes measurements comparable. It does not make a
small synthetic corpus representative of production traffic, and it does not
turn one metric into an overall model ranking. Dataset quality, ambiguity/risk
annotation, and production operating-point selection remain separate evidence
requirements.
