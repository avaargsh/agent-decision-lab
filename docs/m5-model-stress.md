# M5 model stress experiment

The first measured M5 experiment reuses the existing two model paths:

1. Qwen frozen candidate continuation scoring
2. same-base autoregressive structured output

## Frozen prompt variant

The original frozen-logit baseline lists every candidate inside the prompt and
then scores every candidate continuation. That is acceptable at K=4 but causes
avoidable prompt-token growth when K reaches 50 or 100.

M5 therefore uses `compact_candidate_prompt`:

- decision type and context stay in the prompt
- the complete candidate list is not duplicated in the prompt
- the bounded set is still enforced by scoring only the explicit candidate
  continuations
- the report records `compact-candidate-v1` as experiment provenance

The original prompt remains unchanged for historical v1 artifacts.

## Calibration and transfer

For each adapter:

1. score the 16-case calibration seed
2. fit temperature on calibration only
3. select a confidence threshold on calibration only using the declared observed
   risk budget
4. if no non-empty risk-budget operating point exists, use the calibration
   median confidence and record that fallback method
5. reuse the exact threshold for K=5/10/20/50/100 without refitting

For every K the report records:

- Accuracy / Macro-F1 / NLL / Brier / ECE on covered cases
- Risk-Coverage at the transferred threshold
- FAR on missing-candidate cases
- FRR on covered cases
- threshold-transfer deltas
- latency / token accounting from the benchmark runner
- permutation flip rate on a fixed small sample

The initial workflow uses deterministic random distractors. Lexical/semantic hard
negative model runs remain follow-up experiments, not mixed into this first
measurement.
