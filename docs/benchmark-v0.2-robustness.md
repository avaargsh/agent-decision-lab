# Decision Benchmark v0.2 robustness protocol

This protocol extends the existing `decision-eval/v1` evidence path without
changing its schema. The v1 artifact remains the stable release-gate contract;
robustness measurements are sidecar benchmark evidence until their semantics
stabilize.

## 1. Candidate-order robustness

For every bounded decision, score the original candidate order once and then
score deterministic order perturbations:

- reverse
- rotate-left-by-one when it is distinct from reverse

Report:

- baseline accuracy
- perturbed accuracy
- permutation flip rate

`permutation_flip_rate` is the fraction of perturbed evaluations whose top-1
candidate differs from the original top-1 candidate. It detects position/label
sensitivity even when aggregate accuracy happens to remain unchanged.

## 2. Missing-candidate and OOD abstention

Cases that should not be autonomously executed carry:

```json
{"metadata": {"expected_abstain": true, "shift": "missing_candidate"}}
```

or an equivalent `shift` such as `ood`.

The gold candidate may intentionally be absent from the presented candidate set
for these cases. They must be evaluated with the robustness evaluator, not the
ordinary closed-set benchmark runner.

At threshold `t`:

- accept when `max_probability >= t`
- abstain otherwise

Report:

- FAR / `false_accept_rate`: accepted cases among cases expected to abstain
- FRR / `false_rejection_rate`: rejected cases among covered cases
- covered accuracy
- automation coverage
- unsafe automation rate

The existing `CoveragePoint.false_automation_rate` is preserved for backward
compatibility. It measures accepted classification errors over the full
closed-set population and is not the same denominator as OOD/missing-candidate
FAR.

## 3. Threshold transfer

Choose an operating threshold on a declared source calibration/validation
distribution. Reuse that exact threshold on a target distribution without
refitting.

Report source -> target deltas for:

- coverage
- selective risk
- false automation rate
- ECE

This directly tests whether an apparently safe IID operating point transfers to
another domain, tool inventory, or candidate construction policy.

## 4. Next benchmark expansions

The next data work should add, in order:

1. candidate-count scaling: K = 5 / 10 / 20 / 50 / 100
2. semantic-nearest-neighbour distractors
3. frozen real MCP inventory snapshots
4. explicit missing-tool and unsupported-capability cases
5. cross-domain threshold-transfer suites

Do not claim production routing quality from the current synthetic seed corpus.
