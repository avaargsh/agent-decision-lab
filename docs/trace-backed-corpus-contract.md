# Trace-backed routing corpus contract

Schema: `trace-backed-routing/v1`

This contract defines the minimum provenance and release discipline required
before a routing benchmark may be described as **trace-backed**.

It does not create or claim any production data. No raw trace dataset is
included in the repository by this contract.

## Terminology

### Source-grounded authored

The existing v3 corpus uses real upstream MCP tool contracts with authored
requests. It is **not** trace-backed.

### Trace-backed

A released benchmark case has documented lineage to an independently existing
source such as:

- production trace
- incident record
- user study
- support case

A trace-backed case may still be semantically rewritten after sanitization.
Therefore `trace-backed` does not automatically mean verbatim production text.

### Verbatim sanitized trace

Only cases with:

`transformation=verbatim_sanitized`

may be described as verbatim sanitized trace text.

Cases with:

`transformation=semantic_rewrite`

retain source lineage but must be described as rewritten trace-backed cases.

Do not use "production traffic" as a blanket label for a corpus that contains
semantic rewrites.

## Public release boundary

Raw source records should remain outside the public benchmark repository.

The released case stores lineage using digests and pseudonyms:

- `source_artifact_sha256`: content digest of the controlled source artifact
- `source_record_sha256`: pseudonymous record identity
- `source_group_sha256`: pseudonymous leakage-group identity
- `released_case_sha256`: digest of the exact released benchmark content

Record/group identifiers must declare:

`identifier_pseudonymization=salted_sha256`

The salt must remain outside the public repository. Do not hash predictable
low-entropy account, ticket, incident, user, or session identifiers without a
non-public salt.

The source group should be the **largest unit likely to leak task semantics**.
Examples include a complete incident, conversation, session, support case, or
study participant. When uncertain, choose the coarser grouping.

## Required sanitization declaration

Every released case must declare all of the following as true:

- `pii_removed`
- `secrets_removed`
- `customer_identifiers_removed`
- `free_text_reviewed`

The validator treats these as release assertions, not as proof that automated
redaction was perfect. Human/process review remains necessary.

Likewise:

`benchmark_release_approved=true`

records that the producing workflow approved the sanitized case for benchmark
release. It is not a substitute for organizational privacy, security, legal, or
data-governance review.

## Label provenance

Each case declares one label source:

- `observed_tool_selection`
- `human_adjudication`
- `upstream_contract_relabel`

At least one reviewer is required.

Cases marked `underspecified` or `multi_valid` require at least two reviewers,
because ambiguity labels should not be created from a single arbitrary opinion.

Existing `mcp-routing-label/v1` risk, ambiguity, coverage, and abstention
semantics still apply.

## Released-content binding

`released_case_sha256` is computed from:

- decision type
- sanitized released context
- candidate identities
- gold candidate or null

Changing released benchmark content after provenance is generated invalidates the
digest and fails validation.

This prevents the source lineage from silently pointing at a different benchmark
example after editing.

## Split leakage

Calibration and test must not share:

1. `source_group_sha256`
2. `source_record_sha256`
3. exact released-case fingerprint

This is stricter than simple case-ID uniqueness.

For example, two different requests from the same incident must stay in the same
split even when their text and record IDs differ.

Split assignment should happen at source-group level before final benchmark
evaluation.

## Calibration discipline

Trace-backed calibration data must remain single-gold and must not contain
expected-abstain cases.

Abstention/ambiguous trace-backed examples belong in held-out evaluation data,
not in the data used to fit temperature or select an operating threshold.

## Validation

Future retained trace-backed corpora should run:

```bash
python scripts/validate_trace_backed_corpus.py \
  --inventory artifacts/awslabs-mcp.inventory.json \
  --calibration <sanitized-calibration.jsonl> \
  --test <sanitized-test.jsonl>
```

The validator checks:

- frozen inventory identities
- calibration/test context separation
- routing annotations
- trace provenance
- sanitization declarations
- released-content digests
- source-record leakage
- source-group leakage

## Non-goals

This contract does not:

- collect production traces
- authorize access to private data
- define a universal privacy policy
- make source-grounded authored prompts "real traces"
- allow raw secrets/logs/customer identifiers into the public repository
- infer consent or release rights from a checksum

The roadmap item for a production/trace-backed labeled corpus remains open until
actual independently sourced, sanitized, release-approved cases are retained and
validated.
