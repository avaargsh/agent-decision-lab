# M5 semantic hard-negative contract

M5 already has deterministic random and lexical distractors. The semantic arm is
defined separately so it cannot silently fall back to lexical similarity.

## Definition

For the first semantic stress arm, a hard negative is a tool whose embedding is
nearest to the **gold tool identity** in the frozen inventory.

This definition is intentional:

- the candidate universe is stable and independently content-addressed
- ranking does not depend on synthetic query wording
- the same semantic neighbourhood can be reused across calibration and test
  without fitting on either split
- candidate construction remains independent from the model being evaluated

It is not a claim that nearest-to-gold is the only useful semantic stress
definition. Query-to-tool retrieval can be added later as a separate experiment.

## Embedding snapshot

The contract is `semantic-tool-embeddings/v1`.

A snapshot records:

- `embedding_id`
- exact source inventory id and SHA-256
- embedding provider/model/revision metadata
- deterministic text recipe
- one non-zero, finite vector for **every** tool in the inventory
- declared dimension
- canonical snapshot SHA-256

Partial coverage fails closed. A semantic stress suite may not silently mix
semantic and lexical/random fallbacks.

The current text recipe is:

`tool-identity-description-source-v1`

It retains canonical name, server, upstream tool name when available,
description when available, and source path when available. This means tools
without extracted descriptions still have a stable identity/source text surface.

## Ranking

`semantic_gold_neighbor_ranker` computes cosine similarity from the gold tool
embedding to every distractor embedding and orders by:

1. descending cosine similarity
2. canonical tool name for deterministic tie-breaking

The existing stress generator then takes the top K-1 distractors and applies the
same deterministic candidate-order shuffle used by other strategies.

## Provenance

When `semantic` is requested, `m5-stress-suite/v2` records:

- embedding id
- embedding snapshot SHA-256
- embedding dimension
- model metadata
- text recipe

Example future invocation:

```bash
python scripts/build_m5_stress_artifacts.py \
  --inventory artifacts/awslabs-mcp.inventory.json \
  --calibration benchmarks/mcp_tool_router/v2.real_identity.calibration.jsonl \
  --test benchmarks/mcp_tool_router/v2.real_identity.test.jsonl \
  --candidate-counts 5,10,20,50,100 \
  --strategies random,lexical,semantic \
  --semantic-embeddings artifacts/awslabs-mcp.semantic-embeddings.json \
  --output-dir artifacts/m5
```

## Current boundary

This PR freezes the **consumer contract and ranking semantics only**.

No real 804-tool embedding snapshot is claimed yet. The eventual embedding
snapshot must be generated with an exact model revision and retained by digest;
until then the roadmap item "semantic-nearest-neighbour distractor construction"
remains open.
