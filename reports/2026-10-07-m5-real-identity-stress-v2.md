# M5 real-identity stress seed — retained evidence

Date: 2026-10-07

## Source identities

- upstream inventory: `awslabs/mcp@5abe7e82ba70556f80d2f42558d63a20aa3ce56c`
- canonical tools: **804**
- inventory SHA-256:
  `9e36175089bbba8714ad9e5db5efddd92812516937cf8acd3c124d61517e3d67`

## Labelled seed

The utterances are synthetic; the candidate identities are real upstream tool
identities.

- calibration: 16 cases
  - SHA-256: `5facafb4e6488a9ddd19664e1feb67b75293422755df67a15924ee16ba3dd16f`
- test: 16 cases
  - SHA-256: `168e233f611b25a8e3b50e970c60d1ea63fa4e71f78576ef4e264a0f69a91da4`
- each split is balanced across four gold tools
- calibration/test IDs and decision contexts are disjoint

## Stress matrix

Candidate counts: **5 / 10 / 20 / 50 / 100**

Strategies:

- random deterministic distractors
- lexical hard negatives

Coverage variants:

- gold present
- gold intentionally missing / expected abstain

The workflow generated eight JSONL files, **80 cases each**:

| Split | Strategy | Coverage | SHA-256 |
| --- | --- | --- | --- |
| calibration | random | covered | `7fa8451501c537dd5fcf5b06d885acb39ca5b307b7534cfe0235cb0a6c7d4cd7` |
| calibration | random | missing | `a3839f898a1edc5ff093328e1063fdac34ef628ac92939514311a6051d70ca13` |
| calibration | lexical | covered | `c21097c3f126b46884b45fde9575b1bc9866787e2a55450f50b2fede18cae5e1` |
| calibration | lexical | missing | `3da96ee2b7808494e7907637c6a82590ff39e0acccf06f4d9b5921c14e3125f4` |
| test | random | covered | `70f8d7988aad7696f0ba0d80c01d4c121c3253d04cec702389b3fb3d2abdff2e` |
| test | random | missing | `8f24957bcbb61886e39ba5dbfa3cf256eedb1aa0cfdbf8bcf5ff5231d902b9fd` |
| test | lexical | covered | `e2f7d2f03b85f34264534c42257a853551c542b736df6374ec5ba8493fe5c3a5` |
| test | lexical | missing | `a24df0e512134e5af1775171acf8830cc4463454cda768140b5e097027388e26` |

## Workflow evidence

- workflow: `m5-real-identity-stress`
- validating run: `37596026269`
- artifact: `m5-real-identity-stress-v2`
- artifact id: `11470906966`
- artifact ZIP digest:
  `sha256:eaee22a9843b2a6aa0b6aee13c7ee1bce6487b16c42119c45be25a783b327816`

The artifact contains the regenerated frozen inventory, all eight JSONL suites and
the `m5-stress-suite/v1` manifest.

## Boundaries

This closes the reproducible **candidate scaling + missing-candidate workload**
slice. It does not yet provide semantic-nearest-neighbour distractors or model
performance results.

The lexical arm remains a lexical baseline because source-derived descriptions
are incomplete. The next empirical step is to run the existing Qwen frozen-logit
and same-base structured-output adapters against this exact workload.
