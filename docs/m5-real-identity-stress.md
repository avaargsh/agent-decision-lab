# M5 real-identity routing seed

This slice connects the frozen 804-tool AWS Labs MCP inventory to the Agent
Decision Benchmark without pretending synthetic intents are production traffic.

## Base labelled data

- `v2.real_identity.calibration.jsonl`: 16 cases
- `v2.real_identity.test.jsonl`: 16 cases
- four balanced gold tools, four cases each per split
- calibration and test intent text is disjoint
- every candidate and gold identity must exist in the frozen inventory

Gold identities:

1. `prometheus-mcp-server::ExecuteRangeQuery`
2. `cloudwatch-mcp-server::execute_log_insights_query`
3. `eks-mcp-server::list_k8s_resources`
4. `aws-documentation-mcp-server::search_documentation`

The intents are synthetic. Candidate identity is real and pinned to
`awslabs/mcp@5abe7e82ba70556f80d2f42558d63a20aa3ce56c`.

## Generated stress matrix

The CI bundle expands each 16-case split across:

- K = 5 / 10 / 20 / 50 / 100
- random distractors
- lexical hard negatives
- gold present
- gold intentionally missing

That produces eight JSONL outputs, 80 cases each.

The lexical arm is a dependency-free hard-negative baseline. It is not called
semantic-nearest-neighbour evidence because 339 / 804 source-extracted tools do
not currently have a usable description.

## Leakage guard

`validate_calibration_test_pair` rejects:

- incorrect split labels
- duplicate case IDs
- case-ID overlap
- identical decision context across calibration and test
- candidate or gold identities absent from the frozen inventory

The model experiment path must continue fitting temperature / selecting operating
thresholds from calibration evidence only. Test results are evaluation evidence.

## Artifact

`.github/workflows/m5-real-identity-stress.yml` rebuilds the exact upstream
inventory, verifies its 804-tool digest, generates the complete stress matrix and
uploads the inventory, JSONL suites and manifest together.
