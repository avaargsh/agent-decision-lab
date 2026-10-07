# AWS Labs MCP source inventory — frozen evidence

Date: 2026-10-07

## Source

- repository: `awslabs/mcp`
- revision: `5abe7e82ba70556f80d2f42558d63a20aa3ce56c`
- extraction: `python-ast-mcp-tool-v1`
- inventory id: `awslabs-mcp@5abe7e82ba70`

## Frozen result

- extracted canonical tools: **804**
- contributing server directories: **50**
- inventory SHA-256:
  `9e36175089bbba8714ad9e5db5efddd92812516937cf8acd3c124d61517e3d67`
- first validating workflow run: `37594990765`
- workflow artifact id: `11470431640`

The workflow regenerates the inventory from the pinned upstream commit and now
requires the exact count and digest above. Extractor changes therefore surface as
an explicit benchmark-input change rather than silently changing M5 results.

## Identity rule

Raw MCP tool names are not globally unique across servers. The snapshot uses:

`<server-directory>::<upstream-tool-name>`

The original upstream name and exact source path remain in metadata.

## Manual artifact inspection

The first artifact was materialized and inspected after the workflow succeeded:

- 804 tools
- 50 server directories
- no extracted source paths under `tests/` or `integration/`
- repeated raw names exist across different servers (for example database
  operations), which validates the need for server-qualified canonical identity
- 339 / 804 tools have no description recoverable from the supported decorator /
  registration patterns

The last point is important: this inventory is strong evidence for **candidate
identity and scale**, but weaker evidence for description-based semantic-nearest
negative construction. The current dependency-free `lexical` distractor mode
must remain labeled lexical. A future semantic-hard-negative track should enrich
descriptions from runtime `tools/list`, server schemas, or another explicit
evidence source.

## What this proves

This closes the source-derived inventory prerequisite for M5:

- immutable upstream revision
- reproducible extraction rule
- content-addressed inventory identity
- enough real candidates to exercise K=5/10/20/50/100 stress

It does **not** prove that source extraction exactly matches the live runtime MCP
`tools/list` surface. Runtime parity remains a separate experiment.
