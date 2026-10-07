# Upstream MCP inventory extraction

M5 uses source-derived inventories rather than manually curated tool-name lists.

The first reproducibility target is the public `awslabs/mcp` repository pinned
to commit:

`5abe7e82ba70556f80d2f42558d63a20aa3ce56c`

The extractor walks Python source under `src/**` and recognizes:

- `@server.tool()` using the decorated function name
- `@server.tool(name="...")`
- `_register_tool("...", callable)`
- `server.tool(name="...")(callable)`

Tests and integration fixtures are excluded.

## Identity

Cross-server tool names can collide, so benchmark inventory identity is:

`<server-directory>::<upstream-tool-name>`

The original tool name and source path are preserved in tool metadata.

## Evidence workflow

`.github/workflows/mcp-inventory-snapshot.yml` checks out the exact upstream
commit, runs the AST extractor, verifies the content digest, requires at least
100 extracted tools, and uploads the generated JSON inventory as a workflow
artifact.

The workflow artifact proves extraction from a pinned upstream tree. It should
be reviewed before a snapshot is promoted into `benchmarks/` as a retained
benchmark input.

This is intentionally source-derived evidence, not a live MCP handshake. A later
track may add runtime `tools/list` capture and compare it against the source
snapshot.
