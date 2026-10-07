from __future__ import annotations

from decision_lab.inventory_extract import (
    build_inventory_from_python_tree,
    extract_python_mcp_tools,
)


def _write(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_extracts_decorator_function_name_and_explicit_name(tmp_path) -> None:
    _write(
        tmp_path,
        "src/demo-server/pkg/server.py",
        '''
from somewhere import FastMCP

mcp = FastMCP("demo")

@mcp.tool()
async def list_widgets():
    """List widgets from the service."""
    return []

@mcp.tool(name="get_widget", description="Fetch one widget")
def internal_get_widget():
    return {}
''',
    )

    tools = extract_python_mcp_tools(tmp_path)

    assert [tool.canonical_name for tool in tools] == [
        "demo-server::get_widget",
        "demo-server::list_widgets",
    ]
    by_name = {tool.tool_name: tool for tool in tools}
    assert by_name["list_widgets"].description == (
        "List widgets from the service."
    )
    assert by_name["get_widget"].description == "Fetch one widget"


def test_extracts_registration_patterns_and_deduplicates(tmp_path) -> None:
    _write(
        tmp_path,
        "src/demo-server/pkg/registration.py",
        '''
def alpha():
    pass

_register_tool("AlphaTool", alpha)
mcp.tool(name="BetaTool", description="Beta description")(alpha)
mcp.tool(name="BetaTool", description="Beta description")(alpha)
''',
    )

    tools = extract_python_mcp_tools(tmp_path)

    assert [tool.canonical_name for tool in tools] == [
        "demo-server::AlphaTool",
        "demo-server::BetaTool",
    ]


def test_skips_tests_and_binds_source_revision(tmp_path) -> None:
    _write(
        tmp_path,
        "src/demo-server/pkg/server.py",
        '''
@mcp.tool()
def live_tool():
    """Live tool."""
    pass
''',
    )
    _write(
        tmp_path,
        "src/demo-server/tests/test_server.py",
        '''
@mcp.tool()
def fake_test_tool():
    pass
''',
    )

    inventory = build_inventory_from_python_tree(
        tmp_path,
        inventory_id="demo@abc",
        repository="example/demo",
        revision="abc",
    )

    assert len(inventory.tools) == 1
    assert inventory.tools[0].name == "demo-server::live_tool"
    assert inventory.tools[0].metadata["upstream_tool_name"] == "live_tool"
    assert inventory.source["repository"] == "example/demo"
    assert inventory.source["revision"] == "abc"
    assert inventory.source["extraction"] == "python-ast-mcp-tool-v1"
