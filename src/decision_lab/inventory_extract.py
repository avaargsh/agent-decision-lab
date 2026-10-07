from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .inventory import ToolInventory, ToolSpec, make_inventory


@dataclass(frozen=True)
class ExtractedTool:
    server: str
    tool_name: str
    description: str
    source_path: str

    @property
    def canonical_name(self) -> str:
        return f"{self.server}::{self.tool_name}"


def extract_python_mcp_tools(
    root: str | Path,
    *,
    include_glob: str = "src/**/*.py",
) -> list[ExtractedTool]:
    root_path = Path(root)
    extracted: dict[str, ExtractedTool] = {}

    for path in sorted(root_path.glob(include_glob)):
        if not path.is_file():
            continue
        relative = path.relative_to(root_path)
        parts = relative.parts
        if len(parts) < 3 or parts[0] != "src":
            continue
        if "tests" in parts or "integration" in parts:
            continue

        server = parts[1]
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(relative),
            )
        except (SyntaxError, UnicodeDecodeError):
            continue

        for tool in _extract_from_tree(
            tree,
            server=server,
            source_path=str(relative),
        ):
            extracted.setdefault(tool.canonical_name, tool)

    return [
        extracted[name]
        for name in sorted(extracted)
    ]


def build_inventory_from_python_tree(
    root: str | Path,
    *,
    inventory_id: str,
    repository: str,
    revision: str,
    source_kind: str = "github-source-tree",
) -> ToolInventory:
    extracted = extract_python_mcp_tools(root)
    if not extracted:
        raise ValueError("no MCP tools found in source tree")

    tools = [
        ToolSpec(
            name=tool.canonical_name,
            description=tool.description,
            server=tool.server,
            metadata={
                "upstream_tool_name": tool.tool_name,
                "source_path": tool.source_path,
            },
        )
        for tool in extracted
    ]
    return make_inventory(
        inventory_id=inventory_id,
        tools=tools,
        source={
            "kind": source_kind,
            "repository": repository,
            "revision": revision,
            "extraction": "python-ast-mcp-tool-v1",
        },
    )


def _extract_from_tree(
    tree: ast.AST,
    *,
    server: str,
    source_path: str,
) -> Iterable[ExtractedTool]:
    seen: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if not _is_tool_decorator(decorator):
                    continue
                tool_name = _tool_name_from_call(
                    decorator,
                    default=node.name,
                )
                description = _description_from_call(decorator)
                if not description:
                    description = ast.get_docstring(node) or ""
                if tool_name not in seen:
                    seen.add(tool_name)
                    yield ExtractedTool(
                        server=server,
                        tool_name=tool_name,
                        description=_single_line(description),
                        source_path=source_path,
                    )

        if isinstance(node, ast.Call):
            registration = _extract_registration_call(node)
            if registration is None:
                continue
            tool_name, description = registration
            if tool_name in seen:
                continue
            seen.add(tool_name)
            yield ExtractedTool(
                server=server,
                tool_name=tool_name,
                description=_single_line(description),
                source_path=source_path,
            )


def _is_tool_decorator(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tool"
    )


def _tool_name_from_call(
    node: ast.expr,
    *,
    default: str,
) -> str:
    if not isinstance(node, ast.Call):
        return default
    value = _string_keyword(node, "name")
    return value or default


def _description_from_call(node: ast.expr) -> str:
    if not isinstance(node, ast.Call):
        return ""
    return _string_keyword(node, "description") or ""


def _string_keyword(
    node: ast.Call,
    name: str,
) -> str | None:
    for keyword in node.keywords:
        if keyword.arg != name:
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(
            keyword.value.value,
            str,
        ):
            return keyword.value.value
    return None


def _extract_registration_call(
    node: ast.Call,
) -> tuple[str, str] | None:
    # Pattern: _register_tool("ToolName", callable)
    if isinstance(node.func, ast.Name) and node.func.id == "_register_tool":
        if not node.args:
            return None
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value, ""

    # Pattern: mcp.tool(name="tool_name")(callable)
    if isinstance(node.func, ast.Call) and _is_tool_decorator(node.func):
        name = _string_keyword(node.func, "name")
        if name:
            description = _description_from_call(node.func)
            return name, description

    return None


def _single_line(text: str) -> str:
    return " ".join(text.split())
