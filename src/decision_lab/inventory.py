from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


INVENTORY_SCHEMA_VERSION = "mcp-tool-inventory/v1"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    server: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if not self.server:
            raise ValueError("tool server must not be empty")


@dataclass(frozen=True)
class ToolInventory:
    inventory_id: str
    tools: tuple[ToolSpec, ...]
    source: dict[str, Any]
    schema_version: str = INVENTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.inventory_id:
            raise ValueError("inventory_id must not be empty")
        if self.schema_version != INVENTORY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported inventory schema: {self.schema_version}"
            )
        if not self.tools:
            raise ValueError("tool inventory must not be empty")

        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique inside an inventory")


def inventory_payload(inventory: ToolInventory) -> dict[str, Any]:
    return {
        "schema_version": inventory.schema_version,
        "inventory_id": inventory.inventory_id,
        "source": inventory.source,
        "tools": [asdict(tool) for tool in inventory.tools],
    }


def inventory_digest(inventory: ToolInventory) -> str:
    encoded = json.dumps(
        inventory_payload(inventory),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_inventory(path: str | Path) -> ToolInventory:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    tools = tuple(
        ToolSpec(
            name=item["name"],
            description=item.get("description", ""),
            server=item["server"],
            metadata=dict(item.get("metadata", {})),
        )
        for item in obj["tools"]
    )
    inventory = ToolInventory(
        inventory_id=obj["inventory_id"],
        tools=tools,
        source=dict(obj.get("source", {})),
        schema_version=obj.get(
            "schema_version",
            INVENTORY_SCHEMA_VERSION,
        ),
    )

    declared_digest = obj.get("sha256")
    if declared_digest is not None:
        actual_digest = inventory_digest(inventory)
        if declared_digest != actual_digest:
            raise ValueError(
                "inventory sha256 mismatch: "
                f"declared {declared_digest}, actual {actual_digest}"
            )

    return inventory


def write_inventory(
    inventory: ToolInventory,
    path: str | Path,
) -> None:
    payload = inventory_payload(inventory)
    payload["sha256"] = inventory_digest(inventory)
    Path(path).write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def make_inventory(
    *,
    inventory_id: str,
    tools: Iterable[ToolSpec],
    source: dict[str, Any],
) -> ToolInventory:
    return ToolInventory(
        inventory_id=inventory_id,
        tools=tuple(tools),
        source=dict(source),
    )
