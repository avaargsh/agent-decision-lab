from __future__ import annotations

import argparse

from decision_lab.inventory import inventory_digest, write_inventory
from decision_lab.inventory_extract import build_inventory_from_python_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a frozen MCP tool inventory from a Python source tree."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--inventory-id", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = build_inventory_from_python_tree(
        args.root,
        inventory_id=args.inventory_id,
        repository=args.repository,
        revision=args.revision,
    )
    write_inventory(inventory, args.output)
    print(
        f"inventory_id={inventory.inventory_id} "
        f"tools={len(inventory.tools)} "
        f"sha256={inventory_digest(inventory)}"
    )


if __name__ == "__main__":
    main()
