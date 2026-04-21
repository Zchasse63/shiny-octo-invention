# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2026 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the MCP server.

The module is meant to be executed as a script using the command:

```bash
python -m vectorbtpro.mcp_server
```
"""

import argparse

from vectorbtpro import _typing as tp
from vectorbtpro.mcp import tool_registry

__all__ = []


def main(argv: tp.Optional[tp.Sequence[str]] = None) -> None:
    """Run the MCP server.
    
    Args:
        argv (Optional[Sequence[str]]): Command-line arguments.

    Returns:
        None
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("mcp")
    from mcp.server.fastmcp import FastMCP

    parser = argparse.ArgumentParser(description="Run the MCP server for VectorBT PRO.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    args = parser.parse_args(argv)

    mcp = FastMCP("VectorBT PRO")
    for name, tool in tool_registry.items():
        mcp.tool(name=name)(tool)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
