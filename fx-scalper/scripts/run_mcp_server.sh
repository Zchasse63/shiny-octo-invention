#!/usr/bin/env bash
# Launch vectorbtpro's MCP server with fx-scalper/.env sourced for API keys.
# Invoked by /Users/zach/Desktop/Forex/.mcp.json as the "vectorbtpro" server.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FX_DIR="$REPO_ROOT/fx-scalper"
VENV_PY="$FX_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "vbtpro MCP launcher: venv missing at $FX_DIR/.venv/" >&2
  exit 1
fi

# Source .env so vbt.chat finds ANTHROPIC_API_KEY / OPENAI_API_KEY / GITHUB_TOKEN.
if [[ -f "$FX_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$FX_DIR/.env"
  set +a
fi

exec "$VENV_PY" -m vectorbtpro.mcp_server --transport stdio
