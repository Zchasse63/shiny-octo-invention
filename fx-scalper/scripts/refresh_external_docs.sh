#!/usr/bin/env bash
# Refresh fx-scalper/docs/external/ — shallow-clones upstream library repos.
# Safe to re-run; existing clones are wiped and re-pulled fresh.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_DIR="$PROJECT_ROOT/docs/external"

# (directory_name, upstream_url, prune_paths_space_separated_or_empty)
REPOS=(
  "oandapyV20|https://github.com/hootnot/oanda-api-v20.git|"
  "pandas-ta-classic|https://github.com/xgboosted/pandas-ta-classic.git|"
  "ta-lib-python|https://github.com/TA-Lib/ta-lib-python.git|"
  "loguru|https://github.com/Delgan/loguru.git|"
  "nautilus_trader|https://github.com/nautechsystems/nautilus_trader.git|tests crates assets"
  "duka|https://github.com/giuse88/duka.git|"
)

mkdir -p "$EXT_DIR"
cd "$EXT_DIR"

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url prune <<<"$entry"
  echo "==> $name"
  rm -rf "$name"
  git clone --depth 1 "$url" "$name"
  ( cd "$name" && rm -rf .git .github )
  if [[ -n "$prune" ]]; then
    for dir in $prune; do
      rm -rf "$name/$dir"
    done
  fi
done

echo
echo "Sizes:"
du -sh "$EXT_DIR"/*/
