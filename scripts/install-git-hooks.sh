#!/usr/bin/env bash
# Point this clone at the versioned hooks under .githooks/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
chmod +x .githooks/pre-commit
git config core.hooksPath .githooks
echo "Installed: core.hooksPath=.githooks (runs Pulse design contract on relevant commits)"
