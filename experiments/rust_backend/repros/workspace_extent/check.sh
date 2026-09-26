#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source_file=${1:-../../../../mlx/backend/cuda/utils.cpp}
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
sed -n '/^void\* allocate_workspace(/,/^} \/\/ namespace mlx::core/p' \
  "$source_file" | sed '$d' > "$out/workspace.inc"
read -r -a test_flags <<< "${WORKSPACE_TEST_FLAGS:-}"
"${CXX:-c++}" -std=c++20 -I"$out" "${test_flags[@]}" tests.cpp -o "$out/tests"
"$out/tests"
