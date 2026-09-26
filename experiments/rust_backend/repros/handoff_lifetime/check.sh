#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
sed -n '/^void CommandEncoder::commit_impl()/,/^  if (use_cuda_graphs()/p' \
  ../../../../mlx/backend/cuda/device.cpp | sed '1d;$d' > "$out/handoff.inc"
test -s "$out/handoff.inc"
read -r -a test_flags <<< "${HANDOFF_TEST_FLAGS:-}"
"${CXX:-c++}" "${test_flags[@]}" -std=c++20 -I"$out" tests.cpp -o "$out/tests"
"$out/tests"
