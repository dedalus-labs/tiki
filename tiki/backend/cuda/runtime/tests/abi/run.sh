#!/bin/sh
set -eu

test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM

for version in 12090 13000; do
    c++ -std=c++20 -DCUDART_VERSION="$version" -I "$test_dir" \
        -c "$test_dir/callee.cpp" -o "$work_dir/callee.o"
    c++ -std=c++20 -DCUDART_VERSION="$version" -I "$test_dir" \
        -c "$test_dir/../../src/cudart.cpp" -o "$work_dir/shim.o"
    rustc --edition=2021 --test "$test_dir/probe.rs" \
        -C link-arg="$work_dir/callee.o" -C link-arg="$work_dir/shim.o" \
        -o "$work_dir/probe"
    "$work_dir/probe"
done
