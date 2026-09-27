#!/bin/sh
set -eu

test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM

# Invariant: static consumers need only the installed prefix.
# Witness: move a lib64 install, delete its build tree, then link and run.
cmake -S "$test_dir/producer" -B "$work_dir/build" \
    -DCMAKE_INSTALL_PREFIX="$work_dir/install" -DCMAKE_INSTALL_LIBDIR=lib64
cmake --build "$work_dir/build"
cmake --install "$work_dir/build"
mv "$work_dir/install" "$work_dir/relocated"
rm -rf "$work_dir/build"
cmake -S "$test_dir/consumer" -B "$work_dir/consumer" \
    -DTIKI_PREFIX="$work_dir/relocated"
cmake --build "$work_dir/consumer"
"$work_dir/consumer/consumer"
