// Copyright © 2026 Apple Inc.

#pragma once

#include <nanobind/nanobind.h>

#include "tiki/array.h"

namespace tk = tiki::core;
namespace nb = nanobind;

// Clear the `tk.random.state` python object in current thread.
void reset_random_state();

// The process-global `tk.random.state` sentinel.
nb::object random_state_sentinel();

// Read/write the calling thread's current PRNG key.
tk::array random_state_key();
void set_random_state_key(const tk::array& key);
