// Copyright © 2025 Apple Inc.

#pragma once

#include "tiki/array.h"
#include "tiki/stream.h"

namespace tiki::core::cpu {

void new_stream(Stream s);
void new_thread_unsafe_stream(Stream s);
void eval(array& arr);
void clear_streams();

} // namespace tiki::core::cpu
