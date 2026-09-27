#pragma once

#include "tiki/array.h"
#include "tiki/primitives.h"

namespace tiki::core {

void scan_gpu_inplace(
    const array& in,
    array& out,
    Scan::ReduceType reduce_type,
    int axis,
    bool reverse,
    bool inclusive,
    const Stream& s);

} // namespace tiki::core
