// Copyright © 2026 Apple Inc.

#pragma once

namespace tiki::core {

namespace cu {
class CommandEncoder;
}

class array;

void gather_mm(
    bool a_transposed,
    bool b_transposed,
    const array& a,
    const array& b,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    cu::CommandEncoder& encoder);

} // namespace tiki::core
