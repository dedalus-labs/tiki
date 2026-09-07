// Copyright © 2023 Apple Inc.

#include <cassert>

#include "tiki/backend/cpu/binary_ops.h"
#include "tiki/backend/cpu/ternary.h"
#include "tiki/dtype_utils.h"
#include "tiki/primitives.h"

namespace tiki::core {

namespace {

template <typename Op>
void select_op(
    const array& a,
    const array& b,
    const array& c,
    array& out,
    Op op,
    Stream stream) {
  TernaryOpType topt = get_ternary_op_type(a, b, c);
  set_ternary_op_output_data(a, b, c, out, topt);

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_input_array(c);
  encoder.set_output_array(out);
  encoder.dispatch([a = array::unsafe_weak_copy(a),
                    b = array::unsafe_weak_copy(b),
                    c = array::unsafe_weak_copy(c),
                    out = array::unsafe_weak_copy(out),
                    op,
                    topt]() mutable {
    dispatch_all_types(out.dtype(), [&](auto type_tag) {
      using T = TIKI_GET_TYPE(type_tag);
      ternary_op<bool, T, T, T>(a, b, c, out, op, topt);
    });
  });
}

} // namespace

void Select::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);
  const auto& condition = inputs[0];
  const auto& a = inputs[1];
  const auto& b = inputs[2];
  select_op(condition, a, b, out, detail::Select(), stream());
}

} // namespace tiki::core
