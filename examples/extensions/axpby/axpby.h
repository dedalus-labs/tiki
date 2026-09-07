// Copyright © 2023-2025 Apple Inc.

#pragma once

#include "tiki/ops.h"
#include "tiki/primitives.h"

namespace tk = tiki::core;

namespace my_ext {

///////////////////////////////////////////////////////////////////////////////
// Operation
///////////////////////////////////////////////////////////////////////////////

/**
 *  Scale and sum two vectors element-wise
 *  z = alpha * x + beta * y
 *
 *  Follow numpy style broadcasting between x and y
 *  Inputs are upcasted to floats if needed
 **/
tk::array axpby(
    const tk::array& x, // Input array x
    const tk::array& y, // Input array y
    const float alpha, // Scaling factor for x
    const float beta, // Scaling factor for y
    tk::StreamOrDevice s = {} // Stream on which to schedule the operation
);

///////////////////////////////////////////////////////////////////////////////
// Primitive
///////////////////////////////////////////////////////////////////////////////

class Axpby : public tk::Primitive {
 public:
  explicit Axpby(tk::Stream stream, float alpha, float beta)
      : tk::Primitive(stream), alpha_(alpha), beta_(beta) {};

  /**
   * A primitive must know how to evaluate itself on the CPU/GPU
   * for the given inputs and populate the output array.
   *
   * To avoid unnecessary allocations, the evaluation function
   * is responsible for allocating space for the array.
   */
  void eval_cpu(
      const std::vector<tk::array>& inputs,
      std::vector<tk::array>& outputs) override;
  void eval_gpu(
      const std::vector<tk::array>& inputs,
      std::vector<tk::array>& outputs) override;

  /** The Jacobian-vector product. */
  std::vector<tk::array> jvp(
      const std::vector<tk::array>& primals,
      const std::vector<tk::array>& tangents,
      const std::vector<int>& argnums) override;

  /** The vector-Jacobian product. */
  std::vector<tk::array> vjp(
      const std::vector<tk::array>& primals,
      const std::vector<tk::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<tk::array>& outputs) override;

  /**
   * The primitive must know how to vectorize itself across
   * the given axes. The output is a pair containing the array
   * representing the vectorized computation and the axis which
   * corresponds to the output vectorized dimension.
   */
  std::pair<std::vector<tk::array>, std::vector<int>> vmap(
      const std::vector<tk::array>& inputs,
      const std::vector<int>& axes) override;

  /** The name of primitive. */
  const char* name() const override {
    return "Axpby";
  }

  /** Equivalence check **/
  bool is_equivalent(const tk::Primitive& other) const override;

 private:
  float alpha_;
  float beta_;
};

} // namespace my_ext
