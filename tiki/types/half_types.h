// Copyright © 2023 Apple Inc.

#pragma once

#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC

#include <arm_fp16.h>
namespace tiki::core {
using ::float16_t;
} // namespace tiki::core

#else

#define ADD_HALF_BINOPS
#include "tiki/types/fp16.h"
namespace tiki::core {
typedef struct _TIKI_Float16 float16_t;
} // namespace tiki::core

#endif // __ARM_FEATURE_FP16_SCALAR_ARITHMETIC

#ifdef __ARM_FEATURE_BF16

#include <arm_bf16.h>
namespace tiki::core {
using ::bfloat16_t;
} // namespace tiki::core

#else

#define ADD_HALF_BINOPS
#include "tiki/types/bf16.h"
namespace tiki::core {
typedef struct _TIKI_BFloat16 bfloat16_t;
} // namespace tiki::core

#endif // __ARM_FEATURE_BF16

namespace tiki::core {
// clang-format off

#ifdef ADD_HALF_BINOPS

#define fp16_bf16_binop_helper(__op__, __operator__)               \
  inline float __operator__(float16_t lhs, bfloat16_t rhs) {       \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs); \
  }                                                                \
  inline float __operator__(bfloat16_t lhs, float16_t rhs) {       \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs); \
  }

fp16_bf16_binop_helper(+, operator+)
fp16_bf16_binop_helper(-, operator-)
fp16_bf16_binop_helper(*, operator*)
fp16_bf16_binop_helper(/, operator/)

#endif

template <typename T>
constexpr bool is_signed_v = std::is_signed_v<T> ||
    std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

template <typename T>
constexpr bool is_floating_point_v = std::is_floating_point_v<T> ||
    std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

// clang-format on
} // namespace tiki::core
