// Copyright © 2025 Apple Inc.

#pragma once

#define TIKI_UNROLL _Pragma("unroll")

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define TIKI_CUDA_SM_80_ENABLED
#endif
