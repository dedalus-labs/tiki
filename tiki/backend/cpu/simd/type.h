#pragma once

#include "tiki/backend/cpu/simd/base_simd.h"

#ifdef TIKI_USE_ACCELERATE
#if defined(__x86_64__)
// the accelerate_simd implementation require neon -- use base implementation
#else
#include "tiki/backend/cpu/simd/accelerate_simd.h"
#endif
#endif
