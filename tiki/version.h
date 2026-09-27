// Copyright © 2025 Apple Inc.

#pragma once

#include "tiki/api.h"

#define TIKI_VERSION_MAJOR 0
#define TIKI_VERSION_MINOR 32
#define TIKI_VERSION_PATCH 3
#define TIKI_VERSION_NUMERIC \
  (100000 * TIKI_VERSION_MAJOR + 1000 * TIKI_VERSION_MINOR + TIKI_VERSION_PATCH)

namespace tiki::core {

/* A string representation of the Tiki version in the format
 * "major.minor.patch".
 *
 * For dev builds, the version will include the suffix ".devYYYYMMDD+hash"
 */
TIKI_API const char* version();

} // namespace tiki::core
