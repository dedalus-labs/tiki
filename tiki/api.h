// Copyright © 2024 Apple Inc.

#pragma once

// TIKI_API macro for controlling symbol visibility, must add for public APIs.
//
// Usage:
//   TIKI_API void some_function(...);
//   class TIKI_API SomeClass { ... };

#if defined(TIKI_STATIC)

// Static library build - no import/export decorations needed
#define TIKI_API

#else

// Shared library build.
#if defined(_WIN32)
#if defined(TIKI_EXPORT)
#define TIKI_API __declspec(dllexport)
#else
#define TIKI_API __declspec(dllimport)
#endif // defined(TIKI_EXPORT)
#else
#define TIKI_API __attribute__((visibility("default")))
#endif // defined(_WIN32)

#endif // defined(TIKI_STATIC)
