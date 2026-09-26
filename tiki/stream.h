// Copyright © 2023 Apple Inc.

#pragma once

#include <tuple>
#include <vector>

#include "tiki/api.h"
#include "tiki/device.h"

namespace tiki::core {

struct TIKI_API Stream {
  int index;
  Device device;
  explicit Stream(int index, Device device) : index(index), device(device) {}

  // TODO: Use default three-way comparison when it gets supported in XCode.
  bool operator==(const Stream&) const = default;
  bool operator<(const Stream& rhs) const {
    return std::tie(device, index) < std::tie(rhs.device, rhs.index);
  }
};

struct TIKI_API ThreadLocalStream : public Stream {
  using Stream::Stream;
};

/** Get the default stream of current thread for the given device. */
TIKI_API Stream default_stream(Device d);

/** Make the stream the default for its device on current thread. */
TIKI_API void set_default_stream(Stream s);

/** Make a new stream on the given device. */
TIKI_API Stream new_stream(Device d);

/** Make a new stream that can be used in any thread. */
TIKI_API Stream new_thread_unsafe_stream(Device d);

/** Make a new stream that will be unique per thread. */
TIKI_API ThreadLocalStream new_thread_local_stream(Device d);

/** Get the stream for current thread from ThreadLocalStream. */
TIKI_API Stream stream_from_thread_local_stream(ThreadLocalStream tls);

/** Get all available streams. */
TIKI_API std::vector<Stream> get_streams();

/* Synchronize with the default stream. */
TIKI_API void synchronize();

/* Synchronize with the provided stream. */
TIKI_API void synchronize(Stream);

/* Synchronize with the stream corresponding to the current thread. */
TIKI_API void synchronize(ThreadLocalStream);

/* Destroy all streams created in current thread. */
TIKI_API void clear_streams();

} // namespace tiki::core
