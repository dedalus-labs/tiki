// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <utility>

#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace {

template <const uint8_t scalar_size>
void swap_endianness(uint8_t* data_bytes, size_t N) {
  struct Elem {
    uint8_t bytes[scalar_size];
  };

  Elem* data = reinterpret_cast<Elem*>(data_bytes);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < (scalar_size / 2); j++) {
      std::swap(data[i].bytes[j], data[i].bytes[scalar_size - j - 1]);
    }
  }
}

} // namespace

namespace mlx::core {

void Load::eval_cpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));
  auto read_task = [out_ptr = out.data<char>(),
                    nbytes = out.nbytes(),
                    scalar_size = out.dtype() == complex64 ? 4 : out.itemsize(),
                    offset = offset_,
                    reader = reader_,
                    swap_endianness_ = swap_endianness_]() mutable {
    reader->read(out_ptr, nbytes, offset);
    if (swap_endianness_) {
      auto size = nbytes / scalar_size;
      switch (scalar_size) {
        case 2:
          swap_endianness<2>(reinterpret_cast<uint8_t*>(out_ptr), size);
          break;
        case 4:
          swap_endianness<4>(reinterpret_cast<uint8_t*>(out_ptr), size);
          break;
        case 8:
          swap_endianness<8>(reinterpret_cast<uint8_t*>(out_ptr), size);
          break;
      }
    }
  };
  auto fut = io::thread_pool().enqueue(std::move(read_task)).share();
  scheduler::enqueue(stream(), [fut = std::move(fut)]() { fut.get(); });
}

} // namespace mlx::core
