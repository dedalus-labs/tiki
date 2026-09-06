#include "tiki-cxx-spike/src/main.rs.h"
#include "client.h"
#include <cassert>
#include <utility>

std::size_t tiki::run_client() {
  auto buffer = new_buffer(128);
  assert(buffer->len() == 128);
  buffer->fill(2.0f);
  assert(buffer->checksum() == 256.0f);
  return consume_buffer(std::move(buffer));
}
