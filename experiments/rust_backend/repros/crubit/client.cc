#include <tiki_backend_contract.h>
#include <cassert>

int main() {
  auto buffer = tiki_backend_contract::Buffer::new_(128);
  assert(buffer.len() == 128);
  buffer.fill(2.0f);
  assert(buffer.checksum() == 256.0f);
}
