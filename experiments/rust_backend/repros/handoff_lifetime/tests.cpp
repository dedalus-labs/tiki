// Copyright (c) 2026 Dedalus Labs, Inc. All rights reserved.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <new>
#include <vector>

namespace {
int allocation_failure = -1;
int allocations = 0;
int releases = 0;
bool completed = false;
int premature_releases = 0;

struct Storage {
  ~Storage() {
    ++releases;
    premature_releases += !completed;
  }
};

struct Encoder {
  std::vector<std::shared_ptr<Storage>> temporaries_;
  std::vector<std::function<void()>> pending;

  void add_completed_handler(std::function<void()> task) {
    pending.push_back(std::move(task));
  }

  void handoff() {
#include "handoff.inc"
  }
};
} // namespace

void* operator new(std::size_t size) {
  if (allocation_failure >= 0 && allocations++ == allocation_failure) {
    throw std::bad_alloc();
  }
  if (void* ptr = std::malloc(size ? size : 1)) {
    return ptr;
  }
  throw std::bad_alloc();
}

void operator delete(void* ptr) noexcept {
  std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
  std::free(ptr);
}

int main() {
  int failed_allocations = 0;
  for (int failure = 0; failure < 16; ++failure) {
    Encoder encoder;
    encoder.temporaries_.push_back(std::make_shared<Storage>());
    std::weak_ptr<Storage> submitted = encoder.temporaries_.front();
    releases = premature_releases = allocations = 0;
    completed = false;
    allocation_failure = failure;
    bool rejected = false;
    try {
      encoder.handoff();
    } catch (const std::bad_alloc&) {
      rejected = true;
    }
    allocation_failure = -1;
    if (submitted.expired() || premature_releases != 0) {
      std::fprintf(
          stderr, "FAIL  queued storage lost at allocation %d\n", failure);
      return 1;
    }
    if (rejected) {
      assert(encoder.temporaries_.size() == 1);
      assert(encoder.pending.empty());
      ++failed_allocations;
      encoder.handoff();
    }
    assert(encoder.temporaries_.empty());
    assert(encoder.pending.size() == 1);
    assert(!submitted.expired());
    completed = true;
    encoder.pending.clear();
    assert(submitted.expired());
    assert(releases == 1);
    assert(premature_releases == 0);
    if (!rejected) {
      assert(failed_allocations > 0);
      std::printf(
          "  ok  handoff ownership (%d allocation failures)\n",
          failed_allocations);
      return 0;
    }
  }
  std::fputs("FAIL  handoff never admitted\n", stderr);
  return 1;
}
