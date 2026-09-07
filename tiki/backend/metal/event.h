// Copyright © 2026 Apple Inc.

#include "tiki/backend/metal/device.h"
#include "tiki/event.h"

namespace tiki::core::metal {

class EventImpl {
 public:
  EventImpl(Device& d);
  ~EventImpl();

  void wait(uint64_t value);
  void signal(uint64_t value);

  auto& error() {
    return error_;
  }

  auto* mtl_event() const {
    return mtl_event_.get();
  }

 private:
  // All streams outlive events so pointers would be always valid.
  std::atomic<Error*> error_;

  NS::SharedPtr<MTL::SharedEvent> mtl_event_;
};

} // namespace tiki::core::metal
