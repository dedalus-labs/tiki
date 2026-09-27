// Copyright © 2026 Dedalus Labs, Inc.
#![allow(dead_code)]

#[path = "../../src/cudart.rs"]
mod cudart;

extern "C" {
    fn received_device() -> i32;
}

// Invariant: advice preserves the device ordinal across supported toolkit ABIs.
// Witness: devices 0, 2, and 3 plus an invalid device error on CUDA 12 and 13.
#[test]
fn invariant_advice_preserves_device_and_errors() {
    for device in [0, 2, 3] {
        cudart::advise_accessed_by(std::ptr::null(), 0, device).unwrap();
        // SAFETY: the test callee has no pointer arguments.
        assert_eq!(unsafe { received_device() }, device);
    }
    let error = cudart::advise_accessed_by(std::ptr::null(), 0, -1).unwrap_err();
    assert_eq!(error.call, "cudaMemAdvise");
    assert_eq!(error.code, 101);
    assert_eq!(error.name, "cudaErrorInvalidDevice");
}
