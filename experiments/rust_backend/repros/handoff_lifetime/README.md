# Completion handoff ownership

Run `bash check.sh`. This host test compiles the production temporary-buffer
handoff from `CommandEncoder::commit_impl`. It injects `std::bad_alloc` at each
allocation until the handoff succeeds. Queued storage must remain owned after
every rejection, transfer once on retry, and release only after completion.

Moving the temporary vector into the callback destroyed its owners when callback
allocation or queue admission failed. The encoder now keeps its original owners
until admission succeeds. This test does not qualify CUDA execution, migration
leases, or the separate pre-launch retention contract.

Use `HANDOFF_TEST_FLAGS=-fsanitize=address` for host AddressSanitizer checks.
