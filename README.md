# Branchforge

Branchforge is a small C++20 reference runtime for exploring independent
inference continuations. The first implementation provides a synchronous CPU
toy backend whose purpose is to make state lifetime and branching semantics
testable. It is not a real model backend and does not claim GPU support.

## Build and run

Branchforge has no third-party runtime dependency. With CMake and a C++20
compiler available:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
./build/branchforge_demo
```

The demo creates two continuations from one root snapshot, advances them with
different supplied token sequences, retains one branch, forks from the
retained snapshot, and discards completed branches.

## Synchronous API

The public header is [`include/branchforge/branchforge.hpp`](include/branchforge/branchforge.hpp).
The core operations are:

```cpp
branchforge::Runtime runtime;
auto root = runtime.root_snapshot();
auto branch = runtime.fork(root.value());

std::array<branchforge::TokenId, 2> tokens{4, 8};
auto result = runtime.advance(branch.value(), tokens);
auto kept = runtime.discard_or_keep(
    branch.value(), branchforge::BranchDecision::Keep);
```

`advance` consumes exactly the supplied token IDs. The sequence is not a token
budget, and generation or sampling is not part of this milestone. An empty
sequence is valid and returns logits for the current boundary without changing
state.

Snapshots are immutable and copyable. `fork` eagerly copies the toy model's
append-only attention history, recurrent state, position, and sampling state.
The copy is deliberate correctness baseline behavior; this release does not
promise constant-time or zero-copy branching.

`KEEP` publishes a completed immutable snapshot and consumes the mutable branch.
`DISCARD` invalidates the branch and publishes no snapshot. A branch cannot be
merged back into its parent. Handles carry a checked slot generation, so a
stale branch remains rejected even when its slot is later reused. Operations
that fail validation leave the branch at its prior completed boundary.

The reference toy model accepts token IDs from `0` through
`vocabulary_size - 1`. Its `StateView` is available for deterministic tests and
examples; it is not a representation of a production model's internal state.

## Scope and limitations

This milestone is intentionally synchronous. A branch mutation uses a
try-locked branch guard and reports `Busy` if another mutation is in progress,
but there are no asynchronous operation handles, cancellation, device leases,
memory backpressure, or deferred device reclamation yet. The eager state copy
also means the implementation does not measure or provide shared attention
pages, recurrent copy-on-write, checkpoint/replay, or a real model adapter.

Model and adapter compatibility is represented by each runtime's context
ownership; snapshots cannot be used with another runtime. Sampling stream
splitting and a separate sampler API remain future work. External application
side effects are outside the runtime's state boundary. The public `TestHooks`
type exists for deterministic semantic tests of Busy and failure atomicity; it
is not a scheduler or fault model for production backends.

## Related work

Prefix caching and recurrent-state management already exist in serving systems.
[SGLang's unified cache](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/unified_cache/components/README.md)
includes multiple attention/state types and copy-on-write behavior.

Branchforge's research question is whether explicit branch operations can provide
useful semantics and measurable efficiency for a supported workload. The API
names alone are not a claim of novelty.

## Kernelworks

Branchforge is an independent Kernelworks project. It does not require
Stateguard or Yieldpoint.

## Continuous checks and source delivery

See [.github/README.md](.github/README.md) for the compiler matrix, sanitizer
checks, and tested source archives delivered after successful checks on `main`.

## License

Apache License 2.0. See [LICENSE](LICENSE). Model weights and third-party
backends retain their own license terms.
