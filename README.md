# Branchforge

**Fork model state. Explore independent continuations. Keep the result you need.**

Branchforge is a C++ library for experimenting with inference branches: start two
continuations from the same snapshot, feed them different tokens, and retain or
discard each branch without changing the original state.

It is for inference engineers exploring search, branching, and state ownership.
**The current implementation is an experimental CPU toy backend.** You can test
branch behavior today; running a real language model requires a future backend.

## Try it

You need a C++20 compiler and CMake 3.20+. No GPU or model download is needed.
See [build help](docs/BUILDING.md) if `cmake` is missing or compilation fails.

```sh
git clone https://github.com/kernelworks-com/branchforge.git
cd branchforge
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
./build/branchforge_demo
```

Already in the repository? Start at the `cmake` command. The demo prints:

```text
left position=3 tokens=[4,8,15] recurrent[0]=745016773
right position=2 tokens=[16,23] recurrent[0]=362206358
root position=0 tokens=[] recurrent[0]=0
kept position=3 tokens=[4,8,15] recurrent[0]=745016773
continuation position=4 tokens=[4,8,15,24] recurrent[0]=440169141
```

The two branches have different histories while the root stays unchanged. Keeping
`left` creates a reusable snapshot; advancing a new branch from it leaves that
snapshot unchanged too. The recurrent values are deterministic toy-model state.

## Work with branches

| Operation | What you get |
|---|---|
| `fork(snapshot)` | An independent mutable branch copied from a snapshot. |
| `advance(branch, token_ids)` | Updated state and toy logits after consuming those exact token IDs. |
| `discard_or_keep(branch, Keep)` | An immutable snapshot you can fork again. |
| `discard_or_keep(branch, Discard)` | A retired branch with no retained snapshot. |

`advance` takes token IDs, not a number of tokens to generate. Check each returned
`Result` before using its value. The [complete C++ example](examples/branch_explorer.cpp)
shows error handling, branching, inspection, and cleanup. See the
[API reference](docs/REFERENCE.md) for ownership and failure behavior.

## What works today

- Independent copies of token history, recurrent state, position, and RNG state.
- Immutable snapshots, checked branch handles, and stale-handle rejection.
- Atomic advance and KEEP failures: rejected work preserves the prior branch state.
- A `Busy` result when another mutation holds the branch, plus automatic cleanup.

Branches use eager copies. Shared memory pages, copy-on-write, a sampler, async
operations, and GPU/real-model backends are future work. Branches do not merge, and
application side effects are outside the snapshot. No performance advantage is
claimed for the toy backend.

## Tests and feedback

```sh
ctest --test-dir build --output-on-failure --no-tests=error
```

Tests cover branch independence, state/logit equivalence, stale handles, ownership
transfer, allocation failures, and concurrent mutation. Build automation is
described in [CI details](.github/README.md).

[Open an issue](https://github.com/kernelworks-com/branchforge/issues) with a small
example and the expected versus actual branch behavior. Include tool versions for
build problems. Synthetic examples are preferred over customer data.

Branchforge is an independent [Kernelworks](https://github.com/kernelworks-com)
project; no sibling project is required. Licensed under [Apache-2.0](LICENSE).
