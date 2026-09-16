# Branchforge

**Explicit branching for inference state, from Kernelworks.**

Branchforge is a planned native runtime component for exploring independent model
continuations while making state sharing, mutation, and disposal explicit.

## Project status

**Pre-implementation.** This repository contains project documentation. There is
no installable engine, stable API, supported model list, or performance benchmark
yet. The following interface is conceptual pseudocode.

## The interface

```text
branch = fork(state)
advance(branch, tokens)
discard_or_keep(branch)
```

The intended meaning is simple:

- **fork:** create an independent continuation from a consistent state.
- **advance:** consume a supplied token sequence and update that branch.
- **discard or keep:** release a continuation or retain a stable state for further work.

Token generation would be a separate operation with an explicit sampling policy.
Keeping a branch does not merge it with another branch, and discarding one does
not mean its physical memory can always be reused immediately.

## The problem

Applications that explore several continuations need more than a copy of the text.
They need independent model state, correct positions and sampling state, and
ownership rules for any outstanding computation.

Attention state can often share immutable prefixes. Recurrent state may require
copying or reconstruction when a branch advances. Branchforge aims to expose
consistent semantics while allowing different model components to use different
storage strategies.

## Intended capabilities

- Immutable retained states and explicit mutable branch handles.
- Shared attention prefixes with isolation on mutation.
- Model-specific recurrent-state copying or checkpoint/replay.
- Explicit sampling-stream behavior.
- Predictable errors for stale handles, incompatible state, and unsupported operations.
- Deferred physical reclamation when work is still in flight.

The first target is a portable reference backend followed by one documented model
family and native backend. The project does not promise universal model support
or constant-time, zero-copy branching.

## Intended use

Examples include best-of-several continuation experiments, search over candidate
token sequences, and interactive systems that abandon one continuation and retain
another. Branchforge manages model state; application decisions and external tool
side effects remain the application's responsibility.

Build instructions and runnable examples will be added when implementation exists.
The intended runtime is self-hosted and does not require a Kernelworks service.

## Related work

Prefix caching and recurrent-state management already exist in serving systems.
[SGLang's unified cache](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/unified_cache/components/README.md)
includes multiple attention/state types and copy-on-write behavior.

Branchforge's research question is whether explicit branch operations can provide
useful semantics and measurable efficiency for a supported workload. The API names
alone are not a claim of novelty.

## Kernelworks

Branchforge is an independent Kernelworks project. It does not require Stateguard
or Yieldpoint.

## License

Apache License 2.0. See [LICENSE](LICENSE). Model weights and third-party backends
retain their own license terms.
