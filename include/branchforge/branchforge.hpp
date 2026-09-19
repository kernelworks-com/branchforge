#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace branchforge {

using TokenId = std::uint32_t;

struct RuntimeContext;

enum class ErrorCode {
  InvalidArgument,
  InvalidHandle,
  StaleHandle,
  WrongRuntime,
  Busy,
  InvalidToken,
  PositionLimit,
  ResourceExhausted,
  InternalFailure,
};

struct Error {
  ErrorCode code;
  std::string message;
};

const char* to_string(ErrorCode code) noexcept;

// A small C++20 result type used by all operations that can reject a request.
template <typename T>
class Result {
 public:
  Result(const T& value) : value_(value) {}
  Result(T&& value) : value_(std::move(value)) {}
  Result(const Error& error) : value_(error) {}
  Result(Error&& error) : value_(std::move(error)) {}

  [[nodiscard]] bool has_value() const noexcept {
    return std::holds_alternative<T>(value_);
  }

  explicit operator bool() const noexcept { return has_value(); }

  T& value() &;
  const T& value() const&;
  T&& value() &&;
  const Error& error() const;

 private:
  std::variant<T, Error> value_;
};

template <typename T>
T& Result<T>::value() & {
  return std::get<T>(value_);
}

template <typename T>
const T& Result<T>::value() const& {
  return std::get<T>(value_);
}

template <typename T>
T&& Result<T>::value() && {
  return std::get<T>(std::move(value_));
}

template <typename T>
const Error& Result<T>::error() const {
  return std::get<Error>(value_);
}

enum class BranchDecision { Keep, Discard };

struct Config {
  // The reference model consumes token IDs in [0, vocabulary_size).
  std::uint32_t vocabulary_size = 32;
  std::size_t recurrent_width = 4;
  std::uint64_t max_position = 1'000'000;
  std::uint64_t model_revision = 1;
  std::string adapter_id = "base";
  std::uint64_t seed = 0x6a09e667f3bcc909ULL;
  // Optional deterministic controls for semantic tests. Production callers
  // should leave this empty; the controls do not change model semantics.
  std::shared_ptr<class TestHooks> test_hooks;
};

// Deterministic controls used by the reference-runtime tests. They make
// failure atomicity and Busy behavior observable without timing assumptions.
class TestHooks {
 public:
  TestHooks();

  void block_next_advance();
  void wait_until_advance_blocked() const;
  void release_advance();
  void fail_next_advance_after_tokens(std::size_t token_count);
  void fail_next_keep();
  void force_next_branch_generation_for_testing(std::uint64_t generation);

 private:
  struct State;
  std::shared_ptr<State> state_;

  friend class Runtime;
  void wait_at_advance_gate();
  [[nodiscard]] bool fail_after_token_count(std::size_t token_count);
  [[nodiscard]] bool consume_keep_failure();
  [[nodiscard]] std::optional<std::uint64_t> consume_generation_override();
};

struct StateView {
  std::uint64_t position = 0;
  std::vector<TokenId> attention_tokens;
  std::vector<std::int64_t> recurrent_state;
  std::uint64_t rng_state = 0;
};

struct AdvanceResult {
  std::uint64_t position = 0;
  std::vector<float> logits;
};

class Runtime;

class Snapshot {
 public:
  // Opaque storage type used by the implementation. It is public only so the
  // library can keep the handle layout stable without exposing model details.
  struct State;

  Snapshot() = default;

  [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
  [[nodiscard]] std::uint64_t id() const noexcept { return id_; }
  void reset() noexcept;

 private:
  friend class Runtime;

  Snapshot(std::shared_ptr<const State> state,
           std::shared_ptr<const RuntimeContext> context,
           std::uint64_t id)
      : context_(std::move(context)), state_(std::move(state)), id_(id) {}

  std::shared_ptr<const RuntimeContext> context_;
  std::shared_ptr<const State> state_;
  std::uint64_t id_ = 0;
};

class Branch {
 public:
  Branch() = default;
  ~Branch();
  Branch(const Branch&) = delete;
  Branch& operator=(const Branch&) = delete;
  Branch(Branch&& other) noexcept;
  Branch& operator=(Branch&& other) noexcept;

  [[nodiscard]] bool valid() const noexcept {
    return control_ != nullptr && active_;
  }
  [[nodiscard]] std::uint64_t id() const noexcept { return id_; }
  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

 private:
  struct Control;
  friend class Runtime;

  Branch(std::shared_ptr<Control> control,
         std::uint64_t id,
         std::uint64_t generation)
      : control_(std::move(control)),
        id_(id),
        generation_(generation),
        active_(true) {}

  std::shared_ptr<Control> control_;
  std::uint64_t id_ = 0;
  std::uint64_t generation_ = 0;
  bool active_ = false;

  void abandon_noexcept() noexcept;
};

class Runtime {
 public:
  Runtime();
  explicit Runtime(Config config);

  // Performs configuration validation without throwing. The constructors are
  // convenient for known-valid local configurations and throw
  // std::invalid_argument when a configuration is invalid.
  static Result<Runtime> create(Config config);

  Runtime(Runtime&&) noexcept = default;
  Runtime& operator=(Runtime&&) noexcept = default;
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  ~Runtime();

  [[nodiscard]] const Config& config() const noexcept;

  // A root snapshot starts at position zero with empty attention and recurrent
  // state. Snapshots are immutable and may be copied to retain ownership.
  [[nodiscard]] Result<Snapshot> root_snapshot() const;

  // fork eagerly copies every component of the supplied immutable snapshot.
  [[nodiscard]] Result<Branch> fork(const Snapshot& snapshot) const;

  // Consumes exactly the supplied token IDs. An empty sequence is legal and
  // leaves state unchanged while returning logits for the current boundary.
  [[nodiscard]] Result<AdvanceResult> advance(
      Branch& branch,
      std::span<const TokenId> token_ids) const;

  // KEEP creates an immutable snapshot at the completed branch boundary and
  // consumes the branch. DISCARD invalidates the branch and returns no
  // snapshot. Both decisions are atomic with respect to branch mutation.
  [[nodiscard]] Result<std::optional<Snapshot>> discard_or_keep(
      Branch& branch,
      BranchDecision decision) const;

  [[nodiscard]] Result<StateView> inspect(const Snapshot& snapshot) const;
  [[nodiscard]] Result<StateView> inspect(const Branch& branch) const;

 private:
  explicit Runtime(std::shared_ptr<RuntimeContext> context);

  std::shared_ptr<RuntimeContext> context_;
};

}  // namespace branchforge
