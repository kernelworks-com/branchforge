#include "branchforge/branchforge.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string_view>

namespace branchforge {

namespace {

constexpr std::uint64_t kMixConstant = 0x9e3779b97f4a7c15ULL;
constexpr std::uint64_t kRecurrentModulus = 1'000'000'007ULL;

std::uint64_t splitmix64(std::uint64_t value) noexcept {
  value += kMixConstant;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

Error make_error(ErrorCode code, std::string message) {
  return Error{code, std::move(message)};
}

}  // namespace

struct Snapshot::State {
  std::uint64_t position = 0;
  std::vector<TokenId> attention_tokens;
  std::vector<std::int64_t> recurrent_state;
  std::uint64_t rng_state = 0;
};

struct TestHooks::State {
  mutable std::mutex mutex;
  std::condition_variable condition;
  bool block_next_advance = false;
  bool advance_blocked = false;
  bool release_advance = false;
  std::optional<std::size_t> fail_after_tokens;
  bool fail_keep = false;
  std::optional<std::uint64_t> generation_override;
};

TestHooks::TestHooks() : state_(std::make_shared<State>()) {}

void TestHooks::block_next_advance() {
  std::lock_guard lock(state_->mutex);
  state_->block_next_advance = true;
  state_->advance_blocked = false;
  state_->release_advance = false;
}

void TestHooks::wait_until_advance_blocked() const {
  std::unique_lock lock(state_->mutex);
  state_->condition.wait(lock, [this] { return state_->advance_blocked; });
}

void TestHooks::release_advance() {
  {
    std::lock_guard lock(state_->mutex);
    state_->release_advance = true;
  }
  state_->condition.notify_all();
}

void TestHooks::fail_next_advance_after_tokens(std::size_t token_count) {
  std::lock_guard lock(state_->mutex);
  state_->fail_after_tokens = token_count;
}

void TestHooks::fail_next_keep() {
  std::lock_guard lock(state_->mutex);
  state_->fail_keep = true;
}

void TestHooks::force_next_branch_generation_for_testing(
    std::uint64_t generation) {
  std::lock_guard lock(state_->mutex);
  state_->generation_override = generation;
}

void TestHooks::wait_at_advance_gate() {
  std::unique_lock lock(state_->mutex);
  if (!state_->block_next_advance) {
    return;
  }
  state_->advance_blocked = true;
  state_->condition.notify_all();
  state_->condition.wait(lock, [this] { return state_->release_advance; });
  state_->block_next_advance = false;
  state_->advance_blocked = false;
  state_->release_advance = false;
}

bool TestHooks::fail_after_token_count(std::size_t token_count) {
  std::lock_guard lock(state_->mutex);
  if (!state_->fail_after_tokens.has_value() ||
      *state_->fail_after_tokens != token_count) {
    return false;
  }
  state_->fail_after_tokens.reset();
  return true;
}

bool TestHooks::consume_keep_failure() {
  std::lock_guard lock(state_->mutex);
  if (!state_->fail_keep) {
    return false;
  }
  state_->fail_keep = false;
  return true;
}

std::optional<std::uint64_t> TestHooks::consume_generation_override() {
  std::lock_guard lock(state_->mutex);
  auto value = state_->generation_override;
  state_->generation_override.reset();
  return value;
}

struct RuntimeContext {
  static constexpr std::uint64_t kNoBranchSlot =
      std::numeric_limits<std::uint64_t>::max();

  struct BranchSlot {
    std::uint64_t generation = 0;
    bool active = false;
    bool exhausted = false;
    std::uint64_t next_free = kNoBranchSlot;
  };

  explicit RuntimeContext(Config config_value) : config(std::move(config_value)) {}

  Config config;
  mutable std::mutex registry_mutex;
  std::vector<BranchSlot> branch_slots;
  std::uint64_t free_branch_head = kNoBranchSlot;
  std::uint64_t next_snapshot_id = 1;

  void release_branch_slot(std::uint64_t id,
                           std::uint64_t generation) noexcept {
    std::lock_guard lock(registry_mutex);
    if (id >= branch_slots.size()) {
      return;
    }
    auto& slot = branch_slots[id];
    if (!slot.active || slot.generation != generation) {
      return;
    }
    slot.active = false;
    if (slot.generation == std::numeric_limits<std::uint64_t>::max()) {
      slot.exhausted = true;
      slot.next_free = kNoBranchSlot;
    } else {
      slot.next_free = free_branch_head;
      free_branch_head = id;
    }
  }
};

struct Branch::Control {
  explicit Control(std::shared_ptr<RuntimeContext> context_value,
                   std::uint64_t branch_id,
                   std::uint64_t branch_generation,
                   Snapshot::State initial_state)
      : context(std::move(context_value)),
        id(branch_id),
        generation(branch_generation),
        state(std::move(initial_state)) {}

  std::shared_ptr<RuntimeContext> context;
  std::uint64_t id = 0;
  std::uint64_t generation = 0;
  mutable std::mutex mutex;
  std::atomic<bool> abandon_requested{false};

  enum class Status { Ready, Advancing, Kept, Discarded };
  Status status = Status::Ready;
  Snapshot::State state;
};

const char* to_string(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::InvalidArgument:
      return "invalid_argument";
    case ErrorCode::InvalidHandle:
      return "invalid_handle";
    case ErrorCode::StaleHandle:
      return "stale_handle";
    case ErrorCode::WrongRuntime:
      return "wrong_runtime";
    case ErrorCode::Busy:
      return "busy";
    case ErrorCode::InvalidToken:
      return "invalid_token";
    case ErrorCode::PositionLimit:
      return "position_limit";
    case ErrorCode::ResourceExhausted:
      return "resource_exhausted";
    case ErrorCode::InternalFailure:
      return "internal_failure";
  }
  return "unknown";
}

void Snapshot::reset() noexcept {
  context_.reset();
  state_.reset();
  id_ = 0;
}

namespace {

bool valid_config(const Config& config, std::string* reason) {
  if (config.vocabulary_size == 0) {
    if (reason != nullptr) {
      *reason = "vocabulary_size must be greater than zero";
    }
    return false;
  }
  if (config.recurrent_width == 0) {
    if (reason != nullptr) {
      *reason = "recurrent_width must be greater than zero";
    }
    return false;
  }
  if (config.adapter_id.empty()) {
    if (reason != nullptr) {
      *reason = "adapter_id must not be empty";
    }
    return false;
  }
  return true;
}

std::shared_ptr<Snapshot::State> make_root_state(const Config& config) {
  auto state = std::make_shared<Snapshot::State>();
  state->recurrent_state.assign(config.recurrent_width, 0);
  state->rng_state = config.seed;
  return state;
}

StateView view_of(const Snapshot::State& state) {
  return StateView{state.position, state.attention_tokens, state.recurrent_state,
                   state.rng_state};
}

std::vector<float> logits_for(const Snapshot::State& state,
                              const Config& config) {
  std::vector<float> logits(config.vocabulary_size, 0.0F);
  for (std::uint32_t token = 0; token < config.vocabulary_size; ++token) {
    const std::size_t recurrent_index =
        static_cast<std::size_t>(token) % state.recurrent_state.size();
    const auto recurrent = static_cast<std::uint64_t>(
        state.recurrent_state[recurrent_index]);
    const std::uint64_t mixed = splitmix64(
        state.rng_state ^ (static_cast<std::uint64_t>(token) * kMixConstant) ^
        recurrent ^ (state.position * 0x517cc1b727220a95ULL));
    const auto bucket = static_cast<std::uint32_t>(mixed % 2'000'001ULL);
    logits[token] = static_cast<float>(bucket) / 100'000.0F - 10.0F;
  }
  return logits;
}

void apply_token(Snapshot::State& state, TokenId token) {
  const auto position = state.position;
  state.attention_tokens.push_back(token);
  for (std::size_t index = 0; index < state.recurrent_state.size(); ++index) {
    const auto prior = static_cast<std::uint64_t>(state.recurrent_state[index]);
    const auto contribution =
        (static_cast<std::uint64_t>(token) + 1ULL) * (index + 1ULL) + position;
    state.recurrent_state[index] = static_cast<std::int64_t>(
        (prior * 1'315'423'911ULL + contribution) % kRecurrentModulus);
  }
  state.rng_state = splitmix64(
      state.rng_state ^ (static_cast<std::uint64_t>(token) + 1ULL) ^
      (position * kMixConstant));
  ++state.position;
}

}  // namespace

void Branch::abandon_noexcept() noexcept {
  if (!control_ || !active_) {
    return;
  }
  control_->abandon_requested.store(true, std::memory_order_release);
  std::unique_lock lock(control_->mutex, std::try_to_lock);
  if (!lock.owns_lock()) {
    return;
  }
  if (control_->status == Branch::Control::Status::Ready) {
    control_->status = Branch::Control::Status::Discarded;
    control_->state = Snapshot::State{};
    control_->context->release_branch_slot(control_->id, control_->generation);
  }
}

Branch::~Branch() { abandon_noexcept(); }

Branch::Branch(Branch&& other) noexcept
    : control_(std::move(other.control_)),
      id_(other.id_),
      generation_(other.generation_),
      active_(other.active_) {
  other.id_ = 0;
  other.generation_ = 0;
  other.active_ = false;
}

Branch& Branch::operator=(Branch&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  abandon_noexcept();
  control_ = std::move(other.control_);
  id_ = other.id_;
  generation_ = other.generation_;
  active_ = other.active_;
  other.id_ = 0;
  other.generation_ = 0;
  other.active_ = false;
  return *this;
}

Runtime::Runtime() : Runtime(Config{}) {}

Runtime::Runtime(Config config) {
  std::string reason;
  if (!valid_config(config, &reason)) {
    throw std::invalid_argument(reason);
  }
  context_ = std::make_shared<RuntimeContext>(std::move(config));
}

Runtime::Runtime(std::shared_ptr<RuntimeContext> context)
    : context_(std::move(context)) {}

Runtime::~Runtime() = default;

Result<Runtime> Runtime::create(Config config) {
  std::string reason;
  if (!valid_config(config, &reason)) {
    return make_error(ErrorCode::InvalidArgument, std::move(reason));
  }
  try {
    return Runtime(std::make_shared<RuntimeContext>(std::move(config)));
  } catch (const std::bad_alloc&) {
    return make_error(ErrorCode::ResourceExhausted,
                      "unable to allocate runtime context");
  }
}

const Config& Runtime::config() const noexcept { return context_->config; }

Result<Snapshot> Runtime::root_snapshot() const {
  if (!context_) {
    return make_error(ErrorCode::InvalidHandle, "runtime is not initialized");
  }
  std::uint64_t id = 0;
  {
    std::lock_guard lock(context_->registry_mutex);
    id = context_->next_snapshot_id++;
  }
  try {
    return Snapshot(make_root_state(context_->config), context_, id);
  } catch (const std::bad_alloc&) {
    return make_error(ErrorCode::ResourceExhausted,
                      "unable to allocate root snapshot state");
  }
}

Result<Branch> Runtime::fork(const Snapshot& snapshot) const {
  if (!context_) {
    return make_error(ErrorCode::InvalidHandle, "runtime is not initialized");
  }
  if (!snapshot.valid()) {
    return make_error(ErrorCode::InvalidHandle, "snapshot handle is empty");
  }
  if (snapshot.context_.get() != context_.get()) {
    return make_error(ErrorCode::WrongRuntime,
                      "snapshot belongs to a different runtime");
  }

  std::uint64_t id = 0;
  std::uint64_t generation = 0;
  std::optional<std::uint64_t> forced_generation;
  if (context_->config.test_hooks) {
    forced_generation =
        context_->config.test_hooks->consume_generation_override();
    if (forced_generation && *forced_generation == 0) {
      return make_error(ErrorCode::InvalidArgument,
                        "branch generation must be greater than zero");
    }
  }
  try {
    std::lock_guard lock(context_->registry_mutex);
    while (context_->free_branch_head != RuntimeContext::kNoBranchSlot &&
           context_->branch_slots[context_->free_branch_head].exhausted) {
      context_->free_branch_head =
          context_->branch_slots[context_->free_branch_head].next_free;
    }
    if (context_->free_branch_head == RuntimeContext::kNoBranchSlot) {
      id = context_->branch_slots.size();
      context_->branch_slots.push_back({});
    } else {
      id = context_->free_branch_head;
      context_->free_branch_head = context_->branch_slots[id].next_free;
    }
    auto& slot = context_->branch_slots[id];
    slot.next_free = RuntimeContext::kNoBranchSlot;
    if (forced_generation) {
      slot.generation = *forced_generation - 1;
      slot.exhausted = false;
    }
    if (slot.exhausted ||
        slot.generation == std::numeric_limits<std::uint64_t>::max()) {
      // An exhausted slot is never put back on the free list. This branch is
      // defensive for a malformed test hook; normal exhausted slots are
      // skipped above and a fresh slot is allocated instead.
      slot.active = false;
      slot.next_free = RuntimeContext::kNoBranchSlot;
      return make_error(ErrorCode::ResourceExhausted,
                        "branch slot generation is exhausted");
    }
    ++slot.generation;
    slot.active = true;
    generation = slot.generation;
  } catch (const std::bad_alloc&) {
    return make_error(ErrorCode::ResourceExhausted,
                      "unable to allocate a branch registry slot");
  }

  try {
    auto control = std::make_shared<Branch::Control>(
        context_, id, generation, *snapshot.state_);
    return Branch(std::move(control), id, generation);
  } catch (const std::bad_alloc&) {
    std::lock_guard lock(context_->registry_mutex);
    auto& slot = context_->branch_slots[id];
    if (slot.active && slot.generation == generation) {
      slot.active = false;
      if (slot.generation == std::numeric_limits<std::uint64_t>::max()) {
        slot.exhausted = true;
        slot.next_free = RuntimeContext::kNoBranchSlot;
      } else {
        slot.next_free = context_->free_branch_head;
        context_->free_branch_head = id;
      }
    }
    return make_error(ErrorCode::ResourceExhausted,
                      "unable to allocate eager branch state");
  }
}

Result<AdvanceResult> Runtime::advance(Branch& branch,
                                       std::span<const TokenId> token_ids) const {
  if (!context_) {
    return make_error(ErrorCode::InvalidHandle, "runtime is not initialized");
  }
  if (!branch.control_) {
    return make_error(ErrorCode::InvalidHandle, "branch handle is empty");
  }
  auto control = branch.control_;
  if (!control->context || control->context.get() != context_.get()) {
    return make_error(ErrorCode::WrongRuntime,
                      "branch belongs to a different runtime");
  }

  std::unique_lock lock(control->mutex, std::try_to_lock);
  if (!lock.owns_lock()) {
    return make_error(ErrorCode::Busy, "branch has a mutating operation in progress");
  }
  {
    std::lock_guard registry_lock(context_->registry_mutex);
    if (control->id >= context_->branch_slots.size()) {
      return make_error(ErrorCode::StaleHandle, "branch slot no longer exists");
    }
    const auto& slot = context_->branch_slots[control->id];
    if (!slot.active || slot.generation != control->generation) {
      return make_error(ErrorCode::StaleHandle,
                        "branch handle generation is no longer active");
    }
  }
  if (control->status != Branch::Control::Status::Ready) {
    return make_error(ErrorCode::StaleHandle, "branch is no longer mutable");
  }

  if (token_ids.size() >
      static_cast<std::size_t>(context_->config.max_position -
                               std::min(context_->config.max_position,
                                        control->state.position))) {
    return make_error(ErrorCode::PositionLimit,
                      "token sequence exceeds the configured position limit");
  }
  for (const TokenId token : token_ids) {
    if (token >= context_->config.vocabulary_size) {
      return make_error(ErrorCode::InvalidToken,
                        "token ID is outside the configured vocabulary");
    }
  }

  // All mutations happen to this local eager copy. A rejected allocation or
  // computation therefore leaves the published branch boundary untouched.
  control->status = Branch::Control::Status::Advancing;
  try {
    if (context_->config.test_hooks) {
      context_->config.test_hooks->wait_at_advance_gate();
    }
    Snapshot::State next = control->state;
    std::size_t applied_tokens = 0;
    for (const TokenId token : token_ids) {
      apply_token(next, token);
      ++applied_tokens;
      if (context_->config.test_hooks &&
          context_->config.test_hooks->fail_after_token_count(applied_tokens)) {
        throw std::bad_alloc();
      }
    }
    AdvanceResult result{next.position, logits_for(next, context_->config)};
    control->state = std::move(next);
    if (control->abandon_requested.load(std::memory_order_acquire)) {
      control->status = Branch::Control::Status::Discarded;
      control->state = Snapshot::State{};
      context_->release_branch_slot(control->id, control->generation);
    } else {
      control->status = Branch::Control::Status::Ready;
    }
    return result;
  } catch (const std::bad_alloc&) {
    if (control->abandon_requested.load(std::memory_order_acquire)) {
      control->status = Branch::Control::Status::Discarded;
      control->state = Snapshot::State{};
      context_->release_branch_slot(control->id, control->generation);
    } else {
      control->status = Branch::Control::Status::Ready;
    }
    return make_error(ErrorCode::ResourceExhausted,
                      "unable to allocate the advanced branch state");
  } catch (const std::exception& exception) {
    if (control->abandon_requested.load(std::memory_order_acquire)) {
      control->status = Branch::Control::Status::Discarded;
      control->state = Snapshot::State{};
      context_->release_branch_slot(control->id, control->generation);
    } else {
      control->status = Branch::Control::Status::Ready;
    }
    return make_error(ErrorCode::InternalFailure,
                      std::string("reference advance failed: ") + exception.what());
  } catch (...) {
    if (control->abandon_requested.load(std::memory_order_acquire)) {
      control->status = Branch::Control::Status::Discarded;
      control->state = Snapshot::State{};
      context_->release_branch_slot(control->id, control->generation);
    } else {
      control->status = Branch::Control::Status::Ready;
    }
    return make_error(ErrorCode::InternalFailure, "reference advance failed");
  }
}

Result<std::optional<Snapshot>> Runtime::discard_or_keep(
    Branch& branch,
    BranchDecision decision) const {
  if (!context_) {
    return make_error(ErrorCode::InvalidHandle, "runtime is not initialized");
  }
  if (decision != BranchDecision::Keep && decision != BranchDecision::Discard) {
    return make_error(ErrorCode::InvalidArgument,
                      "branch decision must be Keep or Discard");
  }
  if (!branch.control_) {
    return make_error(ErrorCode::InvalidHandle, "branch handle is empty");
  }
  auto control = branch.control_;
  if (!control->context || control->context.get() != context_.get()) {
    return make_error(ErrorCode::WrongRuntime,
                      "branch belongs to a different runtime");
  }
  std::unique_lock lock(control->mutex, std::try_to_lock);
  if (!lock.owns_lock()) {
    return make_error(ErrorCode::Busy, "branch has a mutating operation in progress");
  }
  {
    std::lock_guard registry_lock(context_->registry_mutex);
    if (control->id >= context_->branch_slots.size()) {
      return make_error(ErrorCode::StaleHandle, "branch slot no longer exists");
    }
    const auto& slot = context_->branch_slots[control->id];
    if (!slot.active || slot.generation != control->generation) {
      return make_error(ErrorCode::StaleHandle,
                        "branch handle generation is no longer active");
    }
  }
  if (control->status != Branch::Control::Status::Ready) {
    return make_error(ErrorCode::Busy, "branch is not at a completed boundary");
  }

  std::optional<Snapshot> kept;
  try {
    if (decision == BranchDecision::Keep) {
      std::uint64_t snapshot_id = 0;
      {
        std::lock_guard registry_lock(context_->registry_mutex);
        snapshot_id = context_->next_snapshot_id++;
      }
      if (context_->config.test_hooks &&
          context_->config.test_hooks->consume_keep_failure()) {
        throw std::bad_alloc();
      }
      auto immutable_state =
          std::make_shared<const Snapshot::State>(control->state);
      kept = Snapshot(std::move(immutable_state),
                      std::shared_ptr<const RuntimeContext>(context_), snapshot_id);
      control->status = Branch::Control::Status::Kept;
    } else {
      control->status = Branch::Control::Status::Discarded;
    }

    {
      std::lock_guard registry_lock(context_->registry_mutex);
      auto& slot = context_->branch_slots[control->id];
      if (slot.active && slot.generation == control->generation) {
        slot.active = false;
        if (slot.generation == std::numeric_limits<std::uint64_t>::max()) {
          slot.exhausted = true;
          slot.next_free = RuntimeContext::kNoBranchSlot;
        } else {
          slot.next_free = context_->free_branch_head;
          context_->free_branch_head = control->id;
        }
      }
    }
    // There can be no submitted operation in this synchronous runtime once
    // the branch mutex is held. Clearing the mutable copy releases its eager
    // storage after logical invalidation.
    control->state = Snapshot::State{};
    branch.active_ = false;
    return kept;
  } catch (const std::bad_alloc&) {
    return make_error(ErrorCode::ResourceExhausted,
                      "unable to publish the kept snapshot");
  } catch (const std::exception& exception) {
    return make_error(ErrorCode::InternalFailure,
                      std::string("branch finalization failed: ") + exception.what());
  }
}

Result<StateView> Runtime::inspect(const Snapshot& snapshot) const {
  if (!context_) {
    return make_error(ErrorCode::InvalidHandle, "runtime is not initialized");
  }
  if (!snapshot.valid()) {
    return make_error(ErrorCode::InvalidHandle, "snapshot handle is empty");
  }
  if (snapshot.context_.get() != context_.get()) {
    return make_error(ErrorCode::WrongRuntime,
                      "snapshot belongs to a different runtime");
  }
  return view_of(*snapshot.state_);
}

Result<StateView> Runtime::inspect(const Branch& branch) const {
  if (!context_) {
    return make_error(ErrorCode::InvalidHandle, "runtime is not initialized");
  }
  if (!branch.control_) {
    return make_error(ErrorCode::InvalidHandle, "branch handle is empty");
  }
  auto control = branch.control_;
  if (!control->context || control->context.get() != context_.get()) {
    return make_error(ErrorCode::WrongRuntime,
                      "branch belongs to a different runtime");
  }
  std::unique_lock lock(control->mutex, std::try_to_lock);
  if (!lock.owns_lock()) {
    return make_error(ErrorCode::Busy, "branch has a mutating operation in progress");
  }
  {
    std::lock_guard registry_lock(context_->registry_mutex);
    if (control->id >= context_->branch_slots.size()) {
      return make_error(ErrorCode::StaleHandle, "branch slot no longer exists");
    }
    const auto& slot = context_->branch_slots[control->id];
    if (!slot.active || slot.generation != control->generation) {
      return make_error(ErrorCode::StaleHandle,
                        "branch handle generation is no longer active");
    }
  }
  if (control->status != Branch::Control::Status::Ready) {
    return make_error(ErrorCode::Busy, "branch is not at a readable boundary");
  }
  return view_of(control->state);
}

}  // namespace branchforge
