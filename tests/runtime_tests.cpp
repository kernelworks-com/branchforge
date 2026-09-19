#include "branchforge/branchforge.hpp"

#include <array>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using branchforge::Branch;
using branchforge::BranchDecision;
using branchforge::Config;
using branchforge::ErrorCode;
using branchforge::Runtime;
using branchforge::Snapshot;
using branchforge::TestHooks;
using branchforge::TokenId;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename T>
void require(const branchforge::Result<T>& result, const std::string& message) {
  require(result.has_value(), message);
}

template <typename T>
void require_error(const branchforge::Result<T>& result,
                   ErrorCode expected,
                   const std::string& context) {
  require(!result, context + ": operation unexpectedly succeeded");
  require(result.error().code == expected,
          context + ": got " + branchforge::to_string(result.error().code));
}

void require_view(const branchforge::Result<branchforge::StateView>& result,
                  std::uint64_t position,
                  std::span<const TokenId> tokens,
                  const std::string& context) {
  require(result, context + ": inspect failed");
  require(result.value().position == position,
          context + ": unexpected position");
  require(result.value().attention_tokens.size() == tokens.size(),
          context + ": unexpected token count");
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    require(result.value().attention_tokens[index] == tokens[index],
            context + ": unexpected token history");
  }
}

void test_independent_branches_and_keep() {
  const std::array<TokenId, 0> no_tokens{};
  Runtime runtime;
  const auto root_result = runtime.root_snapshot();
  require(root_result, "root snapshot creation failed");
  const Snapshot root = root_result.value();

  auto left_result = runtime.fork(root);
  auto right_result = runtime.fork(root);
  require(left_result && right_result, "fork from root failed");
  Branch left = std::move(left_result).value();
  Branch right = std::move(right_result).value();

  const std::array<TokenId, 2> left_tokens{1, 2};
  const std::array<TokenId, 1> right_tokens{3};
  require(runtime.advance(left, left_tokens), "left advance failed");
  require(runtime.advance(right, right_tokens), "right advance failed");
  require_view(runtime.inspect(left), 2, left_tokens, "left branch");
  require_view(runtime.inspect(right), 1, right_tokens, "right branch");
  require_view(runtime.inspect(root), 0, no_tokens, "root snapshot");

  const auto kept_result =
      runtime.discard_or_keep(left, BranchDecision::Keep);
  require(kept_result && kept_result.value().has_value(),
          "KEEP did not publish a snapshot");
  const Snapshot kept = *kept_result.value();
  require(!left.valid(), "discard_or_keep should invalidate the branch handle");
  require_view(runtime.inspect(kept), 2, left_tokens, "kept snapshot");
  require_error(runtime.inspect(left), ErrorCode::StaleHandle,
                "inspect kept branch");
  require_error(runtime.advance(left, no_tokens), ErrorCode::StaleHandle,
                "advance kept branch");

  auto child_result = runtime.fork(kept);
  require(child_result, "fork from kept snapshot failed");
  Branch child = std::move(child_result).value();
  const std::array<TokenId, 1> child_tokens{4};
  require(runtime.advance(child, child_tokens), "child advance failed");
  const std::array<TokenId, 3> child_history{1, 2, 4};
  require_view(runtime.inspect(child), 3, child_history, "child branch");
  require_view(runtime.inspect(kept), 2, left_tokens,
               "kept snapshot after child mutation");

  const auto discarded_result =
      runtime.discard_or_keep(right, BranchDecision::Discard);
  require(discarded_result && !discarded_result.value().has_value(),
          "DISCARD returned a snapshot");
  require_error(runtime.inspect(right), ErrorCode::StaleHandle,
                "inspect discarded branch");

  // The slot is reusable, but its generation changes. The old object remains
  // a checked stale handle even after the allocator reuses its slot.
  auto reused_result = runtime.fork(root);
  require(reused_result, "slot reuse fork failed");
  Branch reused = std::move(reused_result).value();
  require(reused.id() == right.id(), "branch slot was not reused");
  require(reused.generation() != right.generation(),
          "branch generation did not change on reuse");
  require_error(runtime.advance(right, no_tokens), ErrorCode::StaleHandle,
                "advance stale generation");
  require(runtime.discard_or_keep(child, BranchDecision::Discard),
          "child cleanup failed");
  require(runtime.discard_or_keep(reused, BranchDecision::Discard),
          "reused branch cleanup failed");
}

void test_supplied_tokens_and_atomic_failures() {
  const std::array<TokenId, 0> no_tokens{};
  Config config;
  config.vocabulary_size = 8;
  config.recurrent_width = 3;
  config.max_position = 4;
  Runtime runtime(config);
  const auto root = runtime.root_snapshot().value();
  auto branch_result = runtime.fork(root);
  require(branch_result, "atomic test fork failed");
  Branch branch = std::move(branch_result).value();

  const std::array<TokenId, 2> invalid_tokens{1, 8};
  require_error(runtime.advance(branch, invalid_tokens), ErrorCode::InvalidToken,
                "invalid token sequence");
  require_view(runtime.inspect(branch), 0, no_tokens,
               "state after invalid token sequence");

  const std::array<TokenId, 2> first_tokens{1, 2};
  require(runtime.advance(branch, first_tokens), "initial token advance failed");
  const auto before_limit = runtime.inspect(branch);
  const std::array<TokenId, 3> over_limit{3, 4, 5};
  require_error(runtime.advance(branch, over_limit), ErrorCode::PositionLimit,
                "position-limit sequence");
  const auto after_limit = runtime.inspect(branch);
  require(before_limit && after_limit, "limit inspection failed");
  require(before_limit.value().position == after_limit.value().position,
          "position changed after rejected sequence");
  require(before_limit.value().attention_tokens ==
              after_limit.value().attention_tokens,
          "attention state changed after rejected sequence");
  require(before_limit.value().recurrent_state ==
              after_limit.value().recurrent_state,
          "recurrent state changed after rejected sequence");
  require(before_limit.value().rng_state == after_limit.value().rng_state,
          "sampling state changed after rejected sequence");

  const auto zero_result = runtime.advance(branch, no_tokens);
  require(zero_result, "zero-token advance failed");
  require(zero_result.value().position == 2,
          "zero-token advance changed the position");
  require_view(runtime.inspect(branch), 2, first_tokens,
               "state after zero-token advance");
  require(runtime.discard_or_keep(branch, BranchDecision::Discard),
          "atomic test cleanup failed");
}

void test_checked_contexts_and_handles() {
  const std::array<TokenId, 0> no_tokens{};
  Runtime first;
  Runtime second;
  const Snapshot first_root = first.root_snapshot().value();
  const Snapshot second_root = second.root_snapshot().value();

  require_error(first.fork(second_root), ErrorCode::WrongRuntime,
                "cross-runtime snapshot fork");
  auto branch_result = first.fork(first_root);
  require(branch_result, "checked handle fork failed");
  Branch branch = std::move(branch_result).value();
  require_error(second.advance(branch, no_tokens), ErrorCode::WrongRuntime,
                "cross-runtime branch advance");
  Branch empty_branch;
  require_error(first.advance(empty_branch, no_tokens), ErrorCode::InvalidHandle,
                "empty branch advance");
  require_error(first.inspect(Snapshot{}), ErrorCode::InvalidHandle,
                "empty snapshot inspect");

  Branch moved = std::move(branch);
  require(!branch.valid() && moved.valid(), "branch move state is invalid");
  require_error(first.advance(branch, no_tokens), ErrorCode::InvalidHandle,
                "moved-from branch advance");
  require(first.discard_or_keep(moved, BranchDecision::Discard),
          "checked handle cleanup failed");

  Config invalid;
  invalid.vocabulary_size = 0;
  const auto invalid_runtime = Runtime::create(invalid);
  require_error(invalid_runtime, ErrorCode::InvalidArgument,
                "invalid runtime configuration");
}

void test_duplicate_finalization_is_rejected() {
  Runtime runtime;
  const auto root = runtime.root_snapshot().value();
  auto branch_result = runtime.fork(root);
  require(branch_result, "duplicate-finalization fork failed");
  Branch branch = std::move(branch_result).value();
  require(runtime.discard_or_keep(branch, BranchDecision::Discard),
          "first discard failed");
  require_error(runtime.discard_or_keep(branch, BranchDecision::Discard),
                ErrorCode::StaleHandle, "duplicate discard");
}

void test_invalid_decision_is_atomic() {
  Runtime runtime;
  const auto root = runtime.root_snapshot().value();
  auto branch_result = runtime.fork(root);
  require(branch_result, "invalid-decision fork failed");
  Branch branch = std::move(branch_result).value();
  const auto invalid = static_cast<BranchDecision>(99);
  require_error(runtime.discard_or_keep(branch, invalid),
                ErrorCode::InvalidArgument, "invalid branch decision");
  require(branch.valid(), "invalid decision changed branch validity");
  require(runtime.discard_or_keep(branch, BranchDecision::Discard),
          "invalid-decision cleanup failed");
}

void test_full_state_and_logits_match_eager_reference() {
  Runtime runtime;
  const auto root = runtime.root_snapshot().value();
  auto first_result = runtime.fork(root);
  auto second_result = runtime.fork(root);
  require(first_result && second_result, "eager-equivalence fork failed");
  Branch first = std::move(first_result).value();
  Branch second = std::move(second_result).value();

  const std::array<TokenId, 4> history{2, 5, 7, 11};
  const auto first_advance = runtime.advance(first, history);
  const auto second_advance = runtime.advance(second, history);
  require(first_advance && second_advance, "eager-equivalence advance failed");
  require(first_advance.value().position == second_advance.value().position,
          "eager-equivalence positions differ");
  require(first_advance.value().logits == second_advance.value().logits,
          "eager-equivalence logits differ");
  const auto first_state = runtime.inspect(first);
  const auto second_state = runtime.inspect(second);
  require(first_state && second_state, "eager-equivalence inspect failed");
  require(first_state.value().position == second_state.value().position,
          "eager-equivalence position state differs");
  require(first_state.value().attention_tokens ==
              second_state.value().attention_tokens,
          "eager-equivalence attention state differs");
  require(first_state.value().recurrent_state ==
              second_state.value().recurrent_state,
          "eager-equivalence recurrent state differs");
  require(first_state.value().rng_state == second_state.value().rng_state,
          "eager-equivalence RNG state differs");

  require(runtime.discard_or_keep(first, BranchDecision::Discard),
          "eager-equivalence first cleanup failed");
  require(runtime.discard_or_keep(second, BranchDecision::Discard),
          "eager-equivalence second cleanup failed");
}

void test_advance_failure_is_atomic_after_partial_candidate_update() {
  auto hooks = std::make_shared<TestHooks>();
  Config config;
  config.test_hooks = hooks;
  Runtime runtime(config);
  const auto root = runtime.root_snapshot().value();
  auto branch_result = runtime.fork(root);
  require(branch_result, "fault-injection fork failed");
  Branch branch = std::move(branch_result).value();
  const auto before = runtime.inspect(branch);
  require(before, "fault-injection pre-inspect failed");

  hooks->fail_next_advance_after_tokens(1);
  const std::array<TokenId, 3> tokens{1, 2, 3};
  const auto failed = runtime.advance(branch, tokens);
  require_error(failed, ErrorCode::ResourceExhausted,
                "fault-injected partial advance");
  const auto after = runtime.inspect(branch);
  require(after, "fault-injection post-inspect failed");
  require(after.value().position == before.value().position,
          "failed advance published a new position");
  require(after.value().attention_tokens == before.value().attention_tokens,
          "failed advance published attention state");
  require(after.value().recurrent_state == before.value().recurrent_state,
          "failed advance published recurrent state");
  require(after.value().rng_state == before.value().rng_state,
          "failed advance published RNG state");

  require(runtime.discard_or_keep(branch, BranchDecision::Discard),
          "fault-injection cleanup failed");
}

void test_keep_failure_is_atomic() {
  auto hooks = std::make_shared<TestHooks>();
  Config config;
  config.test_hooks = hooks;
  Runtime runtime(config);
  const auto root = runtime.root_snapshot().value();
  auto branch_result = runtime.fork(root);
  require(branch_result, "keep-failure fork failed");
  Branch branch = std::move(branch_result).value();
  const std::array<TokenId, 2> tokens{1, 2};
  require(runtime.advance(branch, tokens), "keep-failure advance failed");

  hooks->fail_next_keep();
  const auto failed_keep =
      runtime.discard_or_keep(branch, BranchDecision::Keep);
  require_error(failed_keep, ErrorCode::ResourceExhausted,
                "fault-injected keep");
  require(branch.valid(), "failed KEEP consumed the branch");
  require_view(runtime.inspect(branch), 2, tokens,
               "state after failed KEEP");

  const auto successful_keep =
      runtime.discard_or_keep(branch, BranchDecision::Keep);
  require(successful_keep && successful_keep.value().has_value(),
          "retry KEEP failed");
  require_view(runtime.inspect(*successful_keep.value()), 2, tokens,
               "snapshot after retry KEEP");
}

void test_busy_is_deterministic() {
  auto hooks = std::make_shared<TestHooks>();
  Config config;
  config.test_hooks = hooks;
  Runtime runtime(config);
  const auto root = runtime.root_snapshot().value();
  auto branch_result = runtime.fork(root);
  require(branch_result, "Busy fork failed");
  Branch branch = std::move(branch_result).value();
  const std::array<TokenId, 1> token{1};
  hooks->block_next_advance();

  std::optional<branchforge::Result<branchforge::AdvanceResult>> first_result;
  std::thread worker([&] { first_result = runtime.advance(branch, token); });
  hooks->wait_until_advance_blocked();
  const auto busy = runtime.advance(branch, token);
  require_error(busy, ErrorCode::Busy, "concurrent branch mutation");
  hooks->release_advance();
  worker.join();
  require(first_result.has_value() && *first_result,
          "gated advance did not complete");
  require_view(runtime.inspect(branch), 1, token,
               "state after gated advance");
  require(runtime.discard_or_keep(branch, BranchDecision::Discard),
          "Busy cleanup failed");
}

void test_move_assignment_releases_destination_slot() {
  Runtime runtime;
  const auto root = runtime.root_snapshot().value();
  auto destination_result = runtime.fork(root);
  auto source_result = runtime.fork(root);
  require(destination_result && source_result, "move-assignment forks failed");
  Branch destination = std::move(destination_result).value();
  Branch source = std::move(source_result).value();
  const auto old_destination_id = destination.id();
  const auto old_destination_generation = destination.generation();

  destination = std::move(source);
  require(destination.valid() && !source.valid(),
          "move assignment did not transfer branch ownership");
  auto replacement_result = runtime.fork(root);
  require(replacement_result, "move-assignment replacement fork failed");
  Branch replacement = std::move(replacement_result).value();
  require(replacement.id() == old_destination_id,
          "move assignment leaked the overwritten destination slot");
  require(replacement.generation() != old_destination_generation,
          "move assignment reused the overwritten generation");
  require(runtime.discard_or_keep(destination, BranchDecision::Discard),
          "move-assignment destination cleanup failed");
  require(runtime.discard_or_keep(replacement, BranchDecision::Discard),
          "move-assignment replacement cleanup failed");
}

void test_generation_exhaustion_is_not_reused() {
  auto hooks = std::make_shared<TestHooks>();
  Config config;
  config.test_hooks = hooks;
  Runtime runtime(config);
  const auto root = runtime.root_snapshot().value();
  hooks->force_next_branch_generation_for_testing(
      std::numeric_limits<std::uint64_t>::max());
  auto exhausted_result = runtime.fork(root);
  require(exhausted_result, "generation-exhaustion fork failed");
  Branch exhausted = std::move(exhausted_result).value();
  const auto exhausted_id = exhausted.id();
  require(exhausted.generation() == std::numeric_limits<std::uint64_t>::max(),
          "test did not reach maximum branch generation");
  require(runtime.discard_or_keep(exhausted, BranchDecision::Discard),
          "generation-exhaustion discard failed");

  auto replacement_result = runtime.fork(root);
  require(replacement_result, "post-exhaustion fork failed");
  Branch replacement = std::move(replacement_result).value();
  require(replacement.id() != exhausted_id,
          "exhausted branch slot was reused after generation wrap");
  require(runtime.discard_or_keep(replacement, BranchDecision::Discard),
          "post-exhaustion cleanup failed");
}

void test_raii_releases_unfinalized_branch() {
  Runtime runtime;
  const auto root = runtime.root_snapshot().value();
  std::uint64_t old_id = 0;
  std::uint64_t old_generation = 0;
  {
    auto branch_result = runtime.fork(root);
    require(branch_result, "RAII fork failed");
    old_id = branch_result.value().id();
    old_generation = branch_result.value().generation();
    // Leaving scope abandons the branch and releases its registry slot.
  }
  auto replacement_result = runtime.fork(root);
  require(replacement_result, "RAII replacement fork failed");
  Branch replacement = std::move(replacement_result).value();
  require(replacement.id() == old_id, "RAII did not release branch slot");
  require(replacement.generation() != old_generation,
          "RAII replacement reused the old generation");
  require(runtime.discard_or_keep(replacement, BranchDecision::Discard),
          "RAII replacement cleanup failed");
}

}  // namespace

int main() {
  try {
    test_independent_branches_and_keep();
    test_supplied_tokens_and_atomic_failures();
    test_checked_contexts_and_handles();
    test_duplicate_finalization_is_rejected();
    test_invalid_decision_is_atomic();
    test_full_state_and_logits_match_eager_reference();
    test_advance_failure_is_atomic_after_partial_candidate_update();
    test_keep_failure_is_atomic();
    test_busy_is_deterministic();
    test_move_assignment_releases_destination_slot();
    test_generation_exhaustion_is_not_reused();
    test_raii_releases_unfinalized_branch();
  } catch (const std::exception& exception) {
    std::cerr << "branchforge_tests: " << exception.what() << '\n';
    return 1;
  }
  std::cout << "branchforge_tests: all semantic checks passed\n";
  return 0;
}
