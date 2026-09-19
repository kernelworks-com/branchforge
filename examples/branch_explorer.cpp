#include "branchforge/branchforge.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

template <typename T>
T take(branchforge::Result<T> result, const char* operation) {
  if (!result) {
    throw std::runtime_error(std::string(operation) + ": " +
                             result.error().message);
  }
  return std::move(result).value();
}

void print_state(const char* label, const branchforge::StateView& state) {
  std::cout << label << " position=" << state.position << " tokens=";
  std::cout << '[';
  for (std::size_t index = 0; index < state.attention_tokens.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << state.attention_tokens[index];
  }
  std::cout << "] recurrent[0]=" << state.recurrent_state.front() << '\n';
}

}  // namespace

int main() {
  try {
    branchforge::Runtime runtime;
    const auto root = take(runtime.root_snapshot(), "root_snapshot");
    auto left = take(runtime.fork(root), "fork left");
    auto right = take(runtime.fork(root), "fork right");

    const std::array<branchforge::TokenId, 3> left_tokens{4, 8, 15};
    const std::array<branchforge::TokenId, 2> right_tokens{16, 23};
    take(runtime.advance(left, left_tokens), "advance left");
    take(runtime.advance(right, right_tokens), "advance right");

    print_state("left", take(runtime.inspect(left), "inspect left"));
    print_state("right", take(runtime.inspect(right), "inspect right"));
    print_state("root", take(runtime.inspect(root), "inspect root"));

    auto retained = take(
        runtime.discard_or_keep(left, branchforge::BranchDecision::Keep),
        "keep left");
    auto kept_snapshot = std::move(*retained);
    auto continuation = take(runtime.fork(kept_snapshot), "fork kept snapshot");
    const std::array<branchforge::TokenId, 1> continuation_token{24};
    take(runtime.advance(continuation, continuation_token),
         "advance continuation");
    print_state("kept", take(runtime.inspect(kept_snapshot), "inspect kept"));
    print_state("continuation",
                take(runtime.inspect(continuation), "inspect continuation"));

    take(runtime.discard_or_keep(right, branchforge::BranchDecision::Discard),
         "discard right");
    take(runtime.discard_or_keep(continuation, branchforge::BranchDecision::Discard),
         "discard continuation");
  } catch (const std::exception& exception) {
    std::cerr << "branchforge_demo: " << exception.what() << '\n';
    return 1;
  }
  return 0;
}
