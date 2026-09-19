cmake_minimum_required(VERSION 3.20)

project(branchforge VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)
find_package(Threads REQUIRED)

add_library(branchforge STATIC
  src/branchforge.cpp
)

target_compile_features(branchforge PUBLIC cxx_std_20)
target_include_directories(branchforge
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(branchforge PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
  target_compile_options(branchforge PRIVATE /W4)
endif()

add_executable(branchforge_demo examples/branch_explorer.cpp)
target_link_libraries(branchforge_demo PRIVATE branchforge)

if(BUILD_TESTING)
  add_executable(branchforge_tests tests/runtime_tests.cpp)
  target_link_libraries(branchforge_tests PRIVATE branchforge Threads::Threads)
  add_test(NAME branchforge_tests COMMAND branchforge_tests)
endif()

install(TARGETS branchforge
  EXPORT branchforgeTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT branchforgeTargets
  FILE branchforgeTargets.cmake
  NAMESPACE branchforge::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/branchforge
)
