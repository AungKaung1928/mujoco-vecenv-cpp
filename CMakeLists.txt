cmake_minimum_required(VERSION 3.18)
project(mujoco_vecenv_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(VECENV_BUILD_PYTHON "Build the pybind11 module" ON)
option(VECENV_BUILD_TESTS "Build the doctest binary" ON)
option(VECENV_BUILD_BENCH "Build the benchmark" ON)
set(VECENV_SANITIZE "" CACHE STRING "Comma list for -fsanitize=, e.g. address,undefined or thread")

# No fast-math and no FMA contraction anywhere: the whole point of this repo is
# to produce the same bits as the Python reference, and a fused multiply-add
# rounds once where numpy rounds twice.
add_compile_options(-Wall -Wextra -ffp-contract=off -fno-fast-math)
if(VECENV_SANITIZE)
  add_compile_options(-fsanitize=${VECENV_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${VECENV_SANITIZE})
endif()

find_package(Threads REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter REQUIRED)

# MuJoCo comes from the pip wheel the Python side uses: headers under
# mujoco/include, the shared library next to them. Nothing to install.
if(NOT MUJOCO_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))"
    OUTPUT_VARIABLE MUJOCO_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0 OR NOT MUJOCO_DIR)
    message(FATAL_ERROR "Could not import mujoco with ${Python_EXECUTABLE}. pip install mujoco==3.12.0, or pass -DMUJOCO_DIR=<dir containing include/ and libmujoco.so*>.")
  endif()
endif()
file(GLOB _mujoco_libs "${MUJOCO_DIR}/libmujoco.so*")
list(SORT _mujoco_libs)
if(NOT _mujoco_libs)
  message(FATAL_ERROR "No libmujoco.so* under ${MUJOCO_DIR}")
endif()
list(GET _mujoco_libs -1 MUJOCO_LIBRARY)
message(STATUS "MuJoCo: ${MUJOCO_LIBRARY}")
add_library(mujoco::mujoco SHARED IMPORTED GLOBAL)
set_target_properties(mujoco::mujoco PROPERTIES
  IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_DIR}/include")

add_library(vecenv STATIC
  src/boxcheck.cpp src/dr.cpp src/env.cpp src/model.cpp src/rng.cpp src/vecenv.cpp)
target_include_directories(vecenv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vecenv PUBLIC mujoco::mujoco Threads::Threads)
target_compile_definitions(vecenv PRIVATE VECENV_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# The library has a versioned SONAME and lives in site-packages; binaries need
# an rpath to find it when Python has not already loaded it.
set(_rpath_props BUILD_RPATH "${MUJOCO_DIR}" INSTALL_RPATH "${MUJOCO_DIR}" BUILD_WITH_INSTALL_RPATH ON)

if(VECENV_BUILD_TESTS)
  add_executable(vecenv_tests
    tests/cpp/main.cpp tests/cpp/test_numpy_compat.cpp tests/cpp/test_rng.cpp
    tests/cpp/test_env.cpp tests/cpp/test_dr.cpp tests/cpp/test_vecenv.cpp)
  target_include_directories(vecenv_tests PRIVATE third_party/doctest)
  target_link_libraries(vecenv_tests PRIVATE vecenv)
  set_target_properties(vecenv_tests PROPERTIES ${_rpath_props})
  enable_testing()
  add_test(NAME vecenv_tests COMMAND vecenv_tests)
endif()

if(VECENV_BUILD_BENCH)
  add_executable(bench_vecenv bench/bench_vecenv.cpp)
  target_link_libraries(bench_vecenv PRIVATE vecenv)
  set_target_properties(bench_vecenv PROPERTIES ${_rpath_props})
endif()

if(VECENV_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
  if(NOT pybind11_DIR)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_vecenv_cpp python/bindings.cpp)
  target_link_libraries(_vecenv_cpp PRIVATE vecenv)
  set_target_properties(_vecenv_cpp PROPERTIES ${_rpath_props}
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/vecenv_cpp")
endif()
