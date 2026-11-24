# Builds the pybind11 extension pairslab._risk_cpp. Driven by scikit-build-core (see
# pyproject.toml), so `uv sync` or `uv pip install -e .` compiles it; no manual cmake call.
cmake_minimum_required(VERSION 3.18...3.31)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_risk_cpp bindings.cpp)
target_compile_options(_risk_cpp PRIVATE
  "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>")

# Into the pairslab package, next to risk.py.
install(TARGETS _risk_cpp LIBRARY DESTINATION pairslab)
