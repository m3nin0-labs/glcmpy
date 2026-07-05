cmake_minimum_required(VERSION 3.18...3.30)
project(glcmpy LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python + nanobind
find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module
)

# Locate the nanobind CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# Eigen (header-only, fetched so no system install is needed)
include(FetchContent)

# Only the headers are needed. `SOURCE_SUBDIR none` makes MakeAvailable just
# fetch the source without running Eigen own CMake (no targets/tests, and no
# dependency on Eigen minimum-CMake-version requirements).
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR none
)
FetchContent_MakeAvailable(Eigen3)

# Optional OpenMP
find_package(OpenMP QUIET)

# Extension module
nanobind_add_module(_glcm STABLE_ABI NB_STATIC src/glcm_bindings.cpp)

target_include_directories(_glcm PRIVATE
  "${eigen3_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(_glcm PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(_glcm PRIVATE GLCMPY_HAVE_OPENMP)
endif()

install(TARGETS _glcm LIBRARY DESTINATION glcmpy)
