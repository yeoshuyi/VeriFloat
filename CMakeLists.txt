cmake_minimum_required(VERSION 3.18...3.31)
project(verifloat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# No C++ modules here: without this, CMake 3.28+ scans every source for them,
# which with Clang needs clang-scan-deps to be installed.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Boost.Multiprecision (header-only) backs arbitrary-width formats. Use the
# system headers when present, otherwise fetch the standalone library.
find_package(Boost 1.79 CONFIG QUIET COMPONENTS headers)
if(NOT Boost_FOUND)
  include(FetchContent)
  # Only the headers are used: the SOURCE_SUBDIR that does not exist keeps
  # FetchContent from adding the library's own CMake project.
  FetchContent_Declare(multiprecision
    GIT_REPOSITORY https://github.com/boostorg/multiprecision.git
    GIT_TAG Boost_1_86_0 GIT_SHALLOW TRUE
    SOURCE_SUBDIR _headers_only)
  FetchContent_MakeAvailable(multiprecision)
  set(VF_MP_INCLUDE "${multiprecision_SOURCE_DIR}/include")
  set(VF_MP_DEFS BOOST_MP_STANDALONE)
endif()

# The core: the operations on plain encodings, in C++ with no Python in it.
# It is compiled without the Python headers on its include path, so it
# cannot come to depend on them.
add_library(vf_core OBJECT src/cpp/ops.cpp)
set_target_properties(vf_core PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
target_include_directories(vf_core PRIVATE src/cpp ${VF_MP_INCLUDE})
if(Boost_FOUND)
  target_link_libraries(vf_core PRIVATE Boost::headers)
else()
  target_compile_definitions(vf_core PRIVATE ${VF_MP_DEFS})
endif()
if(NOT MSVC)
  target_compile_options(vf_core PRIVATE -O3 -Wall -Wextra -fno-math-errno)
endif()

nanobind_add_module(_core NOMINSIZE LTO
  $<TARGET_OBJECTS:vf_core>
  src/cpp/module.cpp
  src/cpp/fp.cpp
  src/cpp/uint.cpp
  src/cpp/accum.cpp
  src/cpp/fixed.cpp
  src/cpp/block.cpp
  src/cpp/array.cpp
  src/cpp/selfcheck.cpp
  src/cpp/simd_avx2.cpp
  src/cpp/simd_avx512.cpp
  src/cpp/pyconv.cpp)
target_include_directories(_core PRIVATE src/cpp ${VF_MP_INCLUDE})
if(Boost_FOUND)
  target_link_libraries(_core PRIVATE Boost::headers)
else()
  target_compile_definitions(_core PRIVATE ${VF_MP_DEFS})
endif()
if(NOT MSVC)
  target_compile_options(_core PRIVATE -O3 -Wall -Wextra -fno-math-errno)
endif()

# SIMD kernels: built for x86-64 with their own instruction-set flags and
# chosen at run time from what the CPU supports (src/cpp/simd_api.hpp).
# Elsewhere the two files compile to stubs and the scalar kernels are used.
if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set_source_files_properties(src/cpp/simd_avx2.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx2" COMPILE_DEFINITIONS "VF_SIMD_AVX2")
  set_source_files_properties(src/cpp/simd_avx512.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx512f;-mavx512cd;-mavx512dq" COMPILE_DEFINITIONS "VF_SIMD_AVX512")
endif()

install(TARGETS _core LIBRARY DESTINATION verifloat)

# The C library: the same core behind a plain C API (src/cpp/include/verifloat.h),
# for SystemVerilog DPI-C (src/sv/verifloat_pkg.sv) and C or C++ callers.
# It is installed inside the Python package; `python -m verifloat.dpi` prints
# where.
add_library(verifloat_c SHARED src/cpp/capi.cpp $<TARGET_OBJECTS:vf_core>)
set_target_properties(verifloat_c PROPERTIES OUTPUT_NAME verifloat CXX_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN ON)
target_include_directories(verifloat_c PRIVATE src/cpp src/cpp/include ${VF_MP_INCLUDE})
target_compile_definitions(verifloat_c PRIVATE VF_VERSION="${SKBUILD_PROJECT_VERSION}")
if(Boost_FOUND)
  target_link_libraries(verifloat_c PRIVATE Boost::headers)
else()
  target_compile_definitions(verifloat_c PRIVATE ${VF_MP_DEFS})
endif()
if(NOT MSVC)
  target_compile_options(verifloat_c PRIVATE -O3 -Wall -Wextra -fno-math-errno)
endif()
install(TARGETS verifloat_c LIBRARY DESTINATION verifloat RUNTIME DESTINATION verifloat)
install(FILES src/cpp/include/verifloat.h DESTINATION verifloat/include)
install(FILES src/sv/verifloat_pkg.sv DESTINATION verifloat/sv)
