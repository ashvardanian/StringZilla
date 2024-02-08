function(sz_test target)
  add_executable("${target}" ${ARGN})
  target_link_libraries("${target}" PRIVATE stringzilla::stringzilla)
  add_test(NAME "${target}" COMMAND "${target}")
endfunction()

function(sz_simd_test arch target)
  sz_test("${target}" ${ARGN})
  target_compile_options("${target}" PRIVATE "-march=${arch}")
  set_target_properties(
    "${target}" PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED 1)
  set_property(TEST "${target}" PROPERTY LABELS simd std20)
endfunction()

include(CTest)
if(BUILD_TESTING)
  foreach(std 11 14 17 20)
    sz_test("stringzilla_test_cpp${std}" scripts/test.cpp)
    set_target_properties(
      "stringzilla_test_cpp${std}" PROPERTIES
      CXX_STANDARD "${std}"
      CXX_STANDARD_REQUIRED 1)
    set_property(TEST "stringzilla_test_cpp${std}" PROPERTY LABELS normal "std${std}")
  endforeach()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$|^GNU\$")
    # Check system architecture to avoid complex cross-compilation workflows, but
    # compile multiple backends: disabling all SIMD, enabling only AVX2, only AVX-512, only Arm Neon.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
      # x86 specific backends
      sz_simd_test(ivybridge stringzilla_test_cpp20_x86_serial scripts/test.cpp)
      sz_simd_test(haswell stringzilla_test_cpp20_x86_avx2 scripts/test.cpp)
      sz_simd_test(sapphirerapids stringzilla_test_cpp20_x86_avx512 scripts/test.cpp)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64|AARCH64")
      # ARM specific backends
      sz_simd_test(armv8-a stringzilla_test_cpp20_arm_serial scripts/test.cpp)
      sz_simd_test(armv8-a+simd stringzilla_test_cpp20_arm_neon scripts/test.cpp)
    endif()
  endif()
endif()

option(BUILD_BENCHMARK "Compile a native benchmark in C++" ON)
if(BUILD_BENCHMARK)
  foreach(type search similarity sort token container)
    sz_test("stringzilla_bench_${type}" "scripts/bench_${type}.cpp")
    set_target_properties(
      "stringzilla_bench_${type}" PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED 1)
    set_property(TEST "stringzilla_bench_${type}" PROPERTY LABELS bench std20)
    target_include_directories("stringzilla_bench_${type}" PRIVATE scripts)
  endforeach()
endif()
