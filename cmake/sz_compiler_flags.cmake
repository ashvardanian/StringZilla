# cmake/sz_compiler_flags.cmake — per-target compiler-flag helpers shared by every StringZilla target: warnings,
# optimization, standards, architecture baselines, and the per-capability `STRINGZILLA_TARGET_*` stamps. Included after the option
# block: the helpers read `STRINGZILLA_USE_SANITIZERS`, `STRINGZILLA_BUILD_COVERAGE`, and the `STRINGZILLA_ARCH_*_` platform
# facts at call time.

# Maximum warnings level & warnings as error. MSVC uses numeric values: > 4068 for "unknown pragmas",
# > 4146 for "unary minus operator applied to unsigned type"; `/utf-8` keeps UTF-8 symbols in tests intact.
function (set_warning_flags target compiler_id)
    if (compiler_id STREQUAL "GNU")
        # `-Wno-error=array-bounds`: GCC 12+ false-positives on StringZilla's intentional wide reads (u32/u64/ vector
        # loads near a buffer end) when an ISA kernel is inlined into a string-literal-sized caller. Keep it a visible
        # warning, not a build-breaking error.
        target_compile_options(
            ${target}
            PRIVATE
                "-Wall;-Wextra;-Werror;-Wfatal-errors;-Wno-unknown-pragmas;-Wno-cast-function-type;-Wno-unused-function;-Wno-sign-conversion;-Wno-error=array-bounds"
        )
    elseif (compiler_id STREQUAL "Clang" OR compiler_id STREQUAL "AppleClang")
        target_compile_options(
            ${target} PRIVATE "-Wall;-Wextra;-Werror;-Wfatal-errors;-Wno-unknown-pragmas;-Wno-sign-conversion"
        )
    elseif (compiler_id MATCHES "MSVC")
        target_compile_options(
            ${target}
            PRIVATE "/Bt" # Display build timings
                    "/wd4068" # Disable warning: unknown pragma
                    "/wd5030" # Disable warning: attribute is not recognized
                    "/wd5051" # Disable warning: attribute requires a newer standard (e.g. [[maybe_unused]] in C++11/14)
                    "/wd4146" # Disable warning: unary minus operator applied to unsigned type
                    "/wd4996" # Disable warning: 'unsafe' functions like getenv, fopen (use _s variants)
                    "/wd4244" # Disable warning: conversion with possible loss of data (e.g., float to int)
                    "/wd4267" # Disable warning: conversion from 'size_t' to smaller type, possible loss of data
                    "/utf-8" # Set source and execution character sets to UTF-8
                    "/WX" # Treat warnings as errors
        )
    elseif (compiler_id STREQUAL "NVIDIA")
        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(
                ${target}
                PRIVATE
                    "-Xcompiler=/Zc:preprocessor;-Xcompiler=/Zc:__cplusplus;-Xcompiler=/W3;-Xcompiler=/WX;-Xcompiler=/wd4068;-Xcompiler=/wd5030;-Xcompiler=/wd5051;-Xcompiler=/wd4146;-Xcompiler=/wd4996;-Xcompiler=/wd4244;-Xcompiler=/wd4267;-Xcompiler=/utf-8"
            )
        else ()
            target_compile_options(
                ${target}
                PRIVATE
                    "-Xcompiler=-Wfatal-errors;-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-Wno-error=array-bounds;-Wno-unknown-pragmas;-Wno-cast-function-type;-Wno-unused-function"
            )
        endif ()
    endif ()
endfunction ()

# Optimization, debug-info, and runtime-check flags. Everything keys on `$<CONFIG:...>` generator
# expressions rather than `CMAKE_BUILD_TYPE`, which is empty under multi-config generators like Visual
# Studio - string comparisons there would silently strip every optimization and debug flag.
function (set_optimization_flags target compiler_id target_type)
    if (compiler_id MATCHES "MSVC")
        target_compile_options(${target} PRIVATE "$<$<CONFIG:Debug>:/Od;/Zi>")
        if (NOT target_type STREQUAL "SHARED_LIBRARY")
            target_compile_options(${target} PRIVATE "$<$<CONFIG:Debug>:/RTC1>")
        endif ()
        target_compile_options(${target} PRIVATE "$<$<CONFIG:Release,RelWithDebInfo>:/O2;/Zi>")
    elseif (
        compiler_id STREQUAL "GNU"
        OR compiler_id STREQUAL "Clang"
        OR compiler_id STREQUAL "AppleClang"
    )
        target_compile_options(${target} PRIVATE "$<$<CONFIG:Debug>:-O0;-g>")
        target_compile_options(${target} PRIVATE "$<$<CONFIG:RelWithDebInfo>:-O2;-g>")
        target_compile_options(${target} PRIVATE "$<$<CONFIG:Release>:-O2>")
    elseif (compiler_id STREQUAL "NVIDIA")
        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            set(sz_nvcc_debug_
                "-G" # Device debug symbols
                "-no-compress" # No compression of debug info
                "-Xcompiler=/Zi" # Host debugging symbols
                "-Xcompiler=/Oy-" # Frame pointers for stack traces
                "-Xcompiler=/Ob0" # Prevent host inlining
                "-maxrregcount=0" # No register count limits
            )
            set(sz_nvcc_lineinfo_
                "-lineinfo" # Source correlation through optimized device code
                "-Xcompiler=/Zi" # Host debugging symbols
                "-Xcompiler=/Oy-" # Frame pointers for stack traces
            )
            set(sz_nvcc_release_
                "-O2" # NVCC optimizations
                "-Xptxas=-O2" # PTX assembler optimizations
                "-Xcompiler=/O2" # Host optimizations
            )
        else ()
            set(sz_nvcc_debug_
                "-G" # Device debug symbols
                "-no-compress" # No compression of debug info
                "-Xcompiler=-g" # Host debugging symbols explicitly
                "-Xcompiler=-fno-omit-frame-pointer" # Stack trace clarity
                "-Xcompiler=-fno-inline" # Prevent host inlining
                "-maxrregcount=0" # No register count limits
            )
            set(sz_nvcc_lineinfo_
                "-lineinfo" # Source correlation through optimized device code
                "-Xcompiler=-g" # Host debugging symbols explicitly
                "-Xcompiler=-fno-omit-frame-pointer" # Stack trace clarity
            )
            set(sz_nvcc_release_
                "-O2" # NVCC optimizations
                "-Xptxas=-O2" # PTX assembler optimizations
                "-Xcompiler=-O2" # Host optimizations
            )
        endif ()
        # `RelWithDebInfo` matched both lists and asked `ptxas` for `-G` beside `-O2`, which it refuses.
        # `-G` stays with `Debug`; `-lineinfo` carries the same source correlation through optimized code.
        target_compile_options(
            ${target} PRIVATE "$<$<CONFIG:Debug>:${sz_nvcc_debug_}>" "$<$<CONFIG:RelWithDebInfo>:${sz_nvcc_lineinfo_}>"
                              "$<$<CONFIG:Release,RelWithDebInfo>:${sz_nvcc_release_}>"
        )
    endif ()
endfunction ()

# Function to set the default compiler-specific flags
function (set_compiler_flags target cpp_standard target_arch compiler_id)
    get_target_property(target_type ${target} TYPE)

    target_include_directories(${target} PRIVATE test bench)

    # Set output directory for single-configuration generators (like Make)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<0:>)
    set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<0:>)

    # Set output directory for multi-configuration generators (like Visual Studio)
    foreach (config IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER ${config} config_upper)
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_BINARY_DIR}/$<0:>)
        set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_BINARY_DIR}/$<0:>)
    endforeach ()

    # Set the C++ standard
    if (NOT cpp_standard STREQUAL "")
        if (compiler_id STREQUAL "NVIDIA")
            set_target_properties(${target} PROPERTIES CUDA_STANDARD ${cpp_standard})
        elseif (compiler_id MATCHES "MSVC")
            # For MSVC, explicitly set the /std: flag - don't set CXX_STANDARD property to avoid conflicts. MSVC has no
            # `/std:c++23`; its newest standard is exposed as `/std:c++latest`, so map 23+ onto it.
            if (cpp_standard GREATER_EQUAL 23)
                target_compile_options(${target} PRIVATE "/std:c++latest")
            else ()
                target_compile_options(${target} PRIVATE "/std:c++${cpp_standard}")
            endif ()
        else ()
            set_target_properties(${target} PROPERTIES CXX_STANDARD ${cpp_standard})
        endif ()
    endif ()

    # Use the `/Zc:__cplusplus` flag to correctly define the `__cplusplus` macro in MSVC
    if (compiler_id MATCHES "MSVC")
        target_compile_options(${target} PRIVATE "/Zc:__cplusplus")
    endif ()

    # Make sure CUDA C++ allows calling `constexpr` from device code
    if (compiler_id STREQUAL "NVIDIA")
        target_compile_options(${target} PRIVATE "--expt-relaxed-constexpr")
    endif ()

    set_warning_flags(${target} "${compiler_id}")
    set_optimization_flags(${target} "${compiler_id}" "${target_type}")

    # Avoid builtin functions where we know what we are doing.
    if (compiler_id MATCHES "MSVC")
        target_compile_options(${target} PRIVATE "/Oi-")
    elseif (compiler_id STREQUAL "NVIDIA")
        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(${target} PRIVATE "-Xcompiler=/Oi-")
        else ()
            target_compile_options(
                ${target} PRIVATE "-Xcompiler=-fno-builtin-memcmp" "-Xcompiler=-fno-builtin-memchr"
                                  "-Xcompiler=-fno-builtin-memcpy" "-Xcompiler=-fno-builtin-memset"
            )
        endif ()
    else ()
        target_compile_options(${target} PRIVATE "-fno-builtin-memcmp")
        target_compile_options(${target} PRIVATE "-fno-builtin-memchr")
        target_compile_options(${target} PRIVATE "-fno-builtin-memcpy")
        target_compile_options(${target} PRIVATE "-fno-builtin-memset")
    endif ()

    # On macOS, when using non-AppleClang compilers (e.g., Homebrew LLVM), explicitly link against libc++. AppleClang
    # automatically links the system libc++, but Homebrew LLVM requires explicit configuration.
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin"
        AND compiler_id STREQUAL "Clang"
        AND NOT compiler_id STREQUAL "AppleClang"
    )
        if (NOT target_type STREQUAL "SHARED_LIBRARY")
            target_compile_options(${target} PRIVATE "-stdlib=libc++")
            target_link_options(${target} PRIVATE "-stdlib=libc++")
            # Find and link the C++ standard library from the compiler's installation Homebrew LLVM stores libc++ in
            # lib/c++ subdirectory
            get_filename_component(sz_compiler_dir_ ${CMAKE_CXX_COMPILER} DIRECTORY)
            get_filename_component(sz_compiler_root_ ${sz_compiler_dir_} DIRECTORY)
            if (EXISTS "${sz_compiler_root_}/lib/c++/libc++.dylib")
                target_link_options(${target} PRIVATE "-L${sz_compiler_root_}/lib/c++")
                target_link_libraries(${target} PRIVATE c++abi)
            elseif (EXISTS "${sz_compiler_root_}/lib/libc++.dylib")
                target_link_options(${target} PRIVATE "-L${sz_compiler_root_}/lib")
            endif ()
        endif ()
    endif ()

    # Check for ${target_arch} and set it or use the current system if not defined
    if ("${target_arch}" STREQUAL "")
        # Only use the current system if we are not cross compiling
        if (((NOT MSVC) AND (NOT CMAKE_CROSSCOMPILING)) OR (CMAKE_SYSTEM_PROCESSOR MATCHES
                                                            ${CMAKE_HOST_SYSTEM_PROCESSOR})
        )
            if (compiler_id STREQUAL "NVIDIA")
                # For NVCC, pass architecture flag to host compiler
                if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
                    if (STRINGZILLA_ARCH_ARM64_)
                        target_compile_options(${target} PRIVATE "-Xcompiler=/arch:armv8.0")
                    else ()
                        target_compile_options(${target} PRIVATE "-Xcompiler=/arch:AVX2")
                    endif ()
                elseif (STRINGZILLA_CUDA_ACCEPTS_NATIVE_ARCH)
                    target_compile_options(${target} PRIVATE "-Xcompiler=-march=native")
                endif ()
            elseif (NOT (compiler_id MATCHES "MSVC"))
                check_cxx_compiler_flag("-march=native" supports_march_native)
                if (supports_march_native)
                    target_compile_options(${target} PRIVATE "-march=native")
                endif ()
            else ()
                # MSVC does not have a direct equivalent to -march=native
                if (STRINGZILLA_ARCH_ARM64_)
                    target_compile_options(${target} PRIVATE "/arch:armv8.0")
                else ()
                    target_compile_options(${target} PRIVATE "/arch:AVX2")
                endif ()
            endif ()
        endif ()
    else ()
        if (compiler_id MATCHES "MSVC")
            target_compile_options(${target} PRIVATE "/arch:${target_arch}")
        elseif (compiler_id STREQUAL "NVIDIA")
            # NVCC handles CPU architecture through host compiler flags
            if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
                target_compile_options(${target} PRIVATE "-Xcompiler=/arch:${target_arch}")
            else ()
                target_compile_options(${target} PRIVATE "-Xcompiler=-march=${target_arch}")
            endif ()
        else ()
            target_compile_options(${target} PRIVATE "-march=${target_arch}")
        endif ()
    endif ()

    # Define STRINGZILLA_ARCH_BIG_ENDIAN_ macro based on system byte order
    if (CMAKE_C_BYTE_ORDER STREQUAL "BIG_ENDIAN")
        set(STRINGZILLA_ARCH_BIG_ENDIAN_ 1)
    else ()
        set(STRINGZILLA_ARCH_BIG_ENDIAN_ 0)
    endif ()

    target_compile_definitions(${target} PRIVATE "STRINGZILLA_ARCH_BIG_ENDIAN_=${STRINGZILLA_ARCH_BIG_ENDIAN_}")

    # Sanitizer options for Debug mode
    target_compile_definitions(${target} PRIVATE "$<IF:$<CONFIG:Debug>,STRINGZILLA_DEBUG=1,STRINGZILLA_DEBUG=0>")
    if (STRINGZILLA_USE_SANITIZERS AND NOT target_type STREQUAL "SHARED_LIBRARY")
        if (compiler_id MATCHES "MSVC")
            target_compile_options(${target} PRIVATE "$<$<CONFIG:Debug>:/fsanitize=address;/fsanitize=leak>")
            target_link_options(${target} PRIVATE "$<$<CONFIG:Debug>:/fsanitize=address;/fsanitize=leak>")
        elseif (compiler_id STREQUAL "NVIDIA")
            # ! NVCC can't handle sanitizers?!
            # https://stackoverflow.com/questions/75590579/cuda-fails-to-initialise-when-address-sanitizer-is-enabled
        else ()
            target_compile_options(${target} PRIVATE "$<$<CONFIG:Debug>:-fsanitize=address;-fsanitize=undefined>")
            target_link_options(${target} PRIVATE "$<$<CONFIG:Debug>:-fsanitize=address;-fsanitize=undefined>")
        endif ()
    endif ()

    if (STRINGZILLA_BUILD_COVERAGE AND (compiler_id STREQUAL "Clang" OR compiler_id STREQUAL "AppleClang"))
        target_compile_options(${target} PRIVATE "-fprofile-instr-generate" "-fcoverage-mapping")
        target_link_options(${target} PRIVATE "-fprofile-instr-generate" "-fcoverage-mapping")
    endif ()
endfunction ()

# Stamps the architecture id and the `STRINGZILLA_TARGET_*` verdicts onto a target: `COMPILE` for runtime-dispatched libraries,
# `RUN` for comptime-dispatched executables whose picked tier must also run on this machine.
function (set_architecture_simd_definitions target mode)
    if (STRINGZILLA_ARCH_X86_64_)
        target_compile_definitions(${target} PRIVATE "STRINGZILLA_ARCH_X86_64_=1" "STRINGZILLA_ARCH_ARM64_=0")
    elseif (STRINGZILLA_ARCH_ARM64_)
        target_compile_definitions(${target} PRIVATE "STRINGZILLA_ARCH_X86_64_=0" "STRINGZILLA_ARCH_ARM64_=1")
    else ()
        target_compile_definitions(${target} PRIVATE "STRINGZILLA_ARCH_X86_64_=0" "STRINGZILLA_ARCH_ARM64_=0")
    endif ()
    if (mode STREQUAL "COMPILE")
        target_compile_definitions(${target} PRIVATE ${sz_compile_definitions_})
    elseif (mode STREQUAL "RUN")
        target_compile_definitions(${target} PRIVATE ${sz_run_definitions_})
    else ()
        message(FATAL_ERROR "set_architecture_simd_definitions: mode must be COMPILE or RUN, got `${mode}`")
    endif ()
endfunction ()

# Apply the conservative baseline architecture (`-march`/`-mcpu`) that lets one shared/OBJECT compilation host every
# per-ISA SIMD capability: the per-function target attributes inside the kernels pick the actual instruction set, so
# the baseline only has to be old enough for every capability's intrinsics headers to parse. Cross-compiled targets (RISC-V,
# LoongArch, POWER) get their arch from the toolchain file's `CMAKE_<LANG>_FLAGS_INIT`, so they pass an empty arch.
function (set_baseline_architecture_flags target compiler_id)
    if (STRINGZILLA_ARCH_X86_64_)
        if (MSVC)
            set_compiler_flags(${target} "" "SSE2" "${compiler_id}")
        else ()
            set_compiler_flags(${target} "" "ivybridge" "${compiler_id}")
        endif ()
    elseif (STRINGZILLA_ARCH_ARM64_)
        if (MSVC)
            set_compiler_flags(${target} "" "armv8.0" "${compiler_id}")
        else ()
            set_compiler_flags(${target} "" "armv8-a" "${compiler_id}")
        endif ()
    else ()
        set_compiler_flags(${target} "" "" "${compiler_id}")
    endif ()
endfunction ()
