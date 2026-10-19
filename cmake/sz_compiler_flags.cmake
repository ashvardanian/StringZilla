# cmake/sz_compiler_flags.cmake — per-target compiler-flag helpers shared by every StringZilla target: warnings,
# optimization, standards, architecture baselines, and the per-capability `STRINGZILLA_TARGET_*` stamps. Included after
# the option block: the helpers read `STRINGZILLA_BUILD_SANITIZERS`, `STRINGZILLA_BUILD_COVERAGE`, and the
# `stringzilla_arch_*_` platform facts at call time.

# Flags follow each source's language and compiler, as one target may mix C, C++, CUDA and HIP. `COMPILE_LANG_AND_ID`
# takes one language, hence the `OR`s; "GCC-style" means every compiler but MSVC and NVCC.
set(stringzilla_gnu_ "$<OR:$<COMPILE_LANG_AND_ID:C,GNU>,$<COMPILE_LANG_AND_ID:CXX,GNU>>")
set(stringzilla_clang_
    "$<OR:$<COMPILE_LANG_AND_ID:C,Clang,AppleClang>,$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>,$<COMPILE_LANG_AND_ID:HIP,Clang>>"
)
set(stringzilla_msvc_ "$<OR:$<COMPILE_LANG_AND_ID:C,MSVC>,$<COMPILE_LANG_AND_ID:CXX,MSVC>>")
set(stringzilla_nvcc_ "$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>")
set(stringzilla_gcc_style_ "$<NOT:$<OR:${stringzilla_msvc_},${stringzilla_nvcc_}>>")
set(stringzilla_clang_link_
    "$<OR:$<LINK_LANG_AND_ID:C,Clang,AppleClang>,$<LINK_LANG_AND_ID:CXX,Clang,AppleClang>,$<LINK_LANG_AND_ID:HIP,Clang>>"
)
set(stringzilla_msvc_link_ "$<OR:$<LINK_LANG_AND_ID:C,MSVC>,$<LINK_LANG_AND_ID:CXX,MSVC>>")
set(stringzilla_gcc_style_link_ "$<NOT:$<OR:${stringzilla_msvc_link_},$<LINK_LANG_AND_ID:CUDA,NVIDIA>>>")

# Maximum warnings level & warnings as error. MSVC uses numeric values: > 4068 for "unknown pragmas",
# > 4146 for "unary minus operator applied to unsigned type"; `/utf-8` keeps UTF-8 symbols in tests intact.
function (stringzilla_set_warning_flags target_)
    # `-Wno-error=array-bounds`: GCC 12+ false-positives on StringZilla's intentional wide reads (u32/u64/ vector
    # loads near a buffer end) when an ISA kernel is inlined into a string-literal-sized caller. Keep it a visible
    # warning, not a build-breaking error.
    set(gnu_warnings_
        "-Wall;-Wextra;-Werror;-Wfatal-errors;-Wno-unknown-pragmas;-Wno-cast-function-type;-Wno-unused-function;-Wno-sign-conversion;-Wno-error=array-bounds"
    )
    set(clang_warnings_ "-Wall;-Wextra;-Werror;-Wfatal-errors;-Wno-unknown-pragmas;-Wno-sign-conversion")
    set(msvc_warnings_
        "/Bt" # Display build timings
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
    # NVCC forwards the host flags to the C++ compiler it drives.
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(nvcc_warnings_
            "-Xcompiler=/Zc:preprocessor;-Xcompiler=/Zc:__cplusplus;-Xcompiler=/W3;-Xcompiler=/WX;-Xcompiler=/wd4068;-Xcompiler=/wd5030;-Xcompiler=/wd5051;-Xcompiler=/wd4146;-Xcompiler=/wd4996;-Xcompiler=/wd4244;-Xcompiler=/wd4267;-Xcompiler=/utf-8"
        )
    else ()
        set(nvcc_warnings_
            "-Xcompiler=-Wfatal-errors;-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-Wno-error=array-bounds;-Wno-unknown-pragmas;-Wno-cast-function-type;-Wno-unused-function"
        )
    endif ()
    target_compile_options(
        ${target_} PRIVATE "$<${stringzilla_gnu_}:${gnu_warnings_}>" "$<${stringzilla_clang_}:${clang_warnings_}>"
                           "$<${stringzilla_msvc_}:${msvc_warnings_}>" "$<${stringzilla_nvcc_}:${nvcc_warnings_}>"
    )
endfunction ()

# Optimization, debug-info, and runtime-check flags. Everything keys on `$<CONFIG:...>` generator
# expressions rather than `CMAKE_BUILD_TYPE`, which is empty under multi-config generators like Visual
# Studio - string comparisons there would silently strip every optimization and debug flag.
function (stringzilla_set_optimization_flags target_ target_type_)
    target_compile_options(${target_} PRIVATE "$<$<AND:${stringzilla_msvc_},$<CONFIG:Debug>>:/Od;/Zi>")
    if (NOT target_type_ STREQUAL "SHARED_LIBRARY")
        target_compile_options(${target_} PRIVATE "$<$<AND:${stringzilla_msvc_},$<CONFIG:Debug>>:/RTC1>")
    endif ()
    target_compile_options(${target_} PRIVATE "$<$<AND:${stringzilla_msvc_},$<CONFIG:Release,RelWithDebInfo>>:/O2;/Zi>")

    set(gcc_ "$<OR:${stringzilla_gnu_},${stringzilla_clang_}>")
    target_compile_options(${target_} PRIVATE "$<$<AND:${gcc_},$<CONFIG:Debug>>:-O0;-g>")
    target_compile_options(${target_} PRIVATE "$<$<AND:${gcc_},$<CONFIG:RelWithDebInfo>>:-O2;-g>")
    target_compile_options(${target_} PRIVATE "$<$<AND:${gcc_},$<CONFIG:Release>>:-O2>")

    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(nvcc_debug_
            "-G" # Device debug symbols
            "-no-compress" # No compression of debug info
            "-Xcompiler=/Zi" # Host debugging symbols
            "-Xcompiler=/Oy-" # Frame pointers for stack traces
            "-Xcompiler=/Ob0" # Prevent host inlining
            "-maxrregcount=0" # No register count limits
        )
        set(nvcc_lineinfo_
            "-lineinfo" # Source correlation through optimized device code
            "-Xcompiler=/Zi" # Host debugging symbols
            "-Xcompiler=/Oy-" # Frame pointers for stack traces
        )
        set(nvcc_release_
            "-O2" # NVCC optimizations
            "-Xptxas=-O2" # PTX assembler optimizations
            "-Xcompiler=/O2" # Host optimizations
        )
    else ()
        set(nvcc_debug_
            "-G" # Device debug symbols
            "-no-compress" # No compression of debug info
            "-Xcompiler=-g" # Host debugging symbols explicitly
            "-Xcompiler=-fno-omit-frame-pointer" # Stack trace clarity
            "-Xcompiler=-fno-inline" # Prevent host inlining
            "-maxrregcount=0" # No register count limits
        )
        set(nvcc_lineinfo_
            "-lineinfo" # Source correlation through optimized device code
            "-Xcompiler=-g" # Host debugging symbols explicitly
            "-Xcompiler=-fno-omit-frame-pointer" # Stack trace clarity
        )
        set(nvcc_release_
            "-O2" # NVCC optimizations
            "-Xptxas=-O2" # PTX assembler optimizations
            "-Xcompiler=-O2" # Host optimizations
        )
    endif ()
    # `RelWithDebInfo` matched both lists and asked `ptxas` for `-G` beside `-O2`, which it refuses.
    # `-G` stays with `Debug`; `-lineinfo` carries the same source correlation through optimized code.
    target_compile_options(
        ${target_}
        PRIVATE "$<$<AND:${stringzilla_nvcc_},$<CONFIG:Debug>>:${nvcc_debug_}>"
                "$<$<AND:${stringzilla_nvcc_},$<CONFIG:RelWithDebInfo>>:${nvcc_lineinfo_}>"
                "$<$<AND:${stringzilla_nvcc_},$<CONFIG:Release,RelWithDebInfo>>:${nvcc_release_}>"
    )
endfunction ()

# Function to set the default compiler-specific flags
function (stringzilla_set_compiler_flags target_ cpp_standard_ target_arch_)
    get_target_property(target_type_ ${target_} TYPE)

    # Set output directory for single-configuration generators (like Make)
    set_target_properties(${target_} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<0:>)
    set_target_properties(${target_} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<0:>)

    # Set output directory for multi-configuration generators (like Visual Studio)
    foreach (configuration_ IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER ${configuration_} configuration_uppercase_)
        set_target_properties(
            ${target_} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${configuration_uppercase_} ${CMAKE_BINARY_DIR}/$<0:>
        )
        set_target_properties(
            ${target_} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_${configuration_uppercase_} ${CMAKE_BINARY_DIR}/$<0:>
        )
    endforeach ()

    # Set the C++ standard
    if (NOT cpp_standard_ STREQUAL "")
        set_target_properties(${target_} PROPERTIES CUDA_STANDARD ${cpp_standard_})
        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            # For MSVC, explicitly set the /std: flag - don't set CXX_STANDARD property to avoid conflicts. MSVC has no
            # `/std:c++23`; its newest standard is exposed as `/std:c++latest`, so map 23+ onto it.
            if (cpp_standard_ GREATER_EQUAL 23)
                target_compile_options(${target_} PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/std:c++latest>")
            else ()
                target_compile_options(${target_} PRIVATE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/std:c++${cpp_standard_}>")
            endif ()
        else ()
            set_target_properties(${target_} PROPERTIES CXX_STANDARD ${cpp_standard_})
        endif ()
    endif ()

    # Report the selected C++ standard and preprocess raw strings in test assertions correctly on MSVC.
    target_compile_options(${target_} PRIVATE "$<${stringzilla_msvc_}:/Zc:__cplusplus;/Zc:preprocessor>")

    # Make sure CUDA C++ allows calling `constexpr` from device code
    target_compile_options(${target_} PRIVATE "$<${stringzilla_nvcc_}:--expt-relaxed-constexpr>")

    stringzilla_set_warning_flags(${target_})
    stringzilla_set_optimization_flags(${target_} "${target_type_}")

    # Avoid builtin functions where we know what we are doing.
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(nvcc_builtins_ "-Xcompiler=/Oi-")
    else ()
        set(nvcc_builtins_ "-Xcompiler=-fno-builtin-memcmp" "-Xcompiler=-fno-builtin-memchr"
                           "-Xcompiler=-fno-builtin-memcpy" "-Xcompiler=-fno-builtin-memset"
        )
    endif ()
    target_compile_options(
        ${target_}
        PRIVATE
            "$<${stringzilla_msvc_}:/Oi->"
            "$<${stringzilla_nvcc_}:${nvcc_builtins_}>"
            "$<${stringzilla_gcc_style_}:-fno-builtin-memcmp;-fno-builtin-memchr;-fno-builtin-memcpy;-fno-builtin-memset>"
    )

    # On macOS, when using non-AppleClang compilers (e.g., Homebrew LLVM), explicitly link against libc++. AppleClang
    # automatically links the system libc++, but Homebrew LLVM requires explicit configuration. Only the C++ executables
    # link it: the libraries are C, and an archive would hand `c++abi` to its consumers without the path to find it.
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin"
        AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
        AND target_type_ STREQUAL "EXECUTABLE"
    )
        target_compile_options(${target_} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>")
        target_link_options(${target_} PRIVATE "$<$<LINK_LANGUAGE:CXX>:-stdlib=libc++>")
        # Find and link the C++ standard library from the compiler's installation Homebrew LLVM stores libc++ in
        # lib/c++ subdirectory
        get_filename_component(compiler_directory_ ${CMAKE_CXX_COMPILER} DIRECTORY)
        get_filename_component(compiler_root_ ${compiler_directory_} DIRECTORY)
        if (EXISTS "${compiler_root_}/lib/c++/libc++.dylib")
            target_link_options(${target_} PRIVATE "$<$<LINK_LANGUAGE:CXX>:-L${compiler_root_}/lib/c++>")
            target_link_libraries(${target_} PRIVATE "$<$<LINK_LANGUAGE:CXX>:c++abi>")
        elseif (EXISTS "${compiler_root_}/lib/libc++.dylib")
            target_link_options(${target_} PRIVATE "$<$<LINK_LANGUAGE:CXX>:-L${compiler_root_}/lib>")
        endif ()
    endif ()

    # Check for ${target_arch_} and set it, or tune an executable for the current system if not defined. A library
    # without one keeps the toolchain's default, as it may ship to any CPU of its architecture. NVCC forwards it to its
    # host.
    if ("${target_arch_}" STREQUAL "")
        # Only use the current system if we are not cross compiling
        if (target_type_ STREQUAL "EXECUTABLE" AND (((NOT MSVC) AND (NOT CMAKE_CROSSCOMPILING))
                                                    OR (CMAKE_SYSTEM_PROCESSOR MATCHES ${CMAKE_HOST_SYSTEM_PROCESSOR}))
        )
            if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
                # MSVC does not have a direct equivalent to -march=native
                if (stringzilla_arch_arm64_)
                    set(native_arch_ "/arch:armv8.0")
                else ()
                    set(native_arch_ "/arch:AVX2")
                endif ()
                target_compile_options(
                    ${target_} PRIVATE "$<${stringzilla_msvc_}:${native_arch_}>"
                                       "$<${stringzilla_nvcc_}:-Xcompiler=${native_arch_}>"
                )
            else ()
                check_cxx_compiler_flag("-march=native" stringzilla_supports_march_native_)
                if (stringzilla_supports_march_native_)
                    target_compile_options(${target_} PRIVATE "$<${stringzilla_gcc_style_}:-march=native>")
                endif ()
                if (stringzilla_cuda_native_arch_compiles_)
                    target_compile_options(${target_} PRIVATE "$<${stringzilla_nvcc_}:-Xcompiler=-march=native>")
                endif ()
            endif ()
        endif ()
    elseif (NOT stringzilla_arch_wasm_) # There it names the module's SIMD capability, which a directory-wide flag sets
        if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            set(host_arch_ "/arch:${target_arch_}")
        else ()
            set(host_arch_ "-march=${target_arch_}")
        endif ()
        target_compile_options(
            ${target_}
            PRIVATE "$<${stringzilla_msvc_}:/arch:${target_arch_}>" "$<${stringzilla_nvcc_}:-Xcompiler=${host_arch_}>"
                    "$<${stringzilla_gcc_style_}:-march=${target_arch_}>"
        )
    endif ()

    # Define STRINGZILLA_ARCH_BIG_ENDIAN_ macro based on system byte order
    target_compile_definitions(
        ${target_} PRIVATE "STRINGZILLA_ARCH_BIG_ENDIAN_=$<STREQUAL:${CMAKE_C_BYTE_ORDER},BIG_ENDIAN>"
    )

    # Sanitizer options for Debug mode; NVCC can't handle them:
    # https://stackoverflow.com/questions/75590579/cuda-fails-to-initialise-when-address-sanitizer-is-enabled
    target_compile_definitions(${target_} PRIVATE "$<IF:$<CONFIG:Debug>,STRINGZILLA_DEBUG=1,STRINGZILLA_DEBUG=0>")
    if (STRINGZILLA_BUILD_SANITIZERS AND NOT target_type_ STREQUAL "SHARED_LIBRARY")
        target_compile_options(
            ${target_}
            PRIVATE "$<$<AND:${stringzilla_msvc_},$<CONFIG:Debug>>:/fsanitize=address;/fsanitize=leak>"
                    "$<$<AND:${stringzilla_gcc_style_},$<CONFIG:Debug>>:-fsanitize=address;-fsanitize=undefined>"
        )
        target_link_options(
            ${target_} PRIVATE "$<$<AND:${stringzilla_msvc_link_},$<CONFIG:Debug>>:/fsanitize=address;/fsanitize=leak>"
            "$<$<AND:${stringzilla_gcc_style_link_},$<CONFIG:Debug>>:-fsanitize=address;-fsanitize=undefined>"
        )
    endif ()

    if (STRINGZILLA_BUILD_COVERAGE)
        target_compile_options(
            ${target_} PRIVATE "$<${stringzilla_clang_}:-fprofile-instr-generate;-fcoverage-mapping>"
        )
        target_link_options(
            ${target_} PRIVATE "$<${stringzilla_clang_link_}:-fprofile-instr-generate;-fcoverage-mapping>"
        )
    endif ()
endfunction ()

# Stamps the architecture id and the `STRINGZILLA_TARGET_*` verdicts of the ISA probes onto a target.
function (stringzilla_set_architecture_simd_definitions target_)
    target_compile_definitions(
        ${target_}
        PRIVATE "STRINGZILLA_ARCH_X8664_=$<BOOL:${stringzilla_arch_x8664_}>"
                "STRINGZILLA_ARCH_ARM64_=$<BOOL:${stringzilla_arch_arm64_}>"
                "STRINGZILLA_ARCH_RISCV64_=$<BOOL:${stringzilla_arch_riscv64_}>"
                "STRINGZILLA_ARCH_LOONGARCH64_=$<BOOL:${stringzilla_arch_loongarch64_}>"
                "STRINGZILLA_ARCH_PPC64_=$<BOOL:${stringzilla_arch_ppc64_}>"
                "STRINGZILLA_ARCH_WASM_=$<BOOL:${stringzilla_arch_wasm_}>"
    )
    target_link_libraries(${target_} PRIVATE $<BUILD_INTERFACE:stringzilla_cpu_capabilities_compiled>)
endfunction ()

# Apply the ABI floor (`-march`/`-mcpu`/`/arch`) that lets one compilation host every SIMD
# capability: each capability's headers scope its kernels with a target pragma, so the serial
# code runs on every CPU of the architecture.
# `STRINGZILLA_TARGET_ARCH` swaps the floor for a host-tuned build.
function (stringzilla_set_baseline_architecture_flags target_)
    if (STRINGZILLA_TARGET_ARCH)
        stringzilla_set_compiler_flags(${target_} "" "${STRINGZILLA_TARGET_ARCH}")
    elseif (stringzilla_arch_x8664_)
        if (MSVC)
            stringzilla_set_compiler_flags(${target_} "" "SSE2")
        else ()
            stringzilla_set_compiler_flags(${target_} "" "x86-64")
        endif ()
    elseif (stringzilla_arch_arm64_)
        if (MSVC)
            stringzilla_set_compiler_flags(${target_} "" "armv8.0")
        else ()
            stringzilla_set_compiler_flags(${target_} "" "armv8-a")
        endif ()
    elseif (stringzilla_arch_riscv64_)
        stringzilla_set_compiler_flags(${target_} "" "rv64gc")
    elseif (stringzilla_arch_loongarch64_)
        stringzilla_set_compiler_flags(${target_} "" "loongarch64")
    elseif (stringzilla_arch_ppc64_)
        stringzilla_set_compiler_flags(${target_} "" "")
        target_compile_options(${target_} PRIVATE "-mcpu=power8")
    else ()
        stringzilla_set_compiler_flags(${target_} "" "")
    endif ()
endfunction ()
