# GPU languages and architectures, and the GPU capability rows.

# Enable CUDA if requested
if (STRINGZILLA_BUILD_CUDA)
    enable_language(CUDA)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13)
        message(FATAL_ERROR "StringZilla needs CUDA 13 or newer, found ${CMAKE_CUDA_COMPILER_VERSION}")
    endif ()
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    # One cubin per device family, Hopper's arch-locked `90a` and family-wide `f` cubins from Blackwell on, plus a PTX
    # tail; every repo carries the same list.
    set(STRINGZILLA_CUDA_ARCHITECTURES
        "75-real;80-real;89-real;90a-real;100f-real;120f-real;80-virtual"
        CACHE STRING "Codes every CUDA target carries; override to match the machine at hand"
    )
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDAToolkit REQUIRED)
    # NVCC's host compiler takes the tuning flags too, where the pair accepts them.
    if (stringzilla_host_arch_flags_)
        include(cmake/stringzilla_cuda_probes.cmake)
        stringzilla_probe_cuda_native_arch("${stringzilla_host_arch_flags_}")
        if (stringzilla_cuda_native_arch_compiles_)
            add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${stringzilla_host_arch_flags_}>")
        endif ()
    endif ()
    # "native" maps each visible GPU to the code its capability rows need, family suffix included, where CMake's own
    # resolution would drop it; "all" and "all-major" stand for the lists CMake resolved them to.
    set(stringzilla_cuda_native_codes_
        75-real=75-real
        80-real=80-real
        86-real=86-real
        89-real=89-real
        90-real=90a-real
        100-real=100f-real
        103-real=103f-real
        110-real=110f-real
        120-real=120f-real
        121-real=121f-real
    )
    set(stringzilla_cuda_resolved_architectures_)
    foreach (code_ IN LISTS STRINGZILLA_CUDA_ARCHITECTURES)
        if (code_ STREQUAL "native")
            foreach (visible_ IN LISTS CMAKE_CUDA_ARCHITECTURES_NATIVE)
                set(native_code_)
                foreach (pair_ IN LISTS stringzilla_cuda_native_codes_)
                    if (pair_ MATCHES "^${visible_}=(.+)$")
                        set(native_code_ ${CMAKE_MATCH_1})
                    endif ()
                endforeach ()
                if (NOT native_code_)
                    message(
                        FATAL_ERROR
                            "STRINGZILLA_CUDA_ARCHITECTURES=native has no code for the visible ${visible_} GPU; "
                            "name its code instead"
                    )
                endif ()
                list(APPEND stringzilla_cuda_resolved_architectures_ ${native_code_})
            endforeach ()
        elseif (code_ STREQUAL "all")
            list(APPEND stringzilla_cuda_resolved_architectures_ ${CMAKE_CUDA_ARCHITECTURES_ALL})
        elseif (code_ STREQUAL "all-major")
            list(APPEND stringzilla_cuda_resolved_architectures_ ${CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR})
        else ()
            list(APPEND stringzilla_cuda_resolved_architectures_ ${code_})
        endif ()
    endforeach ()
    if (CMAKE_VERSION VERSION_LESS 3.31 AND stringzilla_cuda_resolved_architectures_ MATCHES "1[0-9][0-9][af]")
        message(FATAL_ERROR "Three-digit \"a\" and \"f\" CUDA codes need CMake 3.31, found ${CMAKE_VERSION}")
    endif ()
    set(CMAKE_CUDA_ARCHITECTURES "${stringzilla_cuda_resolved_architectures_}")
    # NVCC's default leaves Turing and Ampere cubins uncompressed, ten times their compressed size.
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--compress-mode=size>)
    # Clusters and their distributed shared memory: H100, H200, GH200, and every GPU since.
    stringzilla_add_gpu_capability(
        hopper STRINGZILLA_TARGET_HOPPER
        BIT 51
        CUDA_ARCHITECTURES
            90
            90a
            100
            100a
            100f
            103
            103a
            103f
            110
            110a
            110f
            120
            120a
            120f
            121
            121a
            121f
    )
    # Cluster launch control, letting a block take over the tile of one not yet started: B200, B300, RTX 50, and every
    # GPU since.
    stringzilla_add_gpu_capability(
        blackwell STRINGZILLA_TARGET_BLACKWELL
        BIT 52
        CUDA_ARCHITECTURES
            100
            100a
            100f
            103
            103a
            103f
            110
            110a
            110f
            120
            120a
            120f
            121
            121a
            121f
    )

    message(STATUS "CUDA support enabled")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif ()

# HIP comes on before any target, as the libraries carry the ROCm unit. The package lands `hip_DIR` in the cache, where
# the Rust `rocm` feature finds the runtime to link.
if (STRINGZILLA_BUILD_ROCM)
    enable_language(HIP)
    set(STRINGZILLA_ROCM_ARCHITECTURES
        "gfx942;gfx950;gfx1250;gfx1251"
        CACHE STRING "AMD targets the ROCm unit and test carry; override to match the machine at hand"
    )
    set(CMAKE_HIP_ARCHITECTURES "${STRINGZILLA_ROCM_ARCHITECTURES}")
    find_package(hip REQUIRED)
endif ()

if (STRINGZILLA_BUILD_METAL AND NOT APPLE)
    message(FATAL_ERROR "STRINGZILLA_BUILD_METAL needs an Apple platform")
endif ()
set(STRINGZILLA_METAL_ARCHITECTURES
    "apple7;apple9;apple10"
    CACHE STRING "Apple GPU families the Metal unit serves"
)
