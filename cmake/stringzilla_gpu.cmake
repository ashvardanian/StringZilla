# GPU languages and architectures, and the GPU capability rows.

# Enable CUDA if requested
if (STRINGZILLA_CUDA_ARCH)
    enable_language(CUDA)
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13)
        message(FATAL_ERROR "StringZilla needs CUDA 13 or newer, found ${CMAKE_CUDA_COMPILER_VERSION}")
    endif ()
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
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
    # `all` is the release set, of which each unit compiles only the lowest code its capability row lists, as PTX, so it
    # runs on every later GPU: the CUDA baseline at `compute_75`, clusters at `compute_90`, and cluster launch control at
    # `compute_100`. `native` maps each visible GPU to the code its capability rows need, family suffix included, where
    # CMake's own resolution would drop it. A native or named code compiles as SASS alone.
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
    if (STRINGZILLA_CUDA_ARCH STREQUAL "all")
        set(stringzilla_cuda_resolved_architectures_ 75-virtual 90-virtual 100-virtual)
    elseif (STRINGZILLA_CUDA_ARCH STREQUAL "native")
        set(stringzilla_cuda_resolved_architectures_)
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
                        "STRINGZILLA_CUDA_ARCH=native has no code for the visible ${visible_} GPU; name its code instead"
                )
            endif ()
            list(APPEND stringzilla_cuda_resolved_architectures_ ${native_code_})
        endforeach ()
    elseif (STRINGZILLA_CUDA_ARCH MATCHES "^[0-9]+[af]?$")
        set(stringzilla_cuda_resolved_architectures_ ${STRINGZILLA_CUDA_ARCH}-real)
    else ()
        message(
            FATAL_ERROR "STRINGZILLA_CUDA_ARCH takes all, native or one code like 100f, not `${STRINGZILLA_CUDA_ARCH}`"
        )
    endif ()
    if (CMAKE_VERSION VERSION_LESS 3.31 AND stringzilla_cuda_resolved_architectures_ MATCHES "1[0-9][0-9][af]")
        message(FATAL_ERROR "Three-digit \"a\" and \"f\" CUDA codes need CMake 3.31, found ${CMAKE_VERSION}")
    endif ()
    set(CMAKE_CUDA_ARCHITECTURES "${stringzilla_cuda_resolved_architectures_}")
    # NVCC's default leaves Turing and Ampere cubins uncompressed, ten times their compressed size.
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--compress-mode=size>)
    # Clusters and their distributed shared memory: H100, H200, GH200, and every GPU since.
    stringzilla_add_gpu_capability(
        hopper STRINGZILLA_TARGET_HOPPER
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
# the Rust `rocm` feature finds the runtime to link. One code object per listed target; `native` leaves CMake to list
# the visible GPUs through `rocm_agent_enumerator`.
if (STRINGZILLA_ROCM_ARCH)
    if (STRINGZILLA_ROCM_ARCH STREQUAL "all")
        set(CMAKE_HIP_ARCHITECTURES gfx942 gfx950 gfx1250 gfx1251)
    elseif (NOT STRINGZILLA_ROCM_ARCH STREQUAL "native")
        set(CMAKE_HIP_ARCHITECTURES ${STRINGZILLA_ROCM_ARCH})
    endif ()
    enable_language(HIP)
    set(stringzilla_rocm_resolved_architectures_ ${CMAKE_HIP_ARCHITECTURES})
    find_package(hip REQUIRED GLOBAL)
endif ()

# Metal compiles on the device, so the families only name the ones the unit serves.
if (STRINGZILLA_METAL_ARCH AND NOT APPLE)
    message(FATAL_ERROR "STRINGZILLA_METAL_ARCH needs an Apple platform")
elseif (STRINGZILLA_METAL_ARCH STREQUAL "native")
    message(FATAL_ERROR "STRINGZILLA_METAL_ARCH takes all or one family like apple9, as Metal compiles on the device")
elseif (STRINGZILLA_METAL_ARCH STREQUAL "all")
    set(stringzilla_metal_resolved_architectures_ apple7 apple9 apple10)
elseif (STRINGZILLA_METAL_ARCH)
    set(stringzilla_metal_resolved_architectures_ ${STRINGZILLA_METAL_ARCH})
endif ()
