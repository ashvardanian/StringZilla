# cmake/sz_cuda_probes.cmake — CUDA toolchain probes over the checked-in `probes/cuda_*.cu` sources.
#
# Separate from the `sz_*_isa_probes.cmake` family, which asks the C compiler what it can emit and answers in
# `SZ_ISA_CAPABILITIES`. These ask what NVCC and its host compiler will accept together: NVCC delegates host
# compilation but parses the host's headers itself on the device pass, so the pair decides, and putting the
# question to the C++ compiler gets an answer about the wrong toolchain - it accepts flags NVCC then chokes on.

# Whether `.cu` sources may be built for this machine's own instruction set, cached in
# `STRINGZILLA_CUDA_ACCEPTS_NATIVE_ARCH`. A `Failed` verdict leaves them on the baseline architecture and says
# nothing about the CPU tiers, which are compiled by the C and C++ compilers and probed separately.
function (sz_cuda_probe_native_arch_)
    if (DEFINED STRINGZILLA_CUDA_ACCEPTS_NATIVE_ARCH)
        return()
    endif ()
    set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
    try_compile(
        sz_cuda_native_arch_ ${CMAKE_BINARY_DIR}/sz_probes
        ${CMAKE_CURRENT_SOURCE_DIR}/probes/cuda_native_arch.cu
        CMAKE_FLAGS "-DCMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS} -Xcompiler=-march=native"
        OUTPUT_VARIABLE sz_cuda_native_arch_output_
    )
    set(STRINGZILLA_CUDA_ACCEPTS_NATIVE_ARCH
        "${sz_cuda_native_arch_}"
        CACHE INTERNAL "Whether NVCC and its host compiler build a `.cu` for this machine's own architecture"
    )
    if (sz_cuda_native_arch_)
        message(STATUS "Performing CUDA probe native_arch - Success")
    else ()
        message(STATUS "Performing CUDA probe native_arch - Failed")
    endif ()
endfunction ()
