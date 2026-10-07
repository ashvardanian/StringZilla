# cmake/sz_cuda_probes.cmake — whether NVCC and its host compiler build `.cu` units for this machine's own CPU
#
# The CPU capability probes in `sz_isa_probe.cmake` ask the C compiler what it can emit. This one asks NVCC and its
# host compiler together: NVCC delegates host compilation but parses the host's headers itself on the device pass, so
# the pair decides, and the C++ compiler's answer is about the wrong toolchain - it accepts flags NVCC then chokes on.

# Probes `probes/cuda_native_arch.cu` at `-march=native` into the cached `sz_cuda_native_arch_compiles`. A failed probe
# leaves the `.cu` units on the baseline architecture and says nothing of the CPU capabilities.
function (sz_cuda_probe_native_arch_)
    if (NOT DEFINED sz_cuda_native_arch_compiles)
        set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
        try_compile(
            sz_cuda_native_arch_compiles ${CMAKE_BINARY_DIR}/sz_probes
            ${CMAKE_CURRENT_SOURCE_DIR}/probes/cuda_native_arch.cu
            CMAKE_FLAGS "-DCMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS} -Xcompiler=-march=native"
        )
    endif ()
    message(STATUS "Performing CUDA probe native_arch - compiles: ${sz_cuda_native_arch_compiles}")
endfunction ()
