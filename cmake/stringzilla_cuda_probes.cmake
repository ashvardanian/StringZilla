# cmake/stringzilla_cuda_probes.cmake — whether NVCC and its host compiler build `.cu` units for this machine's own CPU
#
# The CPU capability probes in `stringzilla_probes.cmake` ask the C compiler what it can emit. This one asks NVCC and its
# host compiler together: NVCC delegates host compilation but parses the host's headers itself on the device pass, so
# the pair decides, and the C++ compiler's answer is about the wrong toolchain - it accepts flags NVCC then chokes on.

# Probes `probes/cuda_native_arch.cu` at `-march=native` into the cached `stringzilla_cuda_native_arch_compiles_`. A
# failed probe leaves the `.cu` units on the baseline architecture and says nothing of the CPU capabilities.
function (stringzilla_probe_cuda_native_arch)
    if (NOT DEFINED stringzilla_cuda_native_arch_compiles_)
        set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
        try_compile(
            stringzilla_cuda_native_arch_compiles_ ${CMAKE_BINARY_DIR}/stringzilla_probes
            ${CMAKE_CURRENT_SOURCE_DIR}/probes/cuda_native_arch.cu
            CMAKE_FLAGS "-DCMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS} -Xcompiler=-march=native"
        )
    endif ()
    message(STATUS "Performing CUDA probe native_arch - compiles: ${stringzilla_cuda_native_arch_compiles_}")
endfunction ()
