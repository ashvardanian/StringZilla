/**
 *  @file c/dispatch.h
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief The capability lists' shape and the kernel selection every dispatch unit shares.
 *
 *  The @c STRINGZILLA_TARGET_* verdicts come from the probes in `CMakeLists.txt`, which every
 *  binding builds through or links the output of. Builds that pass none get them from `types.h`,
 *  which reads the compiler flags.
 */
#ifndef STRINGZILLA_DISPATCH_H_
#define STRINGZILLA_DISPATCH_H_

#include <stringzilla/stringzilla.h> // `sz_kernel_punned_t`, `sz_capability_t`

#ifdef __cplusplus
extern "C" {
#endif

/** One capability group's kernels of one dispatch point: a null slot 0, then a kernel per bit of
 *  @c capabilities, ascending by bit. */
typedef struct sz_capability_kernels_t {
    sz_capability_t capabilities;
    sz_kernel_punned_t const *kernels;
} sz_capability_kernels_t;

/** The kernels of a group a dispatch point has none in: only the null slot 0. */
static sz_kernel_punned_t const sz_no_kernels_[1] = {STRINGZILLA_NULL};

/** The capability groups a binary may hold: its CPU's, and one per GPU vendor it was built for. */
typedef enum sz_capability_group_t {
    sz_capability_group_cpu_k,
    sz_capability_group_nvidia_k,
    sz_capability_group_amd_k,
    sz_capability_group_apple_k,
    sz_capability_groups_k,
} sz_capability_group_t;

/** Every bit of @p x at or below its highest set one, or zero. */
STRINGZILLA_CONSTEXPR sz_u64_t sz_u64_smear_down_(sz_u64_t x) {
    x |= x >> 1, x |= x >> 2, x |= x >> 4, x |= x >> 8, x |= x >> 16, x |= x >> 32;
    return x;
}

/** The capability group @p capabilities describes: each GPU vendor's bits sit above the CPU's. */
STRINGZILLA_CONSTEXPR sz_capability_group_t sz_capability_group_of_(sz_capability_t capabilities) {
    return (sz_capability_group_t)((capabilities >= sz_cap_cuda_k) + (capabilities >= sz_cap_rocm_k) +
                                   (capabilities >= sz_cap_metal_k));
}

/** The best capability @p capabilities shares with its group's kernels, or zero if none. */
STRINGZILLA_CONSTEXPR sz_capability_t sz_capability_pick_(
    sz_capability_t capabilities, sz_capability_kernels_t const groups[sz_capability_groups_k]) {
    sz_u64_t const at_or_below = sz_u64_smear_down_(capabilities &
                                                    groups[sz_capability_group_of_(capabilities)].capabilities);
    return at_or_below ^ (at_or_below >> 1);
}

/** The kernel of the best capability @p capabilities shares with its group's kernels, or null. */
STRINGZILLA_CONSTEXPR sz_kernel_punned_t sz_kernel_pick_(sz_capability_t capabilities,
                                                         sz_capability_kernels_t const groups[sz_capability_groups_k]) {
    sz_capability_kernels_t const *group = &groups[sz_capability_group_of_(capabilities)];
    sz_u64_t const at_or_below = sz_u64_smear_down_(capabilities & group->capabilities);
    return group->kernels[sz_u64_popcount(group->capabilities & at_or_below)];
}

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_DISPATCH_H_
