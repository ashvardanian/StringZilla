/**
 *  @file include/stringzilla/types.metal
 *  @author Ash Vardanian
 *  @date October 2, 2026
 *  @brief The MSL prelude every Metal family source is compiled behind: the helpers each kernel file
 *      would otherwise mirror on its own.
 *
 *  @sa include/stringzilla/metal.h, which embeds this and compiles it ahead of each family's source
 *
 *  The runtime compiler takes one source string and no include path, so `metal.h` embeds this file
 *  once and joins it with a family's `metal.metal` into the one string each library compiles from.
 *
 *  Kernels address memory through the buffers the host binds, one per block a call touches. A
 *  block's contents point inside it by host address, which @ref sz_reach_metal_ turns into the
 *  kernel's own against the host address the block was bound at.
 */
#include <metal_stdlib>

using namespace metal;

/** Divides rounding up, as @c sz_size_divide_round_up does in the C headers MSL cannot include. */
template <typename scalar_type_>
constexpr scalar_type_ sz_size_divide_round_up_metal_(scalar_type_ number, scalar_type_ divisor) {
    return (number + divisor - 1) / divisor;
}

/** A sequence tape from @c sz_sequence_realloc_best: @b [count+1] byte offsets
 *  from the block's own start, then the bytes they address. */
struct sz_sequence_tape_metal_t {
    device ulong const *offsets;
};

/** The first byte of text @p index of @p tape. */
inline device uchar const *sz_sequence_tape_start_metal_(sz_sequence_tape_metal_t tape, ulong index) {
    return (device uchar const *)tape.offsets + tape.offsets[index];
}

/** The bytes text @p index of @p tape spans. */
inline ulong sz_sequence_tape_length_metal_(sz_sequence_tape_metal_t tape, ulong index) {
    return tape.offsets[index + 1] - tape.offsets[index];
}

/** The kernel's own address of @p host, a host address inside the block bound at @p block, which
 *  the host knows as @p block_host. */
template <typename type_>
inline device type_ *sz_reach_metal_(device uchar *block, ulong block_host, ulong host) {
    return (device type_ *)(block + (host - block_host));
}
