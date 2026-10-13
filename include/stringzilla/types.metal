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

typedef uchar sz_u8_t;
typedef ushort sz_u16_t;
typedef uint sz_u32_t;
typedef ulong sz_u64_t;
typedef uint sz_rune_t;

/** Lookup table storage in the Metal constant address space. */
#define STRINGZILLA_CONSTANT constant

/** Device-resident lookup table storage in the Metal constant address space. */
#define STRINGZILLA_DEVICE_CONSTANT constant

/** Aligns shared records to the same boundary as their host declarations. */
#define sz_align_(n) alignas(n)

/** Divides rounding up, as @c sz_size_divide_round_up does in the C headers MSL cannot include. */
template <typename scalar_type_>
constexpr scalar_type_ sz_size_divide_round_up_metal_(scalar_type_ number, scalar_type_ divisor) {
    return (number + divisor - 1) / divisor;
}

/** The layout of a tape with 64-bit offsets, as @c sz_strings_u64_tape_k numbers it. */
constant ulong sz_strings_u64_tape_metal_k = 1;

/** A tape of @c sz_strings_view_t a kernel reads, bound as two buffers. */
struct sz_strings_tape_metal_t {

    /** The bytes of every string, back to back. */
    device uchar const *data;

    /** One offset per string and one past the last, 32 or 64 bits wide as @c layout says. */
    device uchar const *offsets;

    /** The layout as @c sz_strings_layout_t numbers it, never the accessor. */
    ulong layout;
};

/** Offset @p index of @p tape. */
inline ulong sz_strings_offset_metal_(sz_strings_tape_metal_t tape, ulong index) {
    device uint const *const offsets_u32 = (device uint const *)tape.offsets;
    device ulong const *const offsets_u64 = (device ulong const *)tape.offsets;
    return tape.layout == sz_strings_u64_tape_metal_k ? offsets_u64[index] : ulong(offsets_u32[index]);
}

/** The first byte of string @p index of @p tape. */
inline device uchar const *sz_strings_start_metal_(sz_strings_tape_metal_t tape, ulong index) {
    return tape.data + sz_strings_offset_metal_(tape, index);
}

/** The bytes string @p index of @p tape spans. */
inline ulong sz_strings_length_metal_(sz_strings_tape_metal_t tape, ulong index) {
    return sz_strings_offset_metal_(tape, index + 1) - sz_strings_offset_metal_(tape, index);
}

/** The kernel's own address of @p host, a host address inside the block bound at @p block, which
 *  the host knows as @p block_host. */
template <typename type_>
inline device type_ *sz_reach_metal_(device uchar *block, ulong block_host, ulong host) {
    return (device type_ *)(block + (host - block_host));
}

/** Decodes one well-formed rune, returning zero for an opaque malformed byte. */
inline uint sz_utf8_decode_rune_metal_(device uchar const *text, ulong length, ulong offset, thread uint &rune) {
    if (offset >= length) return 0;
    uint const lead = text[offset];
    if (lead < 0x80) {
        rune = lead;
        return 1;
    }
    if (lead < 0xC2 || lead > 0xF4) return 0;
    uint const needed = lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
    if (length - offset < needed) return 0;
    uint const second = text[offset + 1];
    if ((second & 0xC0) != 0x80 || (lead == 0xE0 && second < 0xA0) || (lead == 0xED && second >= 0xA0) ||
        (lead == 0xF0 && second < 0x90) || (lead == 0xF4 && second >= 0x90))
        return 0;
    for (uint index = 2; index < needed; ++index)
        if ((text[offset + index] & 0xC0) != 0x80) return 0;
    if (needed == 2) rune = (lead & 0x1F) << 6 | (second & 0x3F);
    else if (needed == 3) rune = (lead & 0x0F) << 12 | (second & 0x3F) << 6 | (text[offset + 2] & 0x3F);
    else
        rune = (lead & 0x07) << 18 | (second & 0x3F) << 12 | (text[offset + 2] & 0x3F) << 6 | (text[offset + 3] & 0x3F);
    return needed;
}
