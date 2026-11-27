/**
 *  @file c/stringzilla/capabilities.h
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Capabilities and their groups, kernel kinds and signatures, and CPU and GPU detection.
 *
 *  A capability is one bit, like @c sz_cap_haswell_k or @c sz_cap_cuda_k, and a capability group is
 *  the range of bits of one kind of hardware: the CPU's, NVIDIA's, AMD's or Apple's. A mask names
 *  one device: the CPU, through @c sz_capabilities_enabled_cpu, or one GPU of one vendor, through
 *  its own @c sz_capabilities_enabled_cuda, @c sz_capabilities_enabled_rocm or
 *  @c sz_capabilities_enabled_metal, indexed by that runtime's ordinal.
 *
 *  Within each architecture the bits ascend by preference, so the highest bit a mask shares with
 *  a verb's kernels names the kernel that its @c _best and @c sz_find_kernel_punned both pick.
 */
#ifndef STRINGZILLA_CAPABILITIES_H_
#define STRINGZILLA_CAPABILITIES_H_

#include "stringzilla/types.h" // `sz_u64_t`, `sz_status_t`, `STRINGZILLA_TARGET_*`

/** Buffer size @c sz_capabilities_name never overruns, including its null terminator. */
#define STRINGZILLA_CAPABILITIES_NAME_CAPACITY 256

#ifdef __cplusplus
extern "C" {
#endif

/** Serial (non-SIMD) fallback capability. Always available. */
#define sz_cap_serial_k ((sz_capability_t)1)

/** Mask representing any capability. */
#define sz_cap_any_k (~(sz_capability_t)0)

/*  CPU capabilities, each architecture's in a contiguous run of bits ascending by preference, so
 *  the highest bit a mask shares with a verb's kernels names the one to run. */

/** x86 SSE4.2 with AES-NI, carry-less multiplication, BMI and LZCNT, as on Westmere. */
#define sz_cap_westmere_k ((sz_capability_t)1 << 1)

/** x86 SSSE3 and SSE4.1 with the SHA extensions, as on Goldmont. */
#define sz_cap_goldmont_k ((sz_capability_t)1 << 2)

/** x86 AVX2 with FMA, BMI2, POPCNT, LZCNT and carry-less multiplication, as on Haswell. */
#define sz_cap_haswell_k ((sz_capability_t)1 << 3)

/** x86 AVX-512 F, VL, BW and DQ with AES-NI, as on Skylake servers. */
#define sz_cap_skylake_k ((sz_capability_t)1 << 4)

/** x86 AVX-512 VBMI, VBMI2 and VNNI with VAES, VPCLMULQDQ and SHA, as on Ice Lake. */
#define sz_cap_icelake_k ((sz_capability_t)1 << 5)

/** Arm NEON, the Advanced SIMD of every Armv8.0-A core. */
#define sz_cap_neon_k ((sz_capability_t)1 << 6)

/** Arm NEON with the AES instructions, and the SHA2 ones its <tt>+crypto</tt> target brings. */
#define sz_cap_neonaes_k ((sz_capability_t)1 << 7)

/** Arm NEON with the SHA2 instructions, and the AES ones its <tt>+crypto</tt> target brings. */
#define sz_cap_neonsha_k ((sz_capability_t)1 << 8)

/** Arm SVE, the Scalable Vector Extension, optional since Armv8.2-A. */
#define sz_cap_sve_k ((sz_capability_t)1 << 9)

/** Arm SVE2, optional since Armv9.0-A. */
#define sz_cap_sve2_k ((sz_capability_t)1 << 10)

/** Arm SVE2 with its AES instructions, and the NEON AES and SHA2 its <tt>+crypto</tt> brings. */
#define sz_cap_sve2aes_k ((sz_capability_t)1 << 11)

/** RISC-V Vector extension 1.0, with length-agnostic registers. */
#define sz_cap_rvv_k ((sz_capability_t)1 << 12)

/** RISC-V Vector Crypto: the Zvkned AES, Zvknhb SHA-2 and Zvkg GHASH extensions. */
#define sz_cap_rvvcrypto_k ((sz_capability_t)1 << 13)

/** WebAssembly 128-bit SIMD. */
#define sz_cap_v128_k ((sz_capability_t)1 << 14)

/** WebAssembly 128-bit SIMD with the relaxed SIMD instructions. */
#define sz_cap_v128relaxed_k ((sz_capability_t)1 << 15)

/** LoongArch Advanced SIMD eXtension, the 256-bit LASX. */
#define sz_cap_loongsonasx_k ((sz_capability_t)1 << 16)

/** IBM Power Vector-Scalar eXtension at the POWER9 level, ISA 3.0. */
#define sz_cap_powervsx_k ((sz_capability_t)1 << 17)

/*  GPU capabilities, as each vendor's <tt>sz_&lt;vendor&gt;_capabilities_*</tt> functions report them for one
 *  device, grouped by vendor above bit 47 to leave room for more CPU capabilities and more per
 *  vendor. Each vendor's baseline runs on all its devices, as @c serial runs on every CPU, and each
 *  later tier on every device from its generation on: @c ampere from compute capability 8.0, for
 *  warp reductions and asynchronous copies into shared memory, @c hopper from 9.0, for its clusters
 *  and their distributed shared memory, and @c blackwell from 10.0, for cluster launch control,
 *  which lets a block take over the tile of one not yet started. AMD's tiers are generations
 *  instead: @c cdna3 on gfx942, @c cdna4 on gfx950, and @c cdna5 on gfx1250 and gfx1251, each
 *  moving a one-lane shift through a DPP form rather than the LDS crossbar. */

/** NVIDIA GPUs through CUDA, on every device. */
#define sz_cap_cuda_k ((sz_capability_t)1 << 48)

/** NVIDIA GPUs from compute capability 8.0, for warp reductions and asynchronous copies into
 *  shared memory. */
#define sz_cap_ampere_k ((sz_capability_t)1 << 49)

/** NVIDIA GPUs from compute capability 9.0, for clusters and their distributed shared memory. */
#define sz_cap_hopper_k ((sz_capability_t)1 << 51)

/** NVIDIA GPUs from compute capability 10.0, for cluster launch control. */
#define sz_cap_blackwell_k ((sz_capability_t)1 << 52)

/** AMD GPUs through ROCm, on every device. */
#define sz_cap_rocm_k ((sz_capability_t)1 << 56)

/** AMD gfx942 GPUs on 64-lane wavefronts, for DPP wavefront shifts, as on MI300X and MI325X. */
#define sz_cap_cdna3_k ((sz_capability_t)1 << 57)

/** AMD gfx950 GPUs, for the DPP wavefront shifts of @c cdna3, as on MI350X and MI355X. */
#define sz_cap_cdna4_k ((sz_capability_t)1 << 58)

/** AMD gfx1250 and gfx1251 GPUs on a 32-lane wavefront, for DPP row shifts bridged by
 *  @c v_permlanex16, as on MI450X and MI430X. */
#define sz_cap_cdna5_k ((sz_capability_t)1 << 59)

/** Apple GPUs through Metal, on every device. */
#define sz_cap_metal_k ((sz_capability_t)1 << 60)

/** Every GPU capability above, which the CPU queries never detect, compile or enable. */
#define sz_cap_gpus_k                                                                                          \
    (sz_cap_cuda_k | sz_cap_ampere_k | sz_cap_hopper_k | sz_cap_blackwell_k | sz_cap_rocm_k | sz_cap_cdna3_k | \
     sz_cap_cdna4_k | sz_cap_cdna5_k | sz_cap_metal_k)

/** Every CPU capability, the bits below the first GPU vendor's. */
#define sz_cap_cpus_k (sz_cap_cuda_k - 1)

/** Every named capability in bit order, as the bindings spell them, then a null name. */
static struct {
    char const *name;
    sz_capability_t flag;
} const sz_capability_names_[] = {
    {"serial", sz_cap_serial_k},
    {"westmere", sz_cap_westmere_k},
    {"goldmont", sz_cap_goldmont_k},
    {"haswell", sz_cap_haswell_k},
    {"skylake", sz_cap_skylake_k},
    {"icelake", sz_cap_icelake_k},
    {"neon", sz_cap_neon_k},
    {"neonaes", sz_cap_neonaes_k},
    {"neonsha", sz_cap_neonsha_k},
    {"sve", sz_cap_sve_k},
    {"sve2", sz_cap_sve2_k},
    {"sve2aes", sz_cap_sve2aes_k},
    {"rvv", sz_cap_rvv_k},
    {"rvvcrypto", sz_cap_rvvcrypto_k},
    {"v128", sz_cap_v128_k},
    {"v128relaxed", sz_cap_v128relaxed_k},
    {"loongsonasx", sz_cap_loongsonasx_k},
    {"powervsx", sz_cap_powervsx_k},
    {"cuda", sz_cap_cuda_k},
    {"ampere", sz_cap_ampere_k},
    {"hopper", sz_cap_hopper_k},
    {"blackwell", sz_cap_blackwell_k},
    {"rocm", sz_cap_rocm_k},
    {"cdna3", sz_cap_cdna3_k},
    {"cdna4", sz_cap_cdna4_k},
    {"cdna5", sz_cap_cdna5_k},
    {"metal", sz_cap_metal_k},
    {0, 0},
};

/** Writes the names of @p capabilities into @p buffer, behind @c sz_capabilities_name. */
STRINGZILLA_CONSTEXPR sz_size_t sz_capabilities_name_(sz_capability_t capabilities, char *buffer, sz_size_t capacity) {
    if (!capacity) return 0;
    sz_size_t length = 0;
    for (sz_size_t entry = 0; sz_capability_names_[entry].name; ++entry) {
        if (!(capabilities & sz_capability_names_[entry].flag)) continue;
        char const *character = length ? "," : "";
        for (; *character && length + 1 < capacity; ++character) buffer[length++] = *character;
        for (character = sz_capability_names_[entry].name; *character && length + 1 < capacity; ++character)
            buffer[length++] = *character;
    }
    buffer[length] = '\0';
    return length;
}

/** Every dispatch point a finder can resolve: one kind per verb, grouped by family. */
typedef enum sz_kernel_kind_t {

    /** No kernel: what a finder resolves nothing for. */
    sz_kernel_unknown_k = 0,

    /** Byte-wise equality of two equal-length strings. */
    sz_kernel_equal_k,

    /** Lexicographic three-way comparison. */
    sz_kernel_order_k,

    /** Gather strings into a tape. */
    sz_kernel_strings_to_tape_k,

    /** Write strings into NUL-padded rows. */
    sz_kernel_strings_to_rows_k,

    /** Escape strings for a text format. */
    sz_kernel_strings_escape_k,

    /** Unescape strings from a text format. */
    sz_kernel_strings_unescape_k,

    /** Map every byte of every string through a 256-entry table. */
    sz_kernel_strings_lookup_k,

    /** Bring strings into a Unicode normalization form. */
    sz_kernel_strings_normalize_k,

    /** Split a text at the bytes of a set into a tape. */
    sz_kernel_strings_split_k,

    /** Split a text at Unicode line breaks into a tape. */
    sz_kernel_strings_split_lines_k,

    /** Join columns of strings with frames around and between their rows. */
    sz_kernel_strings_join_k,

    /** First occurrence of a substring. */
    sz_kernel_find_k,

    /** Last occurrence of a substring. */
    sz_kernel_rfind_k,

    /** First occurrence of a byte. */
    sz_kernel_find_byte_k,

    /** Last occurrence of a byte. */
    sz_kernel_rfind_byte_k,

    /** First byte belonging to a set. */
    sz_kernel_find_byteset_k,

    /** Last byte belonging to a set. */
    sz_kernel_rfind_byteset_k,

    /** First occurrence of a substring in every string of a collection. */
    sz_kernel_find_strings_k,

    /** Sum of all bytes. */
    sz_kernel_bytesum_k,

    /** One-shot 64-bit hash. */
    sz_kernel_hash_k,

    /** One string hashed under several seeds at once. */
    sz_kernel_hash_multiseed_k,

    /** Every string of a collection hashed under one seed. */
    sz_kernel_hash_strings_k,

    /** Pseudo-random bytes from a seed. */
    sz_kernel_fill_random_k,

    /** Streaming hash state initialization. */
    sz_kernel_hash_state_init_k,

    /** Streaming hash state update. */
    sz_kernel_hash_state_update_k,

    /** Streaming hash state digest. */
    sz_kernel_hash_state_digest_k,

    /** SHA-256 state initialization. */
    sz_kernel_sha256_state_init_k,

    /** SHA-256 state update. */
    sz_kernel_sha256_state_update_k,

    /** SHA-256 state digest. */
    sz_kernel_sha256_state_digest_k,

    /** SHA-256 update of many independent states. */
    sz_kernel_sha256_multistate_update_k,

    /** SHA-256 digest of many independent states. */
    sz_kernel_sha256_multistate_digest_k,

    /** AES-256 key schedule. */
    sz_kernel_aes256_key_init_k,

    /** AES-256-GCM key schedule and hash subkey. */
    sz_kernel_aes256_gcm_key_init_k,

    /** AES-256 counter-mode keystream XOR. */
    sz_kernel_aes256_ctr_xor_k,

    /** One-shot AES-256-GCM encryption. */
    sz_kernel_aes256_gcm_encrypt_k,

    /** One-shot AES-256-GCM decryption with tag verification. */
    sz_kernel_aes256_gcm_decrypt_k,

    /** Streaming AES-256-GCM encryptor initialization. */
    sz_kernel_aes256_gcm_encryptor_init_k,

    /** Streaming AES-256-GCM encryptor associated data. */
    sz_kernel_aes256_gcm_encryptor_associate_k,

    /** Streaming AES-256-GCM encryptor update. */
    sz_kernel_aes256_gcm_encryptor_update_k,

    /** Streaming AES-256-GCM encryptor tag. */
    sz_kernel_aes256_gcm_encryptor_digest_k,

    /** Streaming AES-256-GCM decryptor initialization. */
    sz_kernel_aes256_gcm_decryptor_init_k,

    /** Streaming AES-256-GCM decryptor associated data. */
    sz_kernel_aes256_gcm_decryptor_associate_k,

    /** Streaming AES-256-GCM decryptor update, before verification. */
    sz_kernel_aes256_gcm_decryptor_update_unverified_k,

    /** Streaming AES-256-GCM decryptor tag verification. */
    sz_kernel_aes256_gcm_decryptor_verify_k,

    /** Stable argsort of a collection of strings. */
    sz_kernel_sort_order_k,

    /** Stable case-insensitive argsort of a collection of strings. */
    sz_kernel_sort_uncased_order_k,

    /** Intersection of two collections of strings. */
    sz_kernel_intersect_k,

    /** Count of UTF-8 runes. */
    sz_kernel_utf8_count_k,

    /** Validation and rune count of every string of a collection. */
    sz_kernel_utf8_count_strings_k,

    /** Offset of the n-th UTF-8 rune. */
    sz_kernel_utf8_seek_k,

    /** Batch UTF-8 decoding into code points. */
    sz_kernel_utf8_decode_k,

    /** Newline boundaries. */
    sz_kernel_utf8_newlines_k,

    /** Whitespace boundaries. */
    sz_kernel_utf8_whitespaces_k,

    /** Delimiter boundaries. */
    sz_kernel_utf8_delimiters_k,

    /** Unicode word boundaries. */
    sz_kernel_utf8_wordbreaks_k,

    /** Unicode grapheme cluster boundaries. */
    sz_kernel_utf8_graphemes_k,

    /** Unicode sentence boundaries. */
    sz_kernel_utf8_sentences_k,

    /** Unicode line-break opportunities. */
    sz_kernel_utf8_linebreaks_k,

    /** Unicode normalization. */
    sz_kernel_utf8_norm_k,

    /** First byte not in the requested normalization form. */
    sz_kernel_utf8_find_denormalized_k,

    /** Unicode case folding. */
    sz_kernel_utf8_uncased_fold_k,

    /** Needle analysis for case-insensitive substring search. */
    sz_kernel_utf8_uncased_needle_init_k,

    /** Case-insensitive substring search. */
    sz_kernel_utf8_uncased_search_k,

    /** Case-insensitive three-way comparison. */
    sz_kernel_utf8_uncased_order_k,

    /** First rune that has a case. */
    sz_kernel_utf8_find_cased_k,

    /** Levenshtein engine preparation. */
    sz_kernel_levenshtein_engine_init_k,

    /** Levenshtein distances of a batch against the prepared queries. */
    sz_kernel_levenshtein_distances_k,

    /** N-grams engine preparation. */
    sz_kernel_ngrams_engine_init_k,

    /** N-gram overlaps of a batch against the prepared queries. */
    sz_kernel_ngrams_overlaps_k,

    /** MinHash sketches of a batch of texts. */
    sz_kernel_ngrams_sketches_k,

    /** Substrings engine preparation. */
    sz_kernel_substrings_engine_init_k,

    /** Haystacks holding any of the prepared needles. */
    sz_kernel_substrings_contains_k,

    /** Occurrence counts of the prepared needles. */
    sz_kernel_substrings_counts_k,

    /** Occurrences of the prepared needles. */
    sz_kernel_substrings_find_k,

    /** Replacement of the prepared needles. */
    sz_kernel_substrings_replace_k,

    /** BM25 scores of the prepared needles. */
    sz_kernel_substrings_bm25_scores_k,

    /** Regex engine preparation. */
    sz_kernel_regex_engine_init_k,

    /** Haystacks matching any of the prepared patterns. */
    sz_kernel_regex_contains_k,

    /** Match counts of the prepared patterns. */
    sz_kernel_regex_counts_k,

    /** Matches of the prepared patterns. */
    sz_kernel_regex_find_k,

    /** Replacement of the prepared patterns' matches, naming their groups. */
    sz_kernel_regex_replace_k,

    /** Groups of one prepared pattern's first match per haystack. */
    sz_kernel_regex_extract_k,

    /** Subdocuments engine preparation. */
    sz_kernel_subdocuments_engine_init_k,

    /** Documents holding a node any of the prepared paths selects. */
    sz_kernel_subdocuments_contains_k,

    /** Match counts of the prepared paths. */
    sz_kernel_subdocuments_counts_k,

    /** Matches of the prepared paths. */
    sz_kernel_subdocuments_find_k,
} sz_kernel_kind_t;

/** Canonical name of a kernel kind - the spelling bindings parse and interchange formats carry;
 *  "unknown" for unrecognized values. */
STRINGZILLA_CONSTEXPR char const *sz_kernel_name(sz_kernel_kind_t kind) {
    switch (kind) {
    case sz_kernel_unknown_k: return "unknown";
    case sz_kernel_equal_k: return "equal";
    case sz_kernel_order_k: return "order";
    case sz_kernel_strings_to_tape_k: return "strings_to_tape";
    case sz_kernel_strings_to_rows_k: return "strings_to_rows";
    case sz_kernel_strings_escape_k: return "strings_escape";
    case sz_kernel_strings_unescape_k: return "strings_unescape";
    case sz_kernel_strings_lookup_k: return "strings_lookup";
    case sz_kernel_strings_normalize_k: return "strings_normalize";
    case sz_kernel_strings_split_k: return "strings_split";
    case sz_kernel_strings_split_lines_k: return "strings_split_lines";
    case sz_kernel_strings_join_k: return "strings_join";
    case sz_kernel_find_k: return "find";
    case sz_kernel_rfind_k: return "rfind";
    case sz_kernel_find_byte_k: return "find_byte";
    case sz_kernel_rfind_byte_k: return "rfind_byte";
    case sz_kernel_find_byteset_k: return "find_byteset";
    case sz_kernel_rfind_byteset_k: return "rfind_byteset";
    case sz_kernel_find_strings_k: return "find_strings";
    case sz_kernel_bytesum_k: return "bytesum";
    case sz_kernel_hash_k: return "hash";
    case sz_kernel_hash_multiseed_k: return "hash_multiseed";
    case sz_kernel_hash_strings_k: return "hash_strings";
    case sz_kernel_fill_random_k: return "fill_random";
    case sz_kernel_hash_state_init_k: return "hash_state_init";
    case sz_kernel_hash_state_update_k: return "hash_state_update";
    case sz_kernel_hash_state_digest_k: return "hash_state_digest";
    case sz_kernel_sha256_state_init_k: return "sha256_state_init";
    case sz_kernel_sha256_state_update_k: return "sha256_state_update";
    case sz_kernel_sha256_state_digest_k: return "sha256_state_digest";
    case sz_kernel_sha256_multistate_update_k: return "sha256_multistate_update";
    case sz_kernel_sha256_multistate_digest_k: return "sha256_multistate_digest";
    case sz_kernel_aes256_key_init_k: return "aes256_key_init";
    case sz_kernel_aes256_gcm_key_init_k: return "aes256_gcm_key_init";
    case sz_kernel_aes256_ctr_xor_k: return "aes256_ctr_xor";
    case sz_kernel_aes256_gcm_encrypt_k: return "aes256_gcm_encrypt";
    case sz_kernel_aes256_gcm_decrypt_k: return "aes256_gcm_decrypt";
    case sz_kernel_aes256_gcm_encryptor_init_k: return "aes256_gcm_encryptor_init";
    case sz_kernel_aes256_gcm_encryptor_associate_k: return "aes256_gcm_encryptor_associate";
    case sz_kernel_aes256_gcm_encryptor_update_k: return "aes256_gcm_encryptor_update";
    case sz_kernel_aes256_gcm_encryptor_digest_k: return "aes256_gcm_encryptor_digest";
    case sz_kernel_aes256_gcm_decryptor_init_k: return "aes256_gcm_decryptor_init";
    case sz_kernel_aes256_gcm_decryptor_associate_k: return "aes256_gcm_decryptor_associate";
    case sz_kernel_aes256_gcm_decryptor_update_unverified_k: return "aes256_gcm_decryptor_update_unverified";
    case sz_kernel_aes256_gcm_decryptor_verify_k: return "aes256_gcm_decryptor_verify";
    case sz_kernel_sort_order_k: return "sort_order";
    case sz_kernel_sort_uncased_order_k: return "sort_uncased_order";
    case sz_kernel_intersect_k: return "intersect";
    case sz_kernel_utf8_count_k: return "utf8_count";
    case sz_kernel_utf8_count_strings_k: return "utf8_count_strings";
    case sz_kernel_utf8_seek_k: return "utf8_seek";
    case sz_kernel_utf8_decode_k: return "utf8_decode";
    case sz_kernel_utf8_newlines_k: return "utf8_newlines";
    case sz_kernel_utf8_whitespaces_k: return "utf8_whitespaces";
    case sz_kernel_utf8_delimiters_k: return "utf8_delimiters";
    case sz_kernel_utf8_wordbreaks_k: return "utf8_wordbreaks";
    case sz_kernel_utf8_graphemes_k: return "utf8_graphemes";
    case sz_kernel_utf8_sentences_k: return "utf8_sentences";
    case sz_kernel_utf8_linebreaks_k: return "utf8_linebreaks";
    case sz_kernel_utf8_norm_k: return "utf8_norm";
    case sz_kernel_utf8_find_denormalized_k: return "utf8_find_denormalized";
    case sz_kernel_utf8_uncased_fold_k: return "utf8_uncased_fold";
    case sz_kernel_utf8_uncased_needle_init_k: return "utf8_uncased_needle_init";
    case sz_kernel_utf8_uncased_search_k: return "utf8_uncased_search";
    case sz_kernel_utf8_uncased_order_k: return "utf8_uncased_order";
    case sz_kernel_utf8_find_cased_k: return "utf8_find_cased";
    case sz_kernel_levenshtein_engine_init_k: return "levenshtein_engine_init";
    case sz_kernel_levenshtein_distances_k: return "levenshtein_distances";
    case sz_kernel_ngrams_engine_init_k: return "ngrams_engine_init";
    case sz_kernel_ngrams_overlaps_k: return "ngrams_overlaps";
    case sz_kernel_ngrams_sketches_k: return "ngrams_sketches";
    case sz_kernel_substrings_engine_init_k: return "substrings_engine_init";
    case sz_kernel_substrings_contains_k: return "substrings_contains";
    case sz_kernel_substrings_counts_k: return "substrings_counts";
    case sz_kernel_substrings_find_k: return "substrings_find";
    case sz_kernel_substrings_replace_k: return "substrings_replace";
    case sz_kernel_substrings_bm25_scores_k: return "substrings_bm25_scores";
    case sz_kernel_regex_engine_init_k: return "regex_engine_init";
    case sz_kernel_regex_contains_k: return "regex_contains";
    case sz_kernel_regex_counts_k: return "regex_counts";
    case sz_kernel_regex_find_k: return "regex_find";
    case sz_kernel_regex_replace_k: return "regex_replace";
    case sz_kernel_regex_extract_k: return "regex_extract";
    case sz_kernel_subdocuments_engine_init_k: return "subdocuments_engine_init";
    case sz_kernel_subdocuments_contains_k: return "subdocuments_contains";
    case sz_kernel_subdocuments_counts_k: return "subdocuments_counts";
    case sz_kernel_subdocuments_find_k: return "subdocuments_find";
    default: return "unknown";
    }
}

/** Inverse of @c sz_kernel_name over an explicit-length string; @c sz_kernel_unknown_k for
 *  unrecognized names. */
STRINGZILLA_CONSTEXPR sz_kernel_kind_t sz_kernel_named(char const *name, sz_size_t length) {
    if (sz_same_literal_(name, length, "equal")) return sz_kernel_equal_k;
    if (sz_same_literal_(name, length, "order")) return sz_kernel_order_k;
    if (sz_same_literal_(name, length, "strings_to_tape")) return sz_kernel_strings_to_tape_k;
    if (sz_same_literal_(name, length, "strings_to_rows")) return sz_kernel_strings_to_rows_k;
    if (sz_same_literal_(name, length, "strings_escape")) return sz_kernel_strings_escape_k;
    if (sz_same_literal_(name, length, "strings_unescape")) return sz_kernel_strings_unescape_k;
    if (sz_same_literal_(name, length, "strings_lookup")) return sz_kernel_strings_lookup_k;
    if (sz_same_literal_(name, length, "strings_normalize")) return sz_kernel_strings_normalize_k;
    if (sz_same_literal_(name, length, "strings_split")) return sz_kernel_strings_split_k;
    if (sz_same_literal_(name, length, "strings_split_lines")) return sz_kernel_strings_split_lines_k;
    if (sz_same_literal_(name, length, "strings_join")) return sz_kernel_strings_join_k;
    if (sz_same_literal_(name, length, "find")) return sz_kernel_find_k;
    if (sz_same_literal_(name, length, "rfind")) return sz_kernel_rfind_k;
    if (sz_same_literal_(name, length, "find_byte")) return sz_kernel_find_byte_k;
    if (sz_same_literal_(name, length, "rfind_byte")) return sz_kernel_rfind_byte_k;
    if (sz_same_literal_(name, length, "find_byteset")) return sz_kernel_find_byteset_k;
    if (sz_same_literal_(name, length, "rfind_byteset")) return sz_kernel_rfind_byteset_k;
    if (sz_same_literal_(name, length, "find_strings")) return sz_kernel_find_strings_k;
    if (sz_same_literal_(name, length, "bytesum")) return sz_kernel_bytesum_k;
    if (sz_same_literal_(name, length, "hash")) return sz_kernel_hash_k;
    if (sz_same_literal_(name, length, "hash_multiseed")) return sz_kernel_hash_multiseed_k;
    if (sz_same_literal_(name, length, "hash_strings")) return sz_kernel_hash_strings_k;
    if (sz_same_literal_(name, length, "fill_random")) return sz_kernel_fill_random_k;
    if (sz_same_literal_(name, length, "hash_state_init")) return sz_kernel_hash_state_init_k;
    if (sz_same_literal_(name, length, "hash_state_update")) return sz_kernel_hash_state_update_k;
    if (sz_same_literal_(name, length, "hash_state_digest")) return sz_kernel_hash_state_digest_k;
    if (sz_same_literal_(name, length, "sha256_state_init")) return sz_kernel_sha256_state_init_k;
    if (sz_same_literal_(name, length, "sha256_state_update")) return sz_kernel_sha256_state_update_k;
    if (sz_same_literal_(name, length, "sha256_state_digest")) return sz_kernel_sha256_state_digest_k;
    if (sz_same_literal_(name, length, "sha256_multistate_update")) return sz_kernel_sha256_multistate_update_k;
    if (sz_same_literal_(name, length, "sha256_multistate_digest")) return sz_kernel_sha256_multistate_digest_k;
    if (sz_same_literal_(name, length, "aes256_key_init")) return sz_kernel_aes256_key_init_k;
    if (sz_same_literal_(name, length, "aes256_gcm_key_init")) return sz_kernel_aes256_gcm_key_init_k;
    if (sz_same_literal_(name, length, "aes256_ctr_xor")) return sz_kernel_aes256_ctr_xor_k;
    if (sz_same_literal_(name, length, "aes256_gcm_encrypt")) return sz_kernel_aes256_gcm_encrypt_k;
    if (sz_same_literal_(name, length, "aes256_gcm_decrypt")) return sz_kernel_aes256_gcm_decrypt_k;
    if (sz_same_literal_(name, length, "aes256_gcm_encryptor_init")) return sz_kernel_aes256_gcm_encryptor_init_k;
    if (sz_same_literal_(name, length, "aes256_gcm_encryptor_associate"))
        return sz_kernel_aes256_gcm_encryptor_associate_k;
    if (sz_same_literal_(name, length, "aes256_gcm_encryptor_update")) return sz_kernel_aes256_gcm_encryptor_update_k;
    if (sz_same_literal_(name, length, "aes256_gcm_encryptor_digest")) return sz_kernel_aes256_gcm_encryptor_digest_k;
    if (sz_same_literal_(name, length, "aes256_gcm_decryptor_init")) return sz_kernel_aes256_gcm_decryptor_init_k;
    if (sz_same_literal_(name, length, "aes256_gcm_decryptor_associate"))
        return sz_kernel_aes256_gcm_decryptor_associate_k;
    if (sz_same_literal_(name, length, "aes256_gcm_decryptor_update_unverified"))
        return sz_kernel_aes256_gcm_decryptor_update_unverified_k;
    if (sz_same_literal_(name, length, "aes256_gcm_decryptor_verify")) return sz_kernel_aes256_gcm_decryptor_verify_k;
    if (sz_same_literal_(name, length, "sort_order")) return sz_kernel_sort_order_k;
    if (sz_same_literal_(name, length, "sort_uncased_order")) return sz_kernel_sort_uncased_order_k;
    if (sz_same_literal_(name, length, "intersect")) return sz_kernel_intersect_k;
    if (sz_same_literal_(name, length, "utf8_count")) return sz_kernel_utf8_count_k;
    if (sz_same_literal_(name, length, "utf8_count_strings")) return sz_kernel_utf8_count_strings_k;
    if (sz_same_literal_(name, length, "utf8_seek")) return sz_kernel_utf8_seek_k;
    if (sz_same_literal_(name, length, "utf8_decode")) return sz_kernel_utf8_decode_k;
    if (sz_same_literal_(name, length, "utf8_newlines")) return sz_kernel_utf8_newlines_k;
    if (sz_same_literal_(name, length, "utf8_whitespaces")) return sz_kernel_utf8_whitespaces_k;
    if (sz_same_literal_(name, length, "utf8_delimiters")) return sz_kernel_utf8_delimiters_k;
    if (sz_same_literal_(name, length, "utf8_wordbreaks")) return sz_kernel_utf8_wordbreaks_k;
    if (sz_same_literal_(name, length, "utf8_graphemes")) return sz_kernel_utf8_graphemes_k;
    if (sz_same_literal_(name, length, "utf8_sentences")) return sz_kernel_utf8_sentences_k;
    if (sz_same_literal_(name, length, "utf8_linebreaks")) return sz_kernel_utf8_linebreaks_k;
    if (sz_same_literal_(name, length, "utf8_norm")) return sz_kernel_utf8_norm_k;
    if (sz_same_literal_(name, length, "utf8_find_denormalized")) return sz_kernel_utf8_find_denormalized_k;
    if (sz_same_literal_(name, length, "utf8_uncased_fold")) return sz_kernel_utf8_uncased_fold_k;
    if (sz_same_literal_(name, length, "utf8_uncased_needle_init")) return sz_kernel_utf8_uncased_needle_init_k;
    if (sz_same_literal_(name, length, "utf8_uncased_search")) return sz_kernel_utf8_uncased_search_k;
    if (sz_same_literal_(name, length, "utf8_uncased_order")) return sz_kernel_utf8_uncased_order_k;
    if (sz_same_literal_(name, length, "utf8_find_cased")) return sz_kernel_utf8_find_cased_k;
    if (sz_same_literal_(name, length, "levenshtein_engine_init")) return sz_kernel_levenshtein_engine_init_k;
    if (sz_same_literal_(name, length, "levenshtein_distances")) return sz_kernel_levenshtein_distances_k;
    if (sz_same_literal_(name, length, "ngrams_engine_init")) return sz_kernel_ngrams_engine_init_k;
    if (sz_same_literal_(name, length, "ngrams_overlaps")) return sz_kernel_ngrams_overlaps_k;
    if (sz_same_literal_(name, length, "ngrams_sketches")) return sz_kernel_ngrams_sketches_k;
    if (sz_same_literal_(name, length, "substrings_engine_init")) return sz_kernel_substrings_engine_init_k;
    if (sz_same_literal_(name, length, "substrings_contains")) return sz_kernel_substrings_contains_k;
    if (sz_same_literal_(name, length, "substrings_counts")) return sz_kernel_substrings_counts_k;
    if (sz_same_literal_(name, length, "substrings_find")) return sz_kernel_substrings_find_k;
    if (sz_same_literal_(name, length, "substrings_replace")) return sz_kernel_substrings_replace_k;
    if (sz_same_literal_(name, length, "substrings_bm25_scores")) return sz_kernel_substrings_bm25_scores_k;
    if (sz_same_literal_(name, length, "regex_engine_init")) return sz_kernel_regex_engine_init_k;
    if (sz_same_literal_(name, length, "regex_contains")) return sz_kernel_regex_contains_k;
    if (sz_same_literal_(name, length, "regex_counts")) return sz_kernel_regex_counts_k;
    if (sz_same_literal_(name, length, "regex_find")) return sz_kernel_regex_find_k;
    if (sz_same_literal_(name, length, "regex_replace")) return sz_kernel_regex_replace_k;
    if (sz_same_literal_(name, length, "regex_extract")) return sz_kernel_regex_extract_k;
    if (sz_same_literal_(name, length, "subdocuments_engine_init")) return sz_kernel_subdocuments_engine_init_k;
    if (sz_same_literal_(name, length, "subdocuments_contains")) return sz_kernel_subdocuments_contains_k;
    if (sz_same_literal_(name, length, "subdocuments_counts")) return sz_kernel_subdocuments_counts_k;
    if (sz_same_literal_(name, length, "subdocuments_find")) return sz_kernel_subdocuments_find_k;
    return sz_kernel_unknown_k;
}

/** Signature of the @c sz_hash_best kernels. */
typedef sz_status_t (*sz_kernel_hash_t)(sz_cptr_t, sz_size_t, sz_u64_t, sz_u64_t *, sz_stream_t);

/** Signature of the @c sz_hash_multiseed_best kernels. */
typedef sz_status_t (*sz_kernel_hash_multiseed_t)(sz_cptr_t, sz_size_t, sz_u64_t const *, sz_size_t, sz_u64_t *,
                                                  sz_stream_t);

/** Signature of the @c sz_hash_strings_best kernels. */
typedef sz_status_t (*sz_kernel_hash_strings_t)(sz_strings_view_t const *, sz_size_t const *, sz_size_t, sz_u64_t,
                                                sz_u64_t *, sz_stream_t);

/** Signature of the @c sz_hash_state_init_best kernels. */
typedef sz_status_t (*sz_kernel_hash_state_init_t)(struct sz_hash_state_t *, sz_u64_t, sz_stream_t);

/** Signature of the @c sz_hash_state_update_best kernels. */
typedef sz_status_t (*sz_kernel_hash_state_update_t)(struct sz_hash_state_t *, sz_cptr_t, sz_size_t, sz_stream_t);

/** Signature of the @c sz_hash_state_digest_best kernels. */
typedef sz_status_t (*sz_kernel_hash_state_digest_t)(struct sz_hash_state_t const *, sz_u64_t *, sz_stream_t);

/** Signature of the @c sz_bytesum_best kernels. */
typedef sz_status_t (*sz_kernel_bytesum_t)(sz_cptr_t, sz_size_t, sz_u64_t *, sz_stream_t);

/** Signature of the kernels behind @c sz_utf8_count_best. */
typedef sz_status_t (*sz_kernel_utf8_count_t)(sz_cptr_t, sz_size_t, sz_size_t *, sz_stream_t);

/** Signature of the kernels behind @c sz_utf8_count_strings_best. */
typedef sz_status_t (*sz_kernel_utf8_count_strings_t)(sz_strings_view_t const *, sz_size_t const *, sz_size_t,
                                                      sz_size_t *, sz_size_t *, sz_stream_t);

/** Signature of the kernels behind @c sz_utf8_seek_best. */
typedef sz_status_t (*sz_kernel_utf8_seek_t)(sz_cptr_t, sz_size_t, sz_size_t, sz_cptr_t *, sz_stream_t);

/** Signature of the kernels behind @c sz_utf8_decode_best. */
typedef sz_status_t (*sz_kernel_utf8_decode_t)(sz_cptr_t, sz_size_t, sz_rune_t *, sz_size_t, sz_size_t *, sz_size_t *,
                                               sz_stream_t);

/** Signature of the kernels behind @c sz_utf8_uncased_fold_best. */
typedef sz_status_t (*sz_kernel_utf8_uncased_fold_t)(sz_cptr_t, sz_size_t, sz_ptr_t, sz_size_t *, sz_stream_t);

/** Signature of the kernels behind @c sz_utf8_norm_best. */
typedef sz_status_t (*sz_kernel_utf8_norm_t)(sz_cptr_t, sz_size_t, sz_normal_form_t, sz_ptr_t, sz_size_t *,
                                             sz_stream_t);

/** Signature of the kernels behind @c sz_utf8_find_denormalized_best. */
typedef sz_status_t (*sz_kernel_utf8_find_denormalized_t)(sz_cptr_t, sz_size_t, sz_normal_form_t, sz_cptr_t *,
                                                          sz_stream_t);

/** Forward declaration of the prepared uncased needle. */
struct sz_utf8_uncased_needle_t;

/** Signature of the kernels behind @c sz_utf8_uncased_needle_init_best. */
typedef sz_status_t (*sz_kernel_utf8_uncased_needle_init_t)(sz_cptr_t, sz_size_t, struct sz_utf8_uncased_needle_t *,
                                                            sz_stream_t);

/** Signature of the kernels behind @c sz_utf8_uncased_search_best. */
typedef sz_status_t (*sz_kernel_utf8_uncased_search_t)(sz_cptr_t, sz_size_t, struct sz_utf8_uncased_needle_t const *,
                                                       sz_cptr_t *, sz_size_t *, sz_stream_t);

/** Signature of the kernels behind @c sz_utf8_uncased_order_best. */
typedef sz_status_t (*sz_kernel_utf8_uncased_order_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t, sz_ordering_t *,
                                                      sz_stream_t);

/** Signature of the kernels behind @c sz_utf8_find_cased_best. */
typedef sz_status_t (*sz_kernel_utf8_find_cased_t)(sz_cptr_t, sz_size_t, sz_cptr_t *, sz_stream_t);

/** Signature of the tiling segmenters - graphemes, words, sentences, lines. Emits one length per
 *  segment and their count; each segment starts where the previous one ended. */
typedef sz_status_t (*sz_kernel_utf8_segmenter_t)(sz_cptr_t, sz_size_t, sz_size_t *, sz_size_t, sz_size_t *,
                                                  sz_stream_t);

/** Signature of the token kernels - newlines, whitespace, delimiters. Emits parallel (offset,
 *  length) arrays for each match, their count, and a resume @c bytes_consumed. */
typedef sz_status_t (*sz_kernel_utf8_tokenizer_t)(sz_cptr_t, sz_size_t, sz_size_t *, sz_size_t *, sz_size_t,
                                                  sz_size_t *, sz_size_t *, sz_stream_t);

/** Signature of the @c sz_fill_random_best kernels. */
typedef sz_status_t (*sz_kernel_fill_random_t)(sz_ptr_t, sz_size_t, sz_u64_t, sz_stream_t);

/** Signature of the @c sz_sha256_state_init_best kernels. */
typedef sz_status_t (*sz_kernel_sha256_state_init_t)(struct sz_sha256_state_t *, sz_stream_t);

/** Signature of the @c sz_sha256_state_update_best kernels. */
typedef sz_status_t (*sz_kernel_sha256_state_update_t)(struct sz_sha256_state_t *, sz_cptr_t, sz_size_t, sz_stream_t);

/** Signature of the @c sz_sha256_state_digest_best kernels. */
typedef sz_status_t (*sz_kernel_sha256_state_digest_t)(struct sz_sha256_state_t const *, sz_u8_t *, sz_stream_t);

/** Signature of the @c sz_sha256_multistate_update_best kernels. */
typedef sz_status_t (*sz_kernel_sha256_multistate_update_t)(struct sz_sha256_state_t *, sz_strings_view_t const *,
                                                            sz_stream_t);

/** Signature of the @c sz_sha256_multistate_digest_best kernels. */
typedef sz_status_t (*sz_kernel_sha256_multistate_digest_t)(struct sz_sha256_state_t const *, sz_size_t, sz_u8_t *,
                                                            sz_stream_t);

/** Signature of the @c sz_aes256_key_init_best kernels. */
typedef sz_status_t (*sz_kernel_aes256_key_init_t)(struct sz_aes256_key_t *, sz_u8_t const *, sz_stream_t);

/** Signature of the @c sz_aes256_gcm_key_init_best kernels. */
typedef sz_status_t (*sz_kernel_aes256_gcm_key_init_t)(struct sz_aes256_gcm_key_t *, sz_u8_t const *, sz_stream_t);

/** Signature of the @c sz_aes256_ctr_xor_best kernels. */
typedef sz_status_t (*sz_kernel_aes256_ctr_xor_t)(struct sz_aes256_key_t const *, sz_u8_t const *, sz_u64_t, sz_cptr_t,
                                                  sz_size_t, sz_ptr_t, sz_stream_t);

/** Signature of the @c sz_aes256_gcm_encrypt_best kernels. */
typedef sz_status_t (*sz_kernel_aes256_gcm_encrypt_t)(struct sz_aes256_gcm_key_t const *, sz_u8_t const *, sz_cptr_t,
                                                      sz_size_t, sz_cptr_t, sz_size_t, sz_ptr_t, sz_u8_t *,
                                                      sz_stream_t);

/** Signature of the @c sz_aes256_gcm_decrypt_best kernels. */
typedef sz_status_t (*sz_kernel_aes256_gcm_decrypt_t)(struct sz_aes256_gcm_key_t const *, sz_u8_t const *, sz_cptr_t,
                                                      sz_size_t, sz_cptr_t, sz_size_t, sz_ptr_t, sz_u8_t const *,
                                                      sz_stream_t);

/** Signature of the @c sz_aes256_gcm_encryptor_init_best kernels. */
typedef sz_status_t (*sz_kernel_aes256_gcm_encryptor_init_t)(struct sz_aes256_gcm_encryptor_t *,
                                                             struct sz_aes256_gcm_key_t const *, sz_u8_t const *,
                                                             sz_stream_t);

/** Signature of the @c sz_aes256_gcm_encryptor_associate_best kernels. */
typedef sz_status_t (*sz_kernel_aes256_gcm_encryptor_associate_t)(struct sz_aes256_gcm_encryptor_t *, sz_cptr_t,
                                                                  sz_size_t, sz_stream_t);

/** Signature of the @c sz_aes256_gcm_encryptor_update_best kernels. */
typedef sz_status_t (*sz_kernel_aes256_gcm_encryptor_update_t)(struct sz_aes256_gcm_encryptor_t *, sz_cptr_t, sz_size_t,
                                                               sz_ptr_t, sz_stream_t);

/** Signature of the @c sz_aes256_gcm_encryptor_digest_best kernels. */
typedef sz_status_t (*sz_kernel_aes256_gcm_encryptor_digest_t)(struct sz_aes256_gcm_encryptor_t const *, sz_u8_t *,
                                                               sz_stream_t);

/** Signature of the @c sz_aes256_gcm_decryptor_init_best kernels. */
typedef sz_status_t (*sz_kernel_aes256_gcm_decryptor_init_t)(struct sz_aes256_gcm_decryptor_t *,
                                                             struct sz_aes256_gcm_key_t const *, sz_u8_t const *,
                                                             sz_stream_t);

/** Signature of the @c sz_aes256_gcm_decryptor_associate_best kernels. */
typedef sz_status_t (*sz_kernel_aes256_gcm_decryptor_associate_t)(struct sz_aes256_gcm_decryptor_t *, sz_cptr_t,
                                                                  sz_size_t, sz_stream_t);

/** Signature of the @c sz_aes256_gcm_decryptor_update_unverified_best kernels. */
typedef sz_status_t (*sz_kernel_aes256_gcm_decryptor_update_unverified_t)(struct sz_aes256_gcm_decryptor_t *, sz_cptr_t,
                                                                          sz_size_t, sz_ptr_t, sz_stream_t);

/** Signature of the @c sz_aes256_gcm_decryptor_verify_best kernels. */
typedef sz_status_t (*sz_kernel_aes256_gcm_decryptor_verify_t)(struct sz_aes256_gcm_decryptor_t const *,
                                                               sz_u8_t const *, sz_stream_t);

/** Signature of the @c sz_equal_best kernels. */
typedef sz_status_t (*sz_kernel_equal_t)(sz_cptr_t, sz_cptr_t, sz_size_t, sz_bool_t *, sz_stream_t);

/** Signature of the @c sz_order_best kernels. */
typedef sz_status_t (*sz_kernel_order_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t, sz_ordering_t *, sz_stream_t);

/** Signature of the @c sz_strings_to_tape_best kernels. */
typedef sz_status_t (*sz_kernel_strings_to_tape_t)(sz_strings_view_t const *, sz_size_t const *, sz_size_t,
                                                   sz_allocator_t *, sz_strings_span_t *, sz_stream_t);

/** Signature of the @c sz_strings_to_rows_best kernels. */
typedef sz_status_t (*sz_kernel_strings_to_rows_t)(sz_strings_view_t const *, sz_size_t const *, sz_size_t,
                                                   sz_strings_rows_t const *, sz_size_t *, sz_stream_t);

/** Signature of the @c sz_strings_lookup_best kernels. */
typedef sz_status_t (*sz_kernel_strings_lookup_t)(sz_strings_view_t const *, sz_size_t const *, sz_size_t, sz_cptr_t,
                                                  sz_allocator_t *, sz_strings_span_t *, sz_stream_t);

/** Signature of the @c sz_strings_normalize_best kernels. */
typedef sz_status_t (*sz_kernel_strings_normalize_t)(sz_strings_view_t const *, sz_size_t const *, sz_size_t,
                                                     sz_normal_form_t, sz_allocator_t *, sz_strings_span_t *,
                                                     sz_stream_t);

/** Signature of the @c sz_strings_escape_best kernels. */
typedef sz_status_t (*sz_kernel_strings_escape_t)(sz_strings_view_t const *, sz_size_t const *, sz_size_t,
                                                  sz_escape_kind_t, sz_allocator_t *, sz_strings_span_t *, sz_stream_t);

/** Signature of the @c sz_strings_unescape_best kernels. */
typedef sz_status_t (*sz_kernel_strings_unescape_t)(sz_strings_view_t const *, sz_size_t const *, sz_size_t,
                                                    sz_escape_kind_t, sz_allocator_t *, sz_strings_span_t *,
                                                    sz_size_t *, sz_stream_t);

/** Signature of the @c sz_strings_split_best kernels. */
typedef sz_status_t (*sz_kernel_strings_split_t)(sz_cptr_t, sz_size_t, sz_byteset_t const *, sz_split_kind_t,
                                                 sz_allocator_t *, sz_strings_span_t *, sz_stream_t);

/** Signature of the @c sz_strings_split_lines_best kernels. */
typedef sz_status_t (*sz_kernel_strings_split_lines_t)(sz_cptr_t, sz_size_t, sz_allocator_t *, sz_strings_span_t *,
                                                       sz_stream_t);

/** Signature of the @c sz_strings_join_best kernels. */
typedef sz_status_t (*sz_kernel_strings_join_t)(sz_strings_view_t const *, sz_size_t, sz_size_t,
                                                sz_string_view_t const *, sz_string_view_t, sz_ptr_t, sz_size_t *,
                                                sz_stream_t);

/** Signature of the @c sz_find_byte_best and @c sz_rfind_byte_best kernels. */
typedef sz_status_t (*sz_kernel_find_byte_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_cptr_t *, sz_stream_t);

/** Signature of the @c sz_find_best and @c sz_rfind_best kernels. */
typedef sz_status_t (*sz_kernel_find_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t, sz_cptr_t *, sz_stream_t);

/** Signature of the @c sz_find_byteset_best and @c sz_rfind_byteset_best kernels. */
typedef sz_status_t (*sz_kernel_find_byteset_t)(sz_cptr_t, sz_size_t, sz_byteset_t const *, sz_cptr_t *, sz_stream_t);

/** Signature of the @c sz_find_strings_best kernels. */
typedef sz_status_t (*sz_kernel_find_strings_t)(sz_strings_view_t const *, sz_size_t const *, sz_size_t, sz_cptr_t,
                                                sz_size_t, sz_size_t *, sz_stream_t);

/** Signature of the @c sz_sort_order_best and @c sz_sort_uncased_order_best kernels. */
typedef sz_status_t (*sz_kernel_sort_order_t)(sz_strings_view_t const *, sz_size_t, sz_bool_t, sz_allocator_t *,
                                              sz_size_t *, sz_stream_t);

/** Signature of the benchmark-only @c sz_sort_pgrams_serial_ integer sort helper and its tiers. */
typedef sz_status_t (*sz_sort_pgrams_t_)(sz_pgram_t *, sz_size_t, sz_allocator_t *, sz_size_t *);

/** Signature of the @c sz_intersect_best kernels. */
typedef sz_status_t (*sz_kernel_intersect_t)(sz_strings_view_t const *, sz_strings_view_t const *, sz_allocator_t *,
                                             sz_u64_t, sz_size_t *, sz_size_t *, sz_size_t *, sz_stream_t);

/** Signature of every @c sz_levenshtein_engine_init kernel. */
typedef sz_status_t (*sz_kernel_levenshtein_engine_init_t)(struct sz_levenshtein_engine_t *, sz_strings_view_t const *,
                                                           sz_levenshtein_symbol_t, sz_allocator_t *, sz_stream_t);

/** Signature of every @c sz_levenshtein_distances kernel, at either alphabet. */
typedef sz_status_t (*sz_kernel_levenshtein_distances_t)(struct sz_levenshtein_engine_t *, sz_strings_view_t const *,
                                                         sz_size_t const *, sz_size_t const *, sz_size_t, sz_size_t *,
                                                         sz_size_t, sz_stream_t);

/** Signature of every @c sz_ngrams_engine_init kernel. */
typedef sz_status_t (*sz_kernel_ngrams_engine_init_t)(struct sz_ngrams_engine_t *, sz_strings_view_t const *,
                                                      sz_size_t const *, sz_size_t, sz_size_t, sz_allocator_t *,
                                                      sz_stream_t);

/** Signature of every @c sz_ngrams_overlaps kernel. */
typedef sz_status_t (*sz_kernel_ngrams_overlaps_t)(struct sz_ngrams_engine_t *, sz_strings_view_t const *, sz_f32_t,
                                                   sz_f32_t *, sz_size_t, sz_size_t, sz_stream_t);

/** Signature of every @c sz_ngrams_sketches kernel. */
typedef sz_status_t (*sz_kernel_ngrams_sketches_t)(struct sz_ngrams_engine_t *, sz_strings_view_t const *, sz_size_t,
                                                   sz_u8_t *, sz_size_t, sz_stream_t);

/** Signature of every @c sz_substrings_engine_init kernel. */
typedef sz_status_t (*sz_kernel_substrings_engine_init_t)(struct sz_substrings_engine_t *, sz_strings_view_t const *,
                                                          sz_case_sensitivity_t, sz_substrings_overlap_policy_t,
                                                          sz_size_t, sz_size_t, sz_size_t, sz_allocator_t *,
                                                          sz_stream_t);

/** Signature of every @c sz_substrings_contains kernel. */
typedef sz_status_t (*sz_kernel_substrings_contains_t)(struct sz_substrings_engine_t *, sz_strings_view_t const *,
                                                       sz_u1x8_t *, sz_stream_t);

/** Signature of every @c sz_substrings_counts kernel. */
typedef sz_status_t (*sz_kernel_substrings_counts_t)(struct sz_substrings_engine_t *, sz_strings_view_t const *,
                                                     sz_size_t *, sz_size_t, sz_stream_t);

/** Signature of every @c sz_substrings_find kernel. */
typedef sz_status_t (*sz_kernel_substrings_find_t)(struct sz_substrings_engine_t *, sz_strings_view_t const *,
                                                   sz_match_t *, sz_size_t, sz_size_t *, sz_stream_t);

/** Signature of every @c sz_substrings_replace kernel. */
typedef sz_status_t (*sz_kernel_substrings_replace_t)(struct sz_substrings_engine_t *, sz_strings_view_t const *,
                                                      sz_strings_view_t const *, sz_ptr_t, sz_size_t, sz_size_t *,
                                                      sz_stream_t);

/** Signature of every @c sz_substrings_bm25_scores kernel. */
typedef sz_status_t (*sz_kernel_substrings_bm25_scores_t)(struct sz_substrings_engine_t *, sz_strings_view_t const *,
                                                          sz_f32_t const *, struct sz_substrings_bm25_t const *,
                                                          sz_f32_t const *, sz_f32_t *, sz_size_t, sz_stream_t);

/** Signature of every @c sz_regex_engine_init kernel. */
typedef sz_status_t (*sz_kernel_regex_engine_init_t)(struct sz_regex_engine_t *, sz_strings_view_t const *,
                                                     sz_case_sensitivity_t, sz_size_t *, sz_allocator_t *, sz_stream_t);

/** Signature of every @c sz_regex_contains kernel. */
typedef sz_status_t (*sz_kernel_regex_contains_t)(struct sz_regex_engine_t *, sz_strings_view_t const *, sz_u1x8_t *,
                                                  sz_stream_t);

/** Signature of every @c sz_regex_counts kernel. */
typedef sz_status_t (*sz_kernel_regex_counts_t)(struct sz_regex_engine_t *, sz_strings_view_t const *, sz_size_t *,
                                                sz_size_t, sz_stream_t);

/** Signature of every @c sz_regex_find kernel. */
typedef sz_status_t (*sz_kernel_regex_find_t)(struct sz_regex_engine_t *, sz_strings_view_t const *, sz_match_t *,
                                              sz_size_t, sz_size_t *, sz_stream_t);

/** Signature of every @c sz_regex_replace kernel. */
typedef sz_status_t (*sz_kernel_regex_replace_t)(struct sz_regex_engine_t *, sz_strings_view_t const *,
                                                 sz_strings_view_t const *, sz_ptr_t, sz_size_t, sz_size_t *,
                                                 sz_stream_t);

/** Signature of every @c sz_regex_extract kernel. */
typedef sz_status_t (*sz_kernel_regex_extract_t)(struct sz_regex_engine_t *, sz_strings_view_t const *, sz_size_t,
                                                 sz_match_t *, sz_size_t, sz_size_t *, sz_stream_t);

/** Signature of every @c sz_subdocuments_engine_init kernel. */
typedef sz_status_t (*sz_kernel_subdocuments_engine_init_t)(struct sz_subdocuments_engine_t *,
                                                            sz_strings_view_t const *, sz_subdocuments_format_t,
                                                            sz_subdocuments_options_t, sz_size_t *, sz_allocator_t *,
                                                            sz_stream_t);

/** Signature of every @c sz_subdocuments_contains kernel. */
typedef sz_status_t (*sz_kernel_subdocuments_contains_t)(struct sz_subdocuments_engine_t *, sz_strings_view_t const *,
                                                         sz_u1x8_t *, sz_size_t *, sz_stream_t);

/** Signature of every @c sz_subdocuments_counts kernel. */
typedef sz_status_t (*sz_kernel_subdocuments_counts_t)(struct sz_subdocuments_engine_t *, sz_strings_view_t const *,
                                                       sz_size_t *, sz_size_t, sz_size_t *, sz_stream_t);

/** Signature of every @c sz_subdocuments_find kernel. */
typedef sz_status_t (*sz_kernel_subdocuments_find_t)(struct sz_subdocuments_engine_t *, sz_strings_view_t const *,
                                                     sz_match_t *, sz_size_t, sz_size_t *, sz_size_t *, sz_stream_t);

/** Any kernel, cast back to its signature before the call. */
typedef void (*sz_kernel_punned_t)(void);

/*  CPU capabilities, reported along two independent axes and the mask dispatch uses by default:
 *
 *  - @b sz_capabilities_detected_cpu() — what this CPU can execute, from CPUID or HWCAP, or where
 *    the OS cannot be asked, what the compiler's own flags guarantee.
 *  - @b sz_capabilities_compiled_cpu() — what this binary contains, as the build set the
 *    `STRINGZILLA_TARGET_*` macros.
 *  - @b sz_capabilities_enabled_cpu() — both axes at once: the mask to dispatch on the CPU with.
 *    Always retains @b sz_cap_serial_k.
 *
 *  The two axes are independent, and conflating them is a silent performance cliff rather than a
 *  build error: a binary whose ISA probes failed still reports this machine's full @b detected
 *  mask while containing no SIMD kernels at all. Ask for @b enabled() unless you specifically mean
 *  one of the raw axes. The library's lookups clamp every CPU mask to @b enabled() themselves, so
 *  @c sz_cap_cpus_k dispatches to the same kernels. */

STRINGZILLA_API sz_status_t sz_capabilities_detected_cpu(sz_capability_t *capabilities);
STRINGZILLA_API sz_status_t sz_capabilities_compiled_cpu(sz_capability_t *capabilities);
STRINGZILLA_API sz_status_t sz_capabilities_enabled_cpu(sz_capability_t *capabilities);

/** The capabilities whose kernels the <tt>STRINGZILLA_TARGET_*</tt> macros of the including unit
 *  compile, as a constant for consumers picking kits at compile time; in the library, the mask
 *  @c sz_capabilities_compiled_cpu reports. Says nothing about the current CPU. */
STRINGZILLA_CONSTEXPR sz_capability_t sz_capabilities_compiled_cpu_mask(void) {
    return (sz_capability_t)(                                     //
        (sz_cap_neon_k * STRINGZILLA_TARGET_NEON) |               //
        (sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES) |         //
        (sz_cap_neonsha_k * STRINGZILLA_TARGET_NEONSHA) |         //
        (sz_cap_sve_k * STRINGZILLA_TARGET_SVE) |                 //
        (sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2) |               //
        (sz_cap_sve2aes_k * STRINGZILLA_TARGET_SVE2AES) |         //
        (sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE) |       //
        (sz_cap_goldmont_k * STRINGZILLA_TARGET_GOLDMONT) |       //
        (sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL) |         //
        (sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE) |         //
        (sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE) |         //
        (sz_cap_v128_k * STRINGZILLA_TARGET_V128) |               //
        (sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED) | //
        (sz_cap_rvv_k * STRINGZILLA_TARGET_RVV) |                 //
        (sz_cap_rvvcrypto_k * STRINGZILLA_TARGET_RVVCRYPTO) |     //
        (sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX) | //
        (sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) |       //
        (sz_cap_serial_k));
}

/**
 *  @brief Prepares the calling thread for the kernels of @p capabilities, and only those.
 *  @param[in] capabilities The capabilities to prepare for, like @c sz_capabilities_enabled_cpu.
 *
 *  Call it once on each thread that dispatches. Every current capability needs nothing, so today
 *  it returns right away.
 */
STRINGZILLA_API sz_status_t sz_configure_thread_cpu(sz_capability_t capabilities);

/**
 *  @brief Writes @p capabilities as a comma-separated name list such as "serial,haswell,skylake".
 *  @param[in] capabilities The mask to name, like @c sz_capabilities_detected_cpu reports.
 *  @param[out] buffer Destination, always null-terminated; the list is truncated to fit.
 *  @param[in] capacity Size of @p buffer, like @c STRINGZILLA_CAPABILITIES_NAME_CAPACITY; a zero
 *      capacity writes nothing.
 *  @return Bytes written, excluding the null terminator.
 */
STRINGZILLA_API sz_size_t sz_capabilities_name(sz_capability_t capabilities, char *buffer, sz_size_t capacity);

/**
 *  @brief Finds the kernel of @p kind that its dispatch point would run for @p capabilities, to
 *      resolve it once and call it many times.
 *  @param[in] kind What functionality, like @c sz_kernel_find_k.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[out] kernel The kernel, cast back to its @c sz_kernel_find_t or sibling before the call,
 *      or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *
 *  @par Statuses:
 *  - @c sz_missing_kernel_k No capability in @p capabilities has it.
 *  - @c sz_missing_library_k Header-only builds.
 */
STRINGZILLA_API sz_status_t sz_find_kernel_punned(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability);

/**
 *  @brief Waits for everything enqueued on @p stream, after which what its rounds wrote is readable
 *      from the host.
 *  @param[in] capabilities One device's capabilities; its group picks the runtime that waits.
 *  @param[in] stream Null on the CPU, which has nothing to wait for. On a GPU, the stream to join,
 *      which also names the device, or null for the default stream of the default device.
 *
 *  @par Statuses:
 *  - @c sz_device_memory_mismatch_k On a device, a @p stream it cannot use.
 *  - @c sz_missing_gpu_k A vendor this build lacks.
 */
STRINGZILLA_API sz_status_t sz_stream_synchronize_best(sz_capability_t capabilities, sz_stream_t stream);

/** @copydoc sz_stream_synchronize_best */
STRINGZILLA_API sz_status_t sz_stream_synchronize_serial(sz_stream_t stream);

/** @copydoc sz_stream_synchronize_best */
STRINGZILLA_API sz_status_t sz_stream_synchronize_cuda(sz_stream_t stream);

/** @copydoc sz_stream_synchronize_best */
STRINGZILLA_API sz_status_t sz_stream_synchronize_rocm(sz_stream_t stream);

/** @copydoc sz_stream_synchronize_best */
STRINGZILLA_API sz_status_t sz_stream_synchronize_metal(sz_stream_t stream);

/*  GPU capabilities, per vendor, of one device named by its runtime's own ordinal: the index that
 *  @c cudaSetDevice or @c hipSetDevice takes, or the position in Metal's device list. Each vendor
 *  has four queries:
 *
 *  - @b sz_count_devices_cuda() — how many devices the runtime sees, zero without its kernels.
 *  - @b sz_capabilities_detected_cuda() — what one device runs.
 *  - @b sz_capabilities_compiled_cuda() — what this binary holds code for that one device runs.
 *  - @b sz_capabilities_enabled_cuda() — both at once: the mask to dispatch on that device with.
 *
 *  ROCm and Metal have the same four. Nothing is cached, as the runtimes answer from their own
 *  state: ask once per device and keep the mask. Each vendor also makes a stream on one device
 *  with @b sz_stream_init_cuda() and frees it with @b sz_stream_free_cuda(), for a caller without
 *  the vendor's runtime at hand. These are the only functions an ordinal reaches: everything
 *  consuming a mask takes a stream instead, which names its device.
 *
 *  A CUDA or ROCm unit holds code for the architectures CMake compiled it for, and a device runs a
 *  capability only where its runtime loads that unit's kernels: @b sz_capabilities_compiled_cuda()
 *  asks @c cudaFuncGetAttributes about one kernel of each unit, and the ROCm twin asks
 *  @c hipFuncGetAttributes.
 *
 *  Every vendor's queries are declared in every build. A library without a vendor answers them with
 *  @c sz_missing_gpu_k, and a header-only unit with @c sz_missing_library_k, even compiled by nvcc,
 *  hipcc or against Metal, as only the library counts and probes devices. */

/**
 *  @brief Counts the CUDA devices the process sees.
 *  @param[out] count The number of devices, zero without one.
 *
 *  @par Statuses:
 *  - @c sz_missing_gpu_k Without one.
 */
STRINGZILLA_API sz_status_t sz_count_devices_cuda(sz_size_t *count);

/**
 *  @brief Reports the capabilities CUDA device @p ordinal runs.
 *  @param[in] ordinal The CUDA runtime's device index, like @c cudaSetDevice takes.
 *  @param[out] capabilities The CUDA baseline and what the device runs, zero on failure.
 *
 *  @par Statuses:
 *  - @c sz_missing_gpu_k Past the last device.
 */
STRINGZILLA_API sz_status_t sz_capabilities_detected_cuda(sz_size_t ordinal, sz_capability_t *capabilities);

/**
 *  @brief Reports the CUDA capabilities this binary holds code for that device @p ordinal runs.
 *  @param[in] ordinal The CUDA runtime's device index, like @c cudaSetDevice takes.
 *  @param[out] capabilities The CUDA capabilities compiled in for that device, zero on failure.
 *
 *  @par Statuses:
 *  - @c sz_missing_gpu_k Past the last device.
 */
STRINGZILLA_API sz_status_t sz_capabilities_compiled_cuda(sz_size_t ordinal, sz_capability_t *capabilities);

/**
 *  @brief Reports the capabilities CUDA device @p ordinal runs that this binary holds kernels for.
 *  @param[in] ordinal The CUDA runtime's device index, like @c cudaSetDevice takes.
 *  @param[out] capabilities The CUDA baseline and what the device runs, zero on failure.
 *
 *  @par Statuses:
 *  - @c sz_missing_gpu_k Past the last device.
 */
STRINGZILLA_API sz_status_t sz_capabilities_enabled_cuda(sz_size_t ordinal, sz_capability_t *capabilities);

/** Creates a stream on CUDA device @p ordinal with @c cudaStreamCreate. */
STRINGZILLA_API sz_status_t sz_stream_init_cuda(sz_size_t ordinal, sz_stream_t *stream);

/** Destroys a stream of @ref sz_stream_init_cuda with @c cudaStreamDestroy, once its work ends. */
STRINGZILLA_API sz_status_t sz_stream_free_cuda(sz_stream_t stream);

/**
 *  @brief Counts the ROCm devices the process sees.
 *  @param[out] count The number of devices, zero without one.
 *
 *  @par Statuses:
 *  - @c sz_missing_gpu_k Without one.
 */
STRINGZILLA_API sz_status_t sz_count_devices_rocm(sz_size_t *count);

/**
 *  @brief Reports the capabilities ROCm device @p ordinal runs.
 *  @param[in] ordinal The HIP runtime's device index, like @c hipSetDevice takes.
 *  @param[out] capabilities The ROCm baseline and what the device runs, zero on failure.
 *
 *  @par Statuses:
 *  - @c sz_missing_gpu_k Past the last device.
 */
STRINGZILLA_API sz_status_t sz_capabilities_detected_rocm(sz_size_t ordinal, sz_capability_t *capabilities);

/**
 *  @brief Reports the ROCm capabilities this binary holds code for that device @p ordinal runs.
 *  @param[in] ordinal The HIP runtime's device index, like @c hipSetDevice takes.
 *  @param[out] capabilities The ROCm capabilities compiled in for that device, zero on failure.
 *
 *  @par Statuses:
 *  - @c sz_missing_gpu_k Past the last device.
 */
STRINGZILLA_API sz_status_t sz_capabilities_compiled_rocm(sz_size_t ordinal, sz_capability_t *capabilities);

/**
 *  @brief Reports the capabilities ROCm device @p ordinal runs that this binary holds kernels for.
 *  @param[in] ordinal The HIP runtime's device index, like @c hipSetDevice takes.
 *  @param[out] capabilities The ROCm baseline and what the device runs, zero on failure.
 *
 *  @par Statuses:
 *  - @c sz_missing_gpu_k Past the last device.
 */
STRINGZILLA_API sz_status_t sz_capabilities_enabled_rocm(sz_size_t ordinal, sz_capability_t *capabilities);

/** Creates a stream on ROCm device @p ordinal with @c hipStreamCreate. */
STRINGZILLA_API sz_status_t sz_stream_init_rocm(sz_size_t ordinal, sz_stream_t *stream);

/** Destroys a stream of @ref sz_stream_init_rocm with @c hipStreamDestroy, once its work ends. */
STRINGZILLA_API sz_status_t sz_stream_free_rocm(sz_stream_t stream);

/**
 *  @brief Counts the Metal devices the system lists.
 *  @param[out] count The number of devices, zero without one.
 *
 *  @par Statuses:
 *  - @c sz_missing_gpu_k Without one.
 */
STRINGZILLA_API sz_status_t sz_count_devices_metal(sz_size_t *count);

/**
 *  @brief Reports the capabilities Metal device @p ordinal runs.
 *  @param[in] ordinal The device's position in the order the system lists them.
 *  @param[out] capabilities The Metal baseline when the device supports it, zero otherwise.
 *
 *  @par Statuses:
 *  - @c sz_missing_gpu_k Past the last device.
 */
STRINGZILLA_API sz_status_t sz_capabilities_detected_metal(sz_size_t ordinal, sz_capability_t *capabilities);

/**
 *  @brief Reports the Metal capabilities this binary holds kernels for, which device @p ordinal
 *      compiles from the embedded source.
 *  @param[in] ordinal The device's position in the order the system lists them.
 *  @param[out] capabilities The Metal capabilities compiled in, zero on failure.
 *
 *  @par Statuses:
 *  - @c sz_missing_gpu_k Past the last device.
 */
STRINGZILLA_API sz_status_t sz_capabilities_compiled_metal(sz_size_t ordinal, sz_capability_t *capabilities);

/**
 *  @brief Reports the capabilities Metal device @p ordinal runs that this binary holds kernels for.
 *  @param[in] ordinal The device's position in the order the system lists them.
 *  @param[out] capabilities The Metal baseline when the device supports it, zero otherwise.
 *
 *  @par Statuses:
 *  - @c sz_missing_gpu_k Past the last device.
 */
STRINGZILLA_API sz_status_t sz_capabilities_enabled_metal(sz_size_t ordinal, sz_capability_t *capabilities);

/** Opens an @c id<MTLCommandQueue> stream on Metal device @p ordinal with @c newCommandQueue. */
STRINGZILLA_API sz_status_t sz_stream_init_metal(sz_size_t ordinal, sz_stream_t *stream);

/** Waits for a queue of @ref sz_stream_init_metal, then releases it. */
STRINGZILLA_API sz_status_t sz_stream_free_metal(sz_stream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif

/*  Header-only units define every export here, after the declarations and outside `extern "C"`, as
 *  the runtimes the vendor headers include declare templates. The CPU is asked on every call, and
 *  every GPU query answers `sz_missing_library_k`, as only the library probes devices. */
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/cpu.h"    // `sz_capabilities_detected_cpu_`
#include "stringzilla/cuda.cuh" // CUDA device primitives, for the header-only kernels
#include "stringzilla/rocm.cuh" // ROCm device primitives, for the header-only kernels
#include "stringzilla/metal.h"  // Metal runtime calls, for the header-only kernels

#ifdef __cplusplus
extern "C" {
#endif

STRINGZILLA_API sz_status_t sz_find_kernel_punned(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_capabilities_detected_cpu(sz_capability_t *capabilities) {
    *capabilities = sz_capabilities_detected_cpu_();
    return sz_success_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_compiled_cpu(sz_capability_t *capabilities) {
    *capabilities = sz_capabilities_compiled_cpu_mask();
    return sz_success_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_enabled_cpu(sz_capability_t *capabilities) {
    *capabilities = sz_capabilities_detected_cpu_() & sz_capabilities_compiled_cpu_mask();
    return sz_success_k;
}

/** @copydoc sz_configure_thread_cpu */
STRINGZILLA_API sz_status_t sz_configure_thread_cpu(sz_capability_t capabilities) {
    return sz_configure_thread_cpu_(capabilities);
}

/** @copydoc sz_capabilities_name */
STRINGZILLA_API sz_size_t sz_capabilities_name(sz_capability_t capabilities, char *buffer, sz_size_t capacity) {
    return sz_capabilities_name_(capabilities, buffer, capacity);
}

STRINGZILLA_API sz_status_t sz_count_devices_cuda(sz_size_t *count) {
    *count = 0;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_detected_cuda(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_compiled_cuda(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_enabled_cuda(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_stream_init_cuda(sz_size_t ordinal, sz_stream_t *stream) {
    sz_unused_(ordinal);
    *stream = STRINGZILLA_NULL;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_stream_free_cuda(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_stream_synchronize_cuda(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_count_devices_rocm(sz_size_t *count) {
    *count = 0;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_detected_rocm(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_compiled_rocm(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_enabled_rocm(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_stream_init_rocm(sz_size_t ordinal, sz_stream_t *stream) {
    sz_unused_(ordinal);
    *stream = STRINGZILLA_NULL;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_stream_free_rocm(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_stream_synchronize_rocm(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_count_devices_metal(sz_size_t *count) {
    *count = 0;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_detected_metal(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_compiled_metal(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_enabled_metal(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_stream_init_metal(sz_size_t ordinal, sz_stream_t *stream) {
    sz_unused_(ordinal);
    *stream = STRINGZILLA_NULL;
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_stream_free_metal(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_library_k;
}
STRINGZILLA_API sz_status_t sz_stream_synchronize_metal(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_best(sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif // STRINGZILLA_HEADER_ONLY

#endif // STRINGZILLA_CAPABILITIES_H_
