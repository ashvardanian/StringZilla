/**
 *  @file include/stringzilla/capabilities.h
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

/** Inferring target OS: Windows, MacOS, Linux, or FreeBSD. */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__) || defined(__CYGWIN__)
#define STRINGZILLA_OS_WINDOWS_ 1
#else
#define STRINGZILLA_OS_WINDOWS_ 0
#endif
#if defined(__APPLE__) && defined(__MACH__)
#define STRINGZILLA_OS_APPLE_ 1
#else
#define STRINGZILLA_OS_APPLE_ 0
#endif
#if defined(__linux__)
#define STRINGZILLA_OS_LINUX_ 1
#else
#define STRINGZILLA_OS_LINUX_ 0
#endif
#if defined(__FreeBSD__)
#define STRINGZILLA_OS_FREEBSD_ 1
#else
#define STRINGZILLA_OS_FREEBSD_ 0
#endif

/* On Apple Silicon, @c mrs is not allowed in user-space, so we need to use the @c sysctl API */
#if STRINGZILLA_OS_APPLE_
#include <sys/sysctl.h>
#endif

/* On Linux and FreeBSD Arm, the kernel reports through @c AT_HWCAP whether it emulates @c mrs reads
 * of the ID registers, so nothing has to trap to find out. */
#if STRINGZILLA_ARCH_ARM64_ && (STRINGZILLA_OS_LINUX_ || STRINGZILLA_OS_FREEBSD_) && STRINGZILLA_WITH_LIBC
#include <sys/auxv.h> // `getauxval`, `elf_aux_info`, `AT_HWCAP`
#endif

/* On 64-bit RISC-V we probe HWCAP via the auxiliary vector and vector sub-extensions via the
 * Linux @c riscv_hwprobe syscall; FreeBSD lacks it and uses @c elf_aux_info for base RVV only. */
#if STRINGZILLA_ARCH_RISCV64_ && STRINGZILLA_WITH_LIBC
#if STRINGZILLA_OS_LINUX_
#include <sys/auxv.h>    // `getauxval`, `AT_HWCAP`
#include <sys/syscall.h> // `SYS_riscv_hwprobe`
#include <unistd.h>      // `syscall`
#elif STRINGZILLA_OS_FREEBSD_
#include <sys/auxv.h> // `elf_aux_info`, `AT_HWCAP`
#endif
#endif

/* On LoongArch and IBM POWER the SIMD extensions are likewise reported through the aux vector. */
#if (STRINGZILLA_ARCH_LOONGARCH64_ || STRINGZILLA_ARCH_PPC64_) && STRINGZILLA_WITH_LIBC
#if STRINGZILLA_OS_LINUX_
#include <sys/auxv.h> // `getauxval`, `AT_HWCAP`, `AT_HWCAP2`
#elif STRINGZILLA_OS_FREEBSD_
#include <sys/auxv.h> // `elf_aux_info`, `AT_HWCAP`, `AT_HWCAP2`
#endif
#endif

/* On Windows Arm, we use IsProcessorFeaturePresent API for capability detection */
#if STRINGZILLA_OS_WINDOWS_ && STRINGZILLA_ARCH_ARM64_
#define NOMINMAX
#include <windows.h>
#endif

#if STRINGZILLA_WITH_METAL
#include "stringzilla/metal.h" // `sz_metal_list_devices_`, `sz_metal_device_`
#endif

/** Buffer size @c sz_capabilities_name never overruns, including its null terminator. */
#define STRINGZILLA_CAPABILITIES_NAME_CAPACITY 256

#ifdef __cplusplus
extern "C" {
#endif

/** 64-bit bitmask of the capabilities of one device. */
typedef sz_u64_t sz_capability_t;

/** Serial (non-SIMD) fallback capability. Always available. */
#define sz_cap_serial_k ((sz_capability_t)1)

/** Mask representing any capability. */
#define sz_cap_any_k (~(sz_capability_t)0)

/** CPU capabilities, each architecture's in a contiguous run of bits ascending by preference, so
 *  the highest bit a mask shares with a verb's kernels names the one to run. */
#define sz_cap_westmere_k ((sz_capability_t)1 << 1)
#define sz_cap_goldmont_k ((sz_capability_t)1 << 2)
#define sz_cap_haswell_k ((sz_capability_t)1 << 3)
#define sz_cap_skylake_k ((sz_capability_t)1 << 4)
#define sz_cap_icelake_k ((sz_capability_t)1 << 5)
#define sz_cap_neon_k ((sz_capability_t)1 << 6)
#define sz_cap_neonaes_k ((sz_capability_t)1 << 7)
#define sz_cap_neonsha_k ((sz_capability_t)1 << 8)
#define sz_cap_sve_k ((sz_capability_t)1 << 9)
#define sz_cap_sve2_k ((sz_capability_t)1 << 10)
#define sz_cap_sve2aes_k ((sz_capability_t)1 << 11)
#define sz_cap_rvv_k ((sz_capability_t)1 << 12)
#define sz_cap_rvvcrypto_k ((sz_capability_t)1 << 13)
#define sz_cap_v128_k ((sz_capability_t)1 << 14)
#define sz_cap_v128relaxed_k ((sz_capability_t)1 << 15)
#define sz_cap_loongsonasx_k ((sz_capability_t)1 << 16)
#define sz_cap_powervsx_k ((sz_capability_t)1 << 17)

/** GPU capabilities, as each vendor's `sz_<vendor>_capabilities_*` functions report them for one
 *  device, grouped by vendor above bit 47 to leave room for more CPU capabilities and more per
 *  vendor. Each vendor's baseline runs on all its devices, as @c serial runs on every CPU, and each
 *  later tier on every device from its generation on: @c hopper from compute capability 9.0, for
 *  its clusters and their distributed shared memory, and @c blackwell from 10.0, for cluster launch
 *  control, which lets a block take over the tile of one not yet started. */
#define sz_cap_cuda_k ((sz_capability_t)1 << 48)
#define sz_cap_hopper_k ((sz_capability_t)1 << 51)
#define sz_cap_blackwell_k ((sz_capability_t)1 << 52)
#define sz_cap_rocm_k ((sz_capability_t)1 << 56)
#define sz_cap_metal_k ((sz_capability_t)1 << 60)

/** Every GPU capability above, which the CPU queries never detect, compile or enable. */
#define sz_cap_gpus_k (sz_cap_cuda_k | sz_cap_hopper_k | sz_cap_blackwell_k | sz_cap_rocm_k | sz_cap_metal_k)

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
    {"hopper", sz_cap_hopper_k},
    {"blackwell", sz_cap_blackwell_k},
    {"rocm", sz_cap_rocm_k},
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

    /** Copy between non-overlapping ranges. */
    sz_kernel_copy_k,

    /** Copy between possibly overlapping ranges. */
    sz_kernel_move_k,

    /** Fill a range with one byte. */
    sz_kernel_fill_k,

    /** Map every byte through a 256-entry table. */
    sz_kernel_lookup_k,

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

    /** Sum of all bytes. */
    sz_kernel_bytesum_k,

    /** One-shot 64-bit hash. */
    sz_kernel_hash_k,

    /** One string hashed under several seeds at once. */
    sz_kernel_hash_multiseed_k,

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

    /** Stable argsort of a sequence of strings. */
    sz_kernel_sequence_argsort_k,

    /** Stable case-insensitive argsort of a sequence of strings. */
    sz_kernel_sequence_argsort_uncased_k,

    /** Intersection of two sequences of strings. */
    sz_kernel_sequence_intersect_k,

    /** Count of UTF-8 runes. */
    sz_kernel_utf8_count_k,

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

    /** Overlap engine preparation. */
    sz_kernel_overlap_engine_init_k,

    /** Overlap scores of a batch against the prepared queries. */
    sz_kernel_overlap_scores_k,

    /** Substrings engine preparation. */
    sz_kernel_substrings_engine_init_k,

    /** Occurrence counts of the prepared needles. */
    sz_kernel_substrings_counts_k,

    /** Occurrences of the prepared needles. */
    sz_kernel_substrings_find_k,

    /** Replacement of the prepared needles. */
    sz_kernel_substrings_replace_k,

    /** BM25 scores of the prepared needles. */
    sz_kernel_substrings_bm25_scores_k,
} sz_kernel_kind_t;

/** Canonical name of a kernel kind - the spelling bindings parse and interchange formats carry;
 *  "unknown" for unrecognized values. */
STRINGZILLA_CONSTEXPR char const *sz_kernel_name(sz_kernel_kind_t kind) {
    switch (kind) {
    case sz_kernel_unknown_k: return "unknown";
    case sz_kernel_equal_k: return "equal";
    case sz_kernel_order_k: return "order";
    case sz_kernel_copy_k: return "copy";
    case sz_kernel_move_k: return "move";
    case sz_kernel_fill_k: return "fill";
    case sz_kernel_lookup_k: return "lookup";
    case sz_kernel_find_k: return "find";
    case sz_kernel_rfind_k: return "rfind";
    case sz_kernel_find_byte_k: return "find_byte";
    case sz_kernel_rfind_byte_k: return "rfind_byte";
    case sz_kernel_find_byteset_k: return "find_byteset";
    case sz_kernel_rfind_byteset_k: return "rfind_byteset";
    case sz_kernel_bytesum_k: return "bytesum";
    case sz_kernel_hash_k: return "hash";
    case sz_kernel_hash_multiseed_k: return "hash_multiseed";
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
    case sz_kernel_sequence_argsort_k: return "sequence_argsort";
    case sz_kernel_sequence_argsort_uncased_k: return "sequence_argsort_uncased";
    case sz_kernel_sequence_intersect_k: return "sequence_intersect";
    case sz_kernel_utf8_count_k: return "utf8_count";
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
    case sz_kernel_overlap_engine_init_k: return "overlap_engine_init";
    case sz_kernel_overlap_scores_k: return "overlap_scores";
    case sz_kernel_substrings_engine_init_k: return "substrings_engine_init";
    case sz_kernel_substrings_counts_k: return "substrings_counts";
    case sz_kernel_substrings_find_k: return "substrings_find";
    case sz_kernel_substrings_replace_k: return "substrings_replace";
    case sz_kernel_substrings_bm25_scores_k: return "substrings_bm25_scores";
    default: return "unknown";
    }
}

/** Inverse of @c sz_kernel_name over an explicit-length string; @c sz_kernel_unknown_k for
 *  unrecognized names. */
STRINGZILLA_CONSTEXPR sz_kernel_kind_t sz_kernel_named(char const *name, sz_size_t length) {
    if (sz_same_literal_(name, length, "equal")) return sz_kernel_equal_k;
    if (sz_same_literal_(name, length, "order")) return sz_kernel_order_k;
    if (sz_same_literal_(name, length, "copy")) return sz_kernel_copy_k;
    if (sz_same_literal_(name, length, "move")) return sz_kernel_move_k;
    if (sz_same_literal_(name, length, "fill")) return sz_kernel_fill_k;
    if (sz_same_literal_(name, length, "lookup")) return sz_kernel_lookup_k;
    if (sz_same_literal_(name, length, "find")) return sz_kernel_find_k;
    if (sz_same_literal_(name, length, "rfind")) return sz_kernel_rfind_k;
    if (sz_same_literal_(name, length, "find_byte")) return sz_kernel_find_byte_k;
    if (sz_same_literal_(name, length, "rfind_byte")) return sz_kernel_rfind_byte_k;
    if (sz_same_literal_(name, length, "find_byteset")) return sz_kernel_find_byteset_k;
    if (sz_same_literal_(name, length, "rfind_byteset")) return sz_kernel_rfind_byteset_k;
    if (sz_same_literal_(name, length, "bytesum")) return sz_kernel_bytesum_k;
    if (sz_same_literal_(name, length, "hash")) return sz_kernel_hash_k;
    if (sz_same_literal_(name, length, "hash_multiseed")) return sz_kernel_hash_multiseed_k;
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
    if (sz_same_literal_(name, length, "sequence_argsort")) return sz_kernel_sequence_argsort_k;
    if (sz_same_literal_(name, length, "sequence_argsort_uncased")) return sz_kernel_sequence_argsort_uncased_k;
    if (sz_same_literal_(name, length, "sequence_intersect")) return sz_kernel_sequence_intersect_k;
    if (sz_same_literal_(name, length, "utf8_count")) return sz_kernel_utf8_count_k;
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
    if (sz_same_literal_(name, length, "overlap_engine_init")) return sz_kernel_overlap_engine_init_k;
    if (sz_same_literal_(name, length, "overlap_scores")) return sz_kernel_overlap_scores_k;
    if (sz_same_literal_(name, length, "substrings_engine_init")) return sz_kernel_substrings_engine_init_k;
    if (sz_same_literal_(name, length, "substrings_counts")) return sz_kernel_substrings_counts_k;
    if (sz_same_literal_(name, length, "substrings_find")) return sz_kernel_substrings_find_k;
    if (sz_same_literal_(name, length, "substrings_replace")) return sz_kernel_substrings_replace_k;
    if (sz_same_literal_(name, length, "substrings_bm25_scores")) return sz_kernel_substrings_bm25_scores_k;
    return sz_kernel_unknown_k;
}

/** Signature of the @c sz_hash_best kernels. */
typedef sz_status_t (*sz_kernel_hash_t)(sz_cptr_t, sz_size_t, sz_u64_t, sz_u64_t *, sz_stream_t);

/** Signature of the @c sz_hash_multiseed_best kernels. */
typedef sz_status_t (*sz_kernel_hash_multiseed_t)(sz_cptr_t, sz_size_t, sz_u64_t const *, sz_size_t, sz_u64_t *,
                                                  sz_stream_t);

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
typedef sz_status_t (*sz_kernel_sha256_multistate_update_t)(struct sz_sha256_state_t *, struct sz_sequence_t const *,
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

/** Signature of the @c sz_lookup_best kernels. */
typedef sz_status_t (*sz_kernel_lookup_t)(sz_ptr_t, sz_cptr_t, sz_size_t, sz_cptr_t, sz_stream_t);

/** Signature of the @c sz_copy_best kernels. */
typedef sz_status_t (*sz_kernel_copy_t)(sz_ptr_t, sz_cptr_t, sz_size_t, sz_stream_t);

/** Signature of the @c sz_move_best kernels. */
typedef sz_status_t (*sz_kernel_move_t)(sz_ptr_t, sz_cptr_t, sz_size_t, sz_stream_t);

/** Signature of the @c sz_fill_best kernels. */
typedef sz_status_t (*sz_kernel_fill_t)(sz_ptr_t, sz_size_t, sz_u8_t, sz_stream_t);

/** Signature of the @c sz_find_byte_best and @c sz_rfind_byte_best kernels. */
typedef sz_status_t (*sz_kernel_find_byte_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_cptr_t *, sz_stream_t);

/** Signature of the @c sz_find_best and @c sz_rfind_best kernels. */
typedef sz_status_t (*sz_kernel_find_t)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t, sz_cptr_t *, sz_stream_t);

/** Signature of the @c sz_find_byteset_best and @c sz_rfind_byteset_best kernels. */
typedef sz_status_t (*sz_kernel_find_byteset_t)(sz_cptr_t, sz_size_t, sz_byteset_t const *, sz_cptr_t *, sz_stream_t);

/** Signature of the @c sz_sequence_argsort_best and @c sz_sequence_argsort_uncased_best kernels. */
typedef sz_status_t (*sz_kernel_sequence_argsort_t)(struct sz_sequence_t const *, sz_size_t, sz_bool_t,
                                                    sz_allocator_t *, sz_sorted_idx_t *, sz_stream_t);

/** Signature of the benchmark-only @c sz_pgrams_sort_serial_ integer sort helper and its tiers. */
typedef sz_status_t (*sz_pgrams_sort_t_)(sz_pgram_t *, sz_size_t, sz_allocator_t *, sz_sorted_idx_t *);

/** Signature of the @c sz_sequence_intersect_best kernels. */
typedef sz_status_t (*sz_kernel_sequence_intersect_t)(struct sz_sequence_t const *, struct sz_sequence_t const *,
                                                      sz_allocator_t *, sz_u64_t, sz_size_t *, sz_sorted_idx_t *,
                                                      sz_sorted_idx_t *, sz_stream_t);

/** Signature of every @c sz_levenshtein_engine_init kernel. */
typedef sz_status_t (*sz_kernel_levenshtein_engine_init_t)(struct sz_levenshtein_engine_t *,
                                                           struct sz_sequence_t const *, sz_levenshtein_symbol_t,
                                                           sz_allocator_t *, sz_stream_t);

/** Signature of every @c sz_levenshtein_distances kernel, at either alphabet. */
typedef sz_status_t (*sz_kernel_levenshtein_distances_t)(struct sz_levenshtein_engine_t *, struct sz_sequence_t const *,
                                                         sz_size_t *, sz_size_t, sz_stream_t);

/** Signature of every @c sz_overlap_engine_init kernel. */
typedef sz_status_t (*sz_kernel_overlap_engine_init_t)(struct sz_overlap_engine_t *, struct sz_sequence_t const *,
                                                       sz_size_t const *, sz_size_t, sz_size_t, sz_allocator_t *,
                                                       sz_stream_t);

/** Signature of every @c sz_overlap_scores kernel. */
typedef sz_status_t (*sz_kernel_overlap_scores_t)(struct sz_overlap_engine_t *, struct sz_sequence_t const *,
                                                  sz_f32_t *, sz_size_t, sz_size_t, sz_stream_t);

/** Signature of every @c sz_substrings_engine_init kernel. */
typedef sz_status_t (*sz_kernel_substrings_engine_init_t)(struct sz_substrings_engine_t *, struct sz_sequence_t const *,
                                                          sz_substrings_case_sensitivity_t,
                                                          sz_substrings_overlap_policy_t, sz_size_t, sz_size_t,
                                                          sz_size_t, sz_allocator_t *, sz_stream_t);

/** Signature of every @c sz_substrings_counts kernel. */
typedef sz_status_t (*sz_kernel_substrings_counts_t)(struct sz_substrings_engine_t *, struct sz_sequence_t const *,
                                                     sz_size_t *, sz_size_t, sz_stream_t);

/** Signature of every @c sz_substrings_find kernel. */
typedef sz_status_t (*sz_kernel_substrings_find_t)(struct sz_substrings_engine_t *, struct sz_sequence_t const *,
                                                   struct sz_substrings_match_t *, sz_size_t, sz_size_t *, sz_stream_t);

/** Signature of every @c sz_substrings_replace kernel. */
typedef sz_status_t (*sz_kernel_substrings_replace_t)(struct sz_substrings_engine_t *, struct sz_sequence_t const *,
                                                      struct sz_sequence_t const *, sz_ptr_t, sz_size_t, sz_size_t *,
                                                      sz_stream_t);

/** Signature of every @c sz_substrings_bm25_scores kernel. */
typedef sz_status_t (*sz_kernel_substrings_bm25_scores_t)(struct sz_substrings_engine_t *, struct sz_sequence_t const *,
                                                          sz_f32_t const *, struct sz_substrings_bm25_t const *,
                                                          sz_f32_t const *, sz_f32_t *, sz_size_t, sz_stream_t);

/** Any kernel, cast back to its signature before the call. */
typedef void (*sz_kernel_punned_t)(void);

/** Returns the capabilities whose kernels were compiled into this binary, as decided by the
 *  `STRINGZILLA_TARGET_*` macros. Says nothing about the current CPU - see
 *  @c sz_capabilities_detected_cpu_. */
STRINGZILLA_CONSTEXPR sz_capability_t sz_capabilities_compiled_cpu_(void) {
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

/** Returns the capabilities the compiler's own flags guarantee, for the platforms whose OS cannot
 *  be asked. An engine validates a WebAssembly module whole, so there they are what it holds. */
STRINGZILLA_CONSTEXPR sz_capability_t sz_capabilities_implied_cpu_(void) {
    sz_capability_t capabilities = sz_cap_serial_k;
#if defined(__ARM_NEON)
    capabilities |= sz_cap_neon_k;
#endif
#if defined(__ARM_FEATURE_AES) || defined(__ARM_FEATURE_CRYPTO)
    capabilities |= sz_cap_neonaes_k;
#endif
#if defined(__ARM_FEATURE_SHA2) || defined(__ARM_FEATURE_CRYPTO)
    capabilities |= sz_cap_neonsha_k;
#endif
#if defined(__ARM_FEATURE_SVE)
    capabilities |= sz_cap_sve_k;
#endif
#if defined(__ARM_FEATURE_SVE2)
    capabilities |= sz_cap_sve2_k;
#endif
#if defined(__ARM_FEATURE_SVE2AES) || defined(__ARM_FEATURE_SVE2_AES)
    capabilities |= sz_cap_sve2aes_k;
#endif
#if defined(__riscv_vector)
    capabilities |= sz_cap_rvv_k;
#endif
#if defined(__riscv_vector) && defined(__riscv_zvkned) && defined(__riscv_zvknhb)
    capabilities |= sz_cap_rvvcrypto_k;
#endif
#if defined(__loongarch_asx)
    capabilities |= sz_cap_loongsonasx_k;
#endif
#if defined(__POWER9_VECTOR__)
    capabilities |= sz_cap_powervsx_k;
#endif
#if defined(__wasm_simd128__)
    capabilities |= sz_cap_v128_k;
#endif
#if defined(__wasm_relaxed_simd__)
    capabilities |= sz_cap_v128relaxed_k;
#endif
    return capabilities;
}

/*  The detectors below report the full hardware capability set, independent of which
 *  `STRINGZILLA_TARGET_*` kits this build compiled in: @c sz_capabilities_enabled_cpu ANDs their
 *  result with the compile-time mask anyway, and the instructions involved are unconditionally
 *  safe. @c cpuid is baseline x86-64 with @c xgetbv behind the OSXSAVE check, and the Arm @c mrs
 *  reads run only once the kernel says it emulates them. Keeping detection unconditional lets even
 *  a serial-only build report what this machine runs. */

#if STRINGZILLA_ARCH_ARM64_

/** The capabilities of the current 64-bit Arm CPU. */
STRINGZILLA_INLINE sz_capability_t sz_capabilities_detected_arm64_(void) {
#if STRINGZILLA_OS_APPLE_

    // On Apple Silicon, `mrs` is not allowed in user-space, so we need to use the `sysctl` API.
    uint32_t supports_neon = 0;
    uint32_t supports_neonaes = 0;
    uint32_t supports_neonsha = 0;
    size_t size = sizeof(supports_neon);
    if (sysctlbyname("hw.optional.neon", &supports_neon, &size, NULL, 0) != 0) supports_neon = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_AES", &supports_neonaes, &size, NULL, 0) != 0) supports_neonaes = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_SHA256", &supports_neonsha, &size, NULL, 0) != 0) supports_neonsha = 0;

    return (sz_capability_t)(                     //
        (sz_cap_neon_k * (supports_neon)) |       //
        (sz_cap_neonaes_k * (supports_neonaes)) | //
        (sz_cap_neonsha_k * (supports_neonsha)) | //
        (sz_cap_serial_k));

#elif (STRINGZILLA_OS_LINUX_ || STRINGZILLA_OS_FREEBSD_) && STRINGZILLA_WITH_LIBC

#if STRINGZILLA_OS_LINUX_
    unsigned long hwcap = getauxval(AT_HWCAP);
#else
    unsigned long hwcap = 0;
    elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
#endif
    // HWCAP_CPUID, bit 11: the kernel emulates EL0 reads of the ID registers, which trap otherwise.
    if (!(hwcap & (1UL << 11))) return (sz_capability_t)(sz_cap_neon_k | sz_cap_serial_k);

    // Each raw `mrs` writes x0, as older assemblers reject the symbolic names of the SVE registers.
    unsigned long id_aa64isar0_el1 = 0, id_aa64pfr0_el1 = 0, id_aa64zfr0_el1 = 0;
    __asm__ __volatile__(".inst 0xD5380600\n\tmov %0, x0" : "=r"(id_aa64isar0_el1) : : "x0"); // ID_AA64ISAR0_EL1
    __asm__ __volatile__(".inst 0xD5380400\n\tmov %0, x0" : "=r"(id_aa64pfr0_el1) : : "x0");  // ID_AA64PFR0_EL1

    // AdvSIMD, bits [23:20] of ID_AA64PFR0_EL1, reads 0b1111 on the R-profile CPUs that lack NEON.
    unsigned supports_neon = ((id_aa64pfr0_el1 >> 20) & 0xF) != 0xF;
    // AES, bits [7:4], and SHA2, bits [15:12], of ID_AA64ISAR0_EL1.
    unsigned supports_neonaes = ((id_aa64isar0_el1 >> 4) & 0xF) >= 1;
    unsigned supports_neonsha = ((id_aa64isar0_el1 >> 12) & 0xF) >= 1;
    // SVE, bits [35:32] of ID_AA64PFR0_EL1, gates ID_AA64ZFR0_EL1: SVEver in [3:0], AES in [7:4].
    unsigned supports_sve = ((id_aa64pfr0_el1 >> 32) & 0xF) >= 1;
    if (supports_sve)
        __asm__ __volatile__(".inst 0xD5380480\n\tmov %0, x0" : "=r"(id_aa64zfr0_el1) : : "x0"); // ID_AA64ZFR0_EL1
    unsigned supports_sve2 = ((id_aa64zfr0_el1) & 0xF) >= 1;
    unsigned supports_sve2aes = ((id_aa64zfr0_el1 >> 4) & 0xF) >= 1;

    return (sz_capability_t)(                     //
        (sz_cap_neon_k * (supports_neon)) |       //
        (sz_cap_neonaes_k * (supports_neonaes)) | //
        (sz_cap_neonsha_k * (supports_neonsha)) | //
        (sz_cap_sve_k * (supports_sve)) |         //
        (sz_cap_sve2_k * (supports_sve2)) |       //
        (sz_cap_sve2aes_k * (supports_sve2aes)) | //
        (sz_cap_serial_k));

#elif STRINGZILLA_OS_WINDOWS_

    // On Windows ARM, use the `IsProcessorFeaturePresent` API for capability detection.
    // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent
    unsigned supports_neon = IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE);
    unsigned supports_crypto = IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE);

    return (sz_capability_t)(                    //
        (sz_cap_neon_k * (supports_neon)) |      //
        (sz_cap_neonaes_k * (supports_crypto)) | //
        (sz_cap_neonsha_k * (supports_crypto)) | //
        (sz_cap_serial_k));

#else
    return sz_capabilities_implied_cpu_();
#endif
}

#endif // STRINGZILLA_ARCH_ARM64_

#if STRINGZILLA_ARCH_X8664_

/** The capabilities of the current x86-64 CPU, as far as the OS saves the registers they use. */
STRINGZILLA_INLINE sz_capability_t sz_capabilities_detected_x8664_(void) {

    /// The states of 4 registers populated for a specific "cpuid" assembly call
    union four_registers_t {
        int array[4];
        struct separate_t {
            unsigned eax, ebx, ecx, edx;
        } named;
    } info0, info1, info7;

#if defined(_MSC_VER) && !defined(__clang__)
    __cpuidex(info0.array, 0, 0);
    __cpuidex(info1.array, 1, 0);
    __cpuidex(info7.array, 7, 0);
#else
    __asm__ __volatile__( //
        "cpuid"
        : "=a"(info0.named.eax), "=b"(info0.named.ebx), "=c"(info0.named.ecx), "=d"(info0.named.edx)
        : "a"(0), "c"(0));
    __asm__ __volatile__( //
        "cpuid"
        : "=a"(info1.named.eax), "=b"(info1.named.ebx), "=c"(info1.named.ecx), "=d"(info1.named.edx)
        : "a"(1), "c"(0));
    __asm__ __volatile__( //
        "cpuid"
        : "=a"(info7.named.eax), "=b"(info7.named.ebx), "=c"(info7.named.ecx), "=d"(info7.named.edx)
        : "a"(7), "c"(0));
#endif

    // Querying a leaf above the highest supported one returns the highest leaf's data, not zeros,
    // so on early x86-64 parts (max basic leaf below 7) the "leaf 7" registers would hold unrelated
    // bits. The AVX family is already masked by the XGETBV/OSXSAVE chain below, but SHA-NI is read
    // from leaf 7 unmasked, so the whole leaf is zeroed when it does not exist.
    if (info0.named.eax < 7) info7.named.eax = info7.named.ebx = info7.named.ecx = info7.named.edx = 0;

    // Gate AVX/AVX-512 on OS-enabled extended state (XGETBV)
    unsigned has_osxsave = (info1.named.ecx & (1u << 27)) != 0; // OSXSAVE
    unsigned has_avx = (info1.named.ecx & (1u << 28)) != 0;     // AVX

    unsigned long long xcr0 = 0;
    if (has_osxsave) {
#if defined(_MSC_VER) && !defined(__clang__)
        xcr0 = _xgetbv(0);
#else
        unsigned eax, edx;
        __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0)); // xgetbv
        xcr0 = ((unsigned long long)edx << 32) | eax;
#endif
    }

    unsigned os_avx_enabled = has_osxsave && has_avx && ((xcr0 & 0x6u) == 0x6u); // XMM+YMM
    unsigned os_avx512_enabled = os_avx_enabled && ((xcr0 & 0xE0u) == 0xE0u);    // OPMASK+ZMM

    // Check for AVX2/AVX-512 (Function ID 7), masked by OS state
    // https://github.com/llvm/llvm-project/blob/50598f0ff44f3a4e75706f8c53f3380fe7faa896/clang/lib/Headers/cpuid.h#L148
    unsigned supports_avx2 = os_avx_enabled && ((info7.named.ebx & 0x00000020u) != 0);
    unsigned supports_avx512f = os_avx512_enabled && ((info7.named.ebx & 0x00010000u) != 0);
    unsigned supports_avx512bw = os_avx512_enabled && ((info7.named.ebx & 0x40000000u) != 0);
    unsigned supports_avx512vl = os_avx512_enabled && ((info7.named.ebx & 0x80000000u) != 0);
    unsigned supports_avx512vbmi = os_avx512_enabled && ((info7.named.ecx & 0x00000002u) != 0);
    unsigned supports_avx512vbmi2 = os_avx512_enabled && ((info7.named.ecx & 0x00000040u) != 0);
    unsigned supports_vaes = os_avx512_enabled && ((info7.named.ecx & 0x00000200u) != 0);

    // Check for SSE4.2, AES-NI, and SHA-NI (Function ID 1 and 7)
    unsigned supports_sse42 = ((info1.named.ecx & 0x00100000u) != 0);
    unsigned supports_aesni = ((info1.named.ecx & 0x02000000u) != 0);
    unsigned supports_shani = ((info7.named.ebx & 0x20000000u) != 0); // SHA-NI bit 29 in EBX from CPUID(7,0)

    return (sz_capability_t)(                                                                 //
        (sz_cap_westmere_k * (supports_sse42 && supports_aesni)) |                            //
        (sz_cap_goldmont_k * (supports_shani)) |                                              //
        (sz_cap_haswell_k * (supports_avx2)) |                                                //
        (sz_cap_skylake_k * (supports_avx512f && supports_avx512vl && supports_avx512bw)) |   //
        (sz_cap_icelake_k * (supports_avx512vbmi && supports_avx512vbmi2 && supports_vaes)) | //
        (sz_cap_serial_k));
}

#endif // STRINGZILLA_ARCH_X8664_

#if STRINGZILLA_ARCH_RISCV64_

/** The capabilities of the current 64-bit RISC-V CPU. */
STRINGZILLA_INLINE sz_capability_t sz_capabilities_detected_riscv64_(void) {
#if STRINGZILLA_OS_LINUX_ && STRINGZILLA_WITH_LIBC

    // The base "V" extension is reported through the auxiliary vector, but the individual
    // vector sub-extensions (vector crypto, bf16, …) are only exposed through the
    // `riscv_hwprobe(2)` syscall (number 258), introduced in Linux 6.4.
    unsigned long hwcap = getauxval(AT_HWCAP);
    sz_capability_t caps = sz_cap_serial_k;

    // HWCAP bit 21 (`COMPAT_HWCAP_ISA_V`, i.e. `1UL << ('V' - 'A')`) marks RVV 1.0.
    if (hwcap & (1UL << 21)) {
        caps = (sz_capability_t)(caps | sz_cap_rvv_k);

        // `riscv_hwprobe(2)`: fill an array of {key, value} pairs. We query a single key,
        // `RISCV_HWPROBE_KEY_IMA_EXT_0` (= 4), whose value carries the extension bitmask.
        // Constants confirmed against `/usr/riscv64-linux-gnu/include/asm/hwprobe.h`:
        //   RISCV_HWPROBE_KEY_IMA_EXT_0 == 4
        //   RISCV_HWPROBE_EXT_ZVKNED    == (1 << 21)  // Zvkned (AES)
        //   RISCV_HWPROBE_EXT_ZVKNHB    == (1 << 23)  // Zvknhb (SHA-256/512)
        struct {
            long long key;
            unsigned long long value;
        } pairs[1];
        pairs[0].key = 4; // RISCV_HWPROBE_KEY_IMA_EXT_0
        pairs[0].value = 0;
        // `long syscall(SYS_riscv_hwprobe, pairs, pair_count, cpu_count, cpus, flags)`.
        if (syscall(258, pairs, (unsigned long)1, (unsigned long)0, (void *)0, (unsigned long)0) == 0) {
            unsigned long long const has_zvkned = pairs[0].value & (1ULL << 21); // RISCV_HWPROBE_EXT_ZVKNED
            unsigned long long const has_zvknhb = pairs[0].value & (1ULL << 23); // RISCV_HWPROBE_EXT_ZVKNHB
            if (has_zvkned && has_zvknhb) caps = (sz_capability_t)(caps | sz_cap_rvvcrypto_k);
        }
    }
    return caps;

#elif STRINGZILLA_OS_FREEBSD_ && STRINGZILLA_WITH_LIBC

    // FreeBSD exposes HWCAP through `elf_aux_info`, but lacks the Linux `riscv_hwprobe`
    // syscall, so the vector crypto sub-extensions stay compile-time only here.
    unsigned long hwcap = 0;
    elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
    sz_capability_t caps = sz_cap_serial_k;
    if (hwcap & (1UL << 21)) caps = (sz_capability_t)(caps | sz_cap_rvv_k);
    return caps;

#else
    return sz_capabilities_implied_cpu_();
#endif
}

#endif // STRINGZILLA_ARCH_RISCV64_

#if STRINGZILLA_ARCH_LOONGARCH64_

/** The capabilities of the current LoongArch CPU. */
STRINGZILLA_INLINE sz_capability_t sz_capabilities_detected_loongarch64_(void) {
#if STRINGZILLA_OS_LINUX_ && STRINGZILLA_WITH_LIBC

    // The SIMD extensions are reported through the auxiliary vector, matching `asm/hwcap.h`:
    //   HWCAP_LOONGARCH_LSX  == (1 << 4)  // 128-bit SIMD
    //   HWCAP_LOONGARCH_LASX == (1 << 5)  // 256-bit SIMD, implies LSX
    unsigned long hwcap = getauxval(AT_HWCAP);
    return (sz_capability_t)((sz_cap_loongsonasx_k * ((hwcap & (1UL << 5)) != 0)) | sz_cap_serial_k);

#else
    return sz_capabilities_implied_cpu_();
#endif
}

#endif // STRINGZILLA_ARCH_LOONGARCH64_

#if STRINGZILLA_ARCH_PPC64_

/** The capabilities of the current IBM POWER CPU. */
STRINGZILLA_INLINE sz_capability_t sz_capabilities_detected_power64_(void) {
#if (STRINGZILLA_OS_LINUX_ || STRINGZILLA_OS_FREEBSD_) && STRINGZILLA_WITH_LIBC

    // The `powervsx` kernels target POWER9 (`-mcpu=power9`), so both facts are required,
    // matching the constants in `arch/powerpc/include/uapi/asm/cputable.h`:
    //   PPC_FEATURE_HAS_VSX     == 0x00000080 // in AT_HWCAP
    //   PPC_FEATURE2_ARCH_3_00  == 0x00800000 // in AT_HWCAP2, the POWER9 ISA level
    unsigned long hwcap = 0, hwcap2 = 0;
#if STRINGZILLA_OS_LINUX_
    hwcap = getauxval(AT_HWCAP);
    hwcap2 = getauxval(AT_HWCAP2);
#else
    elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
    elf_aux_info(AT_HWCAP2, &hwcap2, sizeof(hwcap2));
#endif
    unsigned const supports_powervsx = ((hwcap & 0x00000080UL) != 0) && ((hwcap2 & 0x00800000UL) != 0);
    return (sz_capability_t)((sz_cap_powervsx_k * supports_powervsx) | sz_cap_serial_k);

#else
    return sz_capabilities_implied_cpu_();
#endif
}

#endif // STRINGZILLA_ARCH_PPC64_

/** Prepares the calling thread for @p capabilities, behind @c sz_thread_configure_cpu. */
STRINGZILLA_INLINE sz_status_t sz_thread_configure_cpu_(sz_capability_t capabilities) {
    sz_unused_(capabilities);
    return sz_success_k;
}

/** The capabilities of the current CPU, whatever this binary compiled in. */
STRINGZILLA_INLINE sz_capability_t sz_capabilities_detected_cpu_(void) {
#if STRINGZILLA_ARCH_X8664_
    return sz_capabilities_detected_x8664_();
#elif STRINGZILLA_ARCH_ARM64_
    return sz_capabilities_detected_arm64_();
#elif STRINGZILLA_ARCH_RISCV64_
    return sz_capabilities_detected_riscv64_();
#elif STRINGZILLA_ARCH_LOONGARCH64_
    return sz_capabilities_detected_loongarch64_();
#elif STRINGZILLA_ARCH_PPC64_
    return sz_capabilities_detected_power64_();
#else
    return sz_capabilities_implied_cpu_();
#endif
}

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

/**
 *  @brief Prepares the calling thread for the kernels of @p capabilities, and only those.
 *  @param[in] capabilities The capabilities to prepare for, like @c sz_capabilities_enabled_cpu.
 *  @return @c sz_success_k.
 *
 *  Call it once on each thread that dispatches. Every current capability needs nothing, so today
 *  it returns right away.
 */
STRINGZILLA_API sz_status_t sz_thread_configure_cpu(sz_capability_t capabilities);

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
 *  @return @c sz_success_k, @c sz_missing_kernel_k when no capability in @p capabilities has it,
 *      or @c sz_missing_library_k in header-only builds.
 */
STRINGZILLA_API sz_status_t sz_find_kernel_punned(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                  sz_kernel_punned_t *kernel, sz_capability_t *capability);

/*  The library detects once per process; header-only builds ask on every call. */
#if STRINGZILLA_HEADER_ONLY

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
    *capabilities = sz_capabilities_compiled_cpu_();
    return sz_success_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_enabled_cpu(sz_capability_t *capabilities) {
    *capabilities = sz_capabilities_detected_cpu_() & sz_capabilities_compiled_cpu_();
    return sz_success_k;
}

/** @copydoc sz_thread_configure_cpu */
STRINGZILLA_API sz_status_t sz_thread_configure_cpu(sz_capability_t capabilities) {
    return sz_thread_configure_cpu_(capabilities);
}

/** @copydoc sz_capabilities_name */
STRINGZILLA_API sz_size_t sz_capabilities_name(sz_capability_t capabilities, char *buffer, sz_size_t capacity) {
    return sz_capabilities_name_(capabilities, buffer, capacity);
}

#endif // STRINGZILLA_HEADER_ONLY

/**
 *  @brief Waits for everything enqueued on @p stream, after which what its rounds wrote is readable
 *      from the host.
 *  @param[in] capabilities One device's capabilities; its group picks the runtime that waits.
 *  @param[in] stream Null on the CPU, which has nothing to wait for. On a GPU, the stream to join,
 *      which also names the device, or null for the default stream of the default device.
 *  @return @c sz_success_k; or on a device @c sz_device_memory_mismatch_k for a @p stream it cannot
 *      use, @c sz_device_code_mismatch_k when the runtime reports a failed launch, and
 *      @c sz_missing_gpu_k for a vendor this build lacks or a device that doesn't answer.
 */
STRINGZILLA_API sz_status_t sz_stream_synchronize_best(sz_capability_t capabilities, sz_stream_t stream);

/** @copydoc sz_stream_synchronize_best */
STRINGZILLA_API sz_status_t sz_stream_synchronize_serial(sz_stream_t stream);

#if STRINGZILLA_TARGET_CUDA
/** @copydoc sz_stream_synchronize_best */
STRINGZILLA_API sz_status_t sz_stream_synchronize_cuda(sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ROCM
/** @copydoc sz_stream_synchronize_best */
STRINGZILLA_API sz_status_t sz_stream_synchronize_rocm(sz_stream_t stream);
#endif

#if STRINGZILLA_WITH_METAL
/** @copydoc sz_stream_synchronize_best */
STRINGZILLA_API sz_status_t sz_stream_synchronize_metal(sz_stream_t stream);
#endif

/*  The CUDA and ROCm producers answer from their runtimes in `cuda.cuh` and `rocm.cuh`, which only
 *  their own compilers reach; Metal's answer from here, behind the switch the build stamps. */

/** How many Metal devices the system lists, or zero. */
STRINGZILLA_INLINE sz_size_t sz_device_count_metal_(void) {
#if STRINGZILLA_WITH_METAL
    return sz_metal_list_devices_();
#else
    return 0;
#endif
}

/** The capabilities Metal device @p ordinal runs, in the order the system lists them. The baseline
 *  needs Apple7, whose threadgroup atomics and simdgroup scans the kernels use. */
STRINGZILLA_INLINE sz_status_t sz_capabilities_detected_metal_(sz_size_t ordinal, sz_capability_t *capabilities) {
    *capabilities = 0;
    if (ordinal >= sz_device_count_metal_()) return sz_missing_gpu_k;
#if STRINGZILLA_WITH_METAL
    void *const metal_device = sz_metal_device_(ordinal);
    if (!metal_device) return sz_device_code_mismatch_k;
    SEL const supports = sel_registerName("supportsFamily:");
    sz_size_t const apple7 = 1007; // `MTLGPUFamilyApple7`
    if (((signed char (*)(void *, SEL, sz_size_t))objc_msgSend)(metal_device, supports, apple7))
        *capabilities = sz_cap_metal_k;
    sz_metal_do_(metal_device, "release");
#endif
    return sz_success_k;
}

/** Opens a command queue on Metal device @p ordinal, in the order the system lists them. */
STRINGZILLA_INLINE sz_status_t sz_stream_init_metal_(sz_size_t ordinal, sz_stream_t *stream) {
    *stream = STRINGZILLA_NULL;
    if (ordinal >= sz_device_count_metal_()) return sz_missing_gpu_k;
#if STRINGZILLA_WITH_METAL
    void *const metal_device = sz_metal_device_(ordinal);
    if (!metal_device) return sz_device_code_mismatch_k;
    *stream = sz_metal_get_(metal_device, "newCommandQueue");
    sz_metal_do_(metal_device, "release");
    if (!*stream) return sz_bad_alloc_k;
#endif
    return sz_success_k;
}

/** Waits for @p stream, so its command buffers and the frees deferred behind it drain, then
 *  releases it; a null stream is the default one, which is only waited for. */
STRINGZILLA_INLINE sz_status_t sz_stream_free_metal_(sz_stream_t stream) {
#if STRINGZILLA_WITH_METAL
    sz_status_t const status = sz_stream_synchronize_metal(stream);
    sz_metal_do_(stream, "release");
    return status;
#else
    sz_unused_(stream);
    return sz_missing_gpu_k;
#endif
}

/** The CUDA capabilities this binary holds kernels for. */
STRINGZILLA_CONSTEXPR sz_capability_t sz_capabilities_compiled_cuda_(void) {
    return sz_cap_cuda_k * STRINGZILLA_TARGET_CUDA | sz_cap_hopper_k * STRINGZILLA_TARGET_HOPPER |
           sz_cap_blackwell_k * STRINGZILLA_TARGET_BLACKWELL;
}

/** The ROCm capabilities this binary holds kernels for. */
STRINGZILLA_CONSTEXPR sz_capability_t sz_capabilities_compiled_rocm_(void) {
    return sz_cap_rocm_k * STRINGZILLA_TARGET_ROCM;
}

/** The Metal capabilities this binary holds kernels for. */
STRINGZILLA_CONSTEXPR sz_capability_t sz_capabilities_compiled_metal_(void) {
    return sz_cap_metal_k * STRINGZILLA_TARGET_METAL;
}

/*  GPU capabilities, per vendor, of one device named by its runtime's own ordinal: the index that
 *  @c cudaSetDevice or @c hipSetDevice takes, or the position in Metal's device list. Each vendor
 *  has four queries:
 *
 *  - @b sz_device_count_cuda() — how many devices the runtime sees, zero without its kernels.
 *  - @b sz_capabilities_detected_cuda() — what one device runs.
 *  - @b sz_capabilities_compiled_cuda() — what this binary holds kernels for.
 *  - @b sz_capabilities_enabled_cuda() — both at once: the mask to dispatch on that device with.
 *
 *  ROCm and Metal have the same four. Nothing is cached, as the runtimes answer from their own
 *  state: ask once per device and keep the mask. Each vendor also makes a stream on one device
 *  with @b sz_stream_init_cuda() and frees it with @b sz_stream_free_cuda(), for a caller without
 *  the vendor's runtime at hand. These are the only functions an ordinal reaches: everything
 *  consuming a mask takes a stream instead, which names its device. */

/**
 *  @brief Counts the CUDA devices the process sees.
 *  @return @c sz_success_k, or @c sz_missing_gpu_k without one.
 */
STRINGZILLA_API sz_status_t sz_device_count_cuda(sz_size_t *count);

/**
 *  @brief Reports the capabilities CUDA device @p ordinal runs.
 *  @param[in] ordinal The CUDA runtime's device index, like @c cudaSetDevice takes.
 *  @param[out] capabilities The CUDA baseline and what the device runs, zero on failure.
 *  @return @c sz_success_k, @c sz_missing_gpu_k past the last device, or
 *      @c sz_device_code_mismatch_k when the runtime fails to answer.
 */
STRINGZILLA_API sz_status_t sz_capabilities_detected_cuda(sz_size_t ordinal, sz_capability_t *capabilities);

/** Reports the CUDA capabilities this binary holds kernels for. */
STRINGZILLA_API sz_status_t sz_capabilities_compiled_cuda(sz_capability_t *capabilities);

/** @copydoc sz_capabilities_detected_cuda, narrowed to what this binary holds kernels for. */
STRINGZILLA_API sz_status_t sz_capabilities_enabled_cuda(sz_size_t ordinal, sz_capability_t *capabilities);

/** Creates a stream on CUDA device @p ordinal with @c cudaStreamCreate. */
STRINGZILLA_API sz_status_t sz_stream_init_cuda(sz_size_t ordinal, sz_stream_t *stream);

/** Destroys a stream of @ref sz_stream_init_cuda with @c cudaStreamDestroy, once its work ends. */
STRINGZILLA_API sz_status_t sz_stream_free_cuda(sz_stream_t stream);

/** @copydoc sz_device_count_cuda, for ROCm. */
STRINGZILLA_API sz_status_t sz_device_count_rocm(sz_size_t *count);

/** @copydoc sz_capabilities_detected_cuda, for ROCm, whose ordinal @c hipSetDevice takes. */
STRINGZILLA_API sz_status_t sz_capabilities_detected_rocm(sz_size_t ordinal, sz_capability_t *capabilities);

/** @copydoc sz_capabilities_compiled_cuda, for ROCm. */
STRINGZILLA_API sz_status_t sz_capabilities_compiled_rocm(sz_capability_t *capabilities);

/** @copydoc sz_capabilities_enabled_cuda, for ROCm. */
STRINGZILLA_API sz_status_t sz_capabilities_enabled_rocm(sz_size_t ordinal, sz_capability_t *capabilities);

/** Creates a stream on ROCm device @p ordinal with @c hipStreamCreate. */
STRINGZILLA_API sz_status_t sz_stream_init_rocm(sz_size_t ordinal, sz_stream_t *stream);

/** Destroys a stream of @ref sz_stream_init_rocm with @c hipStreamDestroy, once its work ends. */
STRINGZILLA_API sz_status_t sz_stream_free_rocm(sz_stream_t stream);

/** @copydoc sz_device_count_cuda, for Metal. */
STRINGZILLA_API sz_status_t sz_device_count_metal(sz_size_t *count);

/** @copydoc sz_capabilities_detected_cuda, for Metal, whose devices count in system order. */
STRINGZILLA_API sz_status_t sz_capabilities_detected_metal(sz_size_t ordinal, sz_capability_t *capabilities);

/** @copydoc sz_capabilities_compiled_cuda, for Metal. */
STRINGZILLA_API sz_status_t sz_capabilities_compiled_metal(sz_capability_t *capabilities);

/** @copydoc sz_capabilities_enabled_cuda, for Metal. */
STRINGZILLA_API sz_status_t sz_capabilities_enabled_metal(sz_size_t ordinal, sz_capability_t *capabilities);

/** Opens an @c id<MTLCommandQueue> stream on Metal device @p ordinal with @c newCommandQueue. */
STRINGZILLA_API sz_status_t sz_stream_init_metal(sz_size_t ordinal, sz_stream_t *stream);

/** Waits for a queue of @ref sz_stream_init_metal, then releases it. */
STRINGZILLA_API sz_status_t sz_stream_free_metal(sz_stream_t stream);

#if STRINGZILLA_HEADER_ONLY

/*  Compiled for CUDA or ROCm, the producers come from `cuda.cuh` or `rocm.cuh`, included below. */
#if !STRINGZILLA_TARGET_CUDA
STRINGZILLA_API sz_status_t sz_device_count_cuda(sz_size_t *count) {
    *count = 0;
    return sz_missing_gpu_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_detected_cuda(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_gpu_k;
}
STRINGZILLA_API sz_status_t sz_stream_init_cuda(sz_size_t ordinal, sz_stream_t *stream) {
    sz_unused_(ordinal);
    *stream = STRINGZILLA_NULL;
    return sz_missing_gpu_k;
}
STRINGZILLA_API sz_status_t sz_stream_free_cuda(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_gpu_k;
}
#endif // !STRINGZILLA_TARGET_CUDA
STRINGZILLA_API sz_status_t sz_capabilities_compiled_cuda(sz_capability_t *capabilities) {
    *capabilities = sz_capabilities_compiled_cuda_();
    return sz_success_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_enabled_cuda(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_status_t const status = sz_capabilities_detected_cuda(ordinal, capabilities);
    *capabilities &= sz_capabilities_compiled_cuda_();
    return status;
}
#if !STRINGZILLA_TARGET_ROCM
STRINGZILLA_API sz_status_t sz_device_count_rocm(sz_size_t *count) {
    *count = 0;
    return sz_missing_gpu_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_detected_rocm(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_unused_(ordinal);
    *capabilities = 0;
    return sz_missing_gpu_k;
}
STRINGZILLA_API sz_status_t sz_stream_init_rocm(sz_size_t ordinal, sz_stream_t *stream) {
    sz_unused_(ordinal);
    *stream = STRINGZILLA_NULL;
    return sz_missing_gpu_k;
}
STRINGZILLA_API sz_status_t sz_stream_free_rocm(sz_stream_t stream) {
    sz_unused_(stream);
    return sz_missing_gpu_k;
}
#endif // !STRINGZILLA_TARGET_ROCM
STRINGZILLA_API sz_status_t sz_capabilities_compiled_rocm(sz_capability_t *capabilities) {
    *capabilities = sz_capabilities_compiled_rocm_();
    return sz_success_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_enabled_rocm(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_status_t const status = sz_capabilities_detected_rocm(ordinal, capabilities);
    *capabilities &= sz_capabilities_compiled_rocm_();
    return status;
}
STRINGZILLA_API sz_status_t sz_device_count_metal(sz_size_t *count) {
    *count = sz_device_count_metal_();
    return *count ? sz_success_k : sz_missing_gpu_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_detected_metal(sz_size_t ordinal, sz_capability_t *capabilities) {
    return sz_capabilities_detected_metal_(ordinal, capabilities);
}
STRINGZILLA_API sz_status_t sz_capabilities_compiled_metal(sz_capability_t *capabilities) {
    *capabilities = sz_capabilities_compiled_metal_();
    return sz_success_k;
}
STRINGZILLA_API sz_status_t sz_capabilities_enabled_metal(sz_size_t ordinal, sz_capability_t *capabilities) {
    sz_status_t const status = sz_capabilities_detected_metal_(ordinal, capabilities);
    *capabilities &= sz_capabilities_compiled_metal_();
    return status;
}
STRINGZILLA_API sz_status_t sz_stream_init_metal(sz_size_t ordinal, sz_stream_t *stream) {
    return sz_stream_init_metal_(ordinal, stream);
}
STRINGZILLA_API sz_status_t sz_stream_free_metal(sz_stream_t stream) { return sz_stream_free_metal_(stream); }

STRINGZILLA_API sz_status_t sz_stream_synchronize_best(sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
} // extern "C"
#endif

/*  After the declarations they define, outside `extern "C"`, as the runtimes declare templates. */
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/cuda.cuh"
#include "stringzilla/rocm.cuh"
#endif

#endif // STRINGZILLA_CAPABILITIES_H_
