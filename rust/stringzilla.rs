//! Single-string operations and batch engines, both with SIMD acceleration.
//!
//! Provides fast string search, comparison, hashing, and manipulation functions optimized with SWAR
//! and SIMD instructions, plus the stateful cross-product engines - Levenshtein distances, window
//! overlap, and multi-pattern search - that prepare a batch of queries once and reuse it per round.
//!
//! File: rust/stringzilla.rs
//! Author: Ash Vardanian

mod cipher;
mod compare;
mod find;
mod hash;
mod intersect;
mod levenshtein;
mod memory;
mod overlap;
mod sort;
mod substrings;
mod types;
mod utf8_graphemes;
mod utf8_linebreaks;
mod utf8_norm;
mod utf8_runes;
mod utf8_sentences;
mod utf8_tokens;
mod utf8_uncased;
mod utf8_uncased_fold;
mod utf8_wordbreaks;

use core::ffi::{c_char, c_int, c_void};

pub use cipher::*;
pub use compare::*;
pub use find::*;
pub use hash::*;
pub use intersect::*;
pub use levenshtein::*;
pub use memory::*;
pub use overlap::*;
pub use sort::*;
pub use substrings::*;
pub use types::*;
pub use utf8_graphemes::*;
pub use utf8_linebreaks::*;
pub use utf8_norm::*;
pub use utf8_runes::*;
pub use utf8_sentences::*;
pub use utf8_tokens::*;
pub use utf8_uncased::*;
pub use utf8_uncased_fold::*;
pub use utf8_wordbreaks::*;

// Import the functions from the StringZillable C library. Every dispatch point takes the capability
// mask to pick a kernel from and a stream, null on the CPU, and reports an `sz_status_t`.
extern "C" {

    pub(crate) fn sz_version_major() -> c_int;
    pub(crate) fn sz_version_minor() -> c_int;
    pub(crate) fn sz_version_patch() -> c_int;
    pub(crate) fn sz_status_name(status: sz_status_t) -> *const c_char;
    pub(crate) fn sz_capabilities_name(
        capabilities: sz_capability_t,
        buffer: *mut c_char,
        capacity: sz_size_t,
    ) -> sz_size_t;
    pub(crate) fn sz_cpu_capabilities_detected(capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_cpu_capabilities_compiled(capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_cpu_capabilities_enabled(capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_cpu_configure_thread(capabilities: sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_cuda_count_devices(count: *mut sz_size_t) -> sz_status_t;
    pub(crate) fn sz_cuda_capabilities_detected(ordinal: sz_size_t, capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_cuda_capabilities_compiled(capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_cuda_capabilities_enabled(ordinal: sz_size_t, capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_rocm_count_devices(count: *mut sz_size_t) -> sz_status_t;
    pub(crate) fn sz_rocm_capabilities_detected(ordinal: sz_size_t, capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_rocm_capabilities_compiled(capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_rocm_capabilities_enabled(ordinal: sz_size_t, capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_metal_count_devices(count: *mut sz_size_t) -> sz_status_t;
    pub(crate) fn sz_metal_capabilities_detected(ordinal: sz_size_t, capabilities: *mut sz_capability_t)
        -> sz_status_t;
    pub(crate) fn sz_metal_capabilities_compiled(capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_metal_capabilities_enabled(ordinal: sz_size_t, capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_cuda_stream_init(ordinal: sz_size_t, stream: *mut *mut c_void) -> sz_status_t;
    pub(crate) fn sz_cuda_stream_free(stream: *mut c_void) -> sz_status_t;
    pub(crate) fn sz_rocm_stream_init(ordinal: sz_size_t, stream: *mut *mut c_void) -> sz_status_t;
    pub(crate) fn sz_rocm_stream_free(stream: *mut c_void) -> sz_status_t;
    pub(crate) fn sz_metal_stream_init(ordinal: sz_size_t, stream: *mut *mut c_void) -> sz_status_t;
    pub(crate) fn sz_metal_stream_free(stream: *mut c_void) -> sz_status_t;

    pub(crate) fn sz_memory_allocator_init_unified_best(
        allocator: *mut _SzMemoryAllocator,
        capabilities: sz_capability_t,
    ) -> sz_status_t;
    pub(crate) fn sz_sequence_copy_best(
        target: *mut _SzSequence,
        source: *const _SzSequence,
        allocator: *mut _SzMemoryAllocator,
        allocated_bytes: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_stream_synchronize_best(capabilities: sz_capability_t, stream: *mut c_void) -> sz_status_t;

    pub(crate) fn sz_copy_best(
        target: *mut c_void,
        source: *const c_void,
        length: sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_fill_best(
        target: *mut c_void,
        length: sz_size_t,
        value: u8,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_move_best(
        target: *mut c_void,
        source: *const c_void,
        length: sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_fill_random_best(
        text: *mut c_void,
        length: sz_size_t,
        nonce: u64,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_lookup_best(
        target: *mut c_void,
        source: *const c_void,
        length: sz_size_t,
        lut: *const u8,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;

    pub(crate) fn sz_find_best(
        haystack: *const c_void,
        haystack_length: sz_size_t,
        needle: *const c_void,
        needle_length: sz_size_t,
        found: *mut *const c_void,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_rfind_best(
        haystack: *const c_void,
        haystack_length: sz_size_t,
        needle: *const c_void,
        needle_length: sz_size_t,
        found: *mut *const c_void,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_find_byteset_best(
        haystack: *const c_void,
        haystack_length: sz_size_t,
        byteset: *const c_void,
        found: *mut *const c_void,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_rfind_byteset_best(
        haystack: *const c_void,
        haystack_length: sz_size_t,
        byteset: *const c_void,
        found: *mut *const c_void,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;

    pub(crate) fn sz_utf8_count_best(
        text: *const c_void,
        length: sz_size_t,
        count: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_seek_best(
        text: *const c_void,
        length: sz_size_t,
        n: sz_size_t,
        position: *mut *const c_void,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_decode_best(
        text: *const c_void,
        length: sz_size_t,
        runes: *mut u32,
        runes_capacity: sz_size_t,
        runes_count: *mut sz_size_t,
        bytes_consumed: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_newlines_best(
        text: *const c_void,
        length: sz_size_t,
        match_offsets: *mut sz_size_t,
        match_lengths: *mut sz_size_t,
        matches_capacity: sz_size_t,
        matches_count: *mut sz_size_t,
        bytes_consumed: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_whitespaces_best(
        text: *const c_void,
        length: sz_size_t,
        match_offsets: *mut sz_size_t,
        match_lengths: *mut sz_size_t,
        matches_capacity: sz_size_t,
        matches_count: *mut sz_size_t,
        bytes_consumed: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_delimiters_best(
        text: *const c_void,
        length: sz_size_t,
        match_offsets: *mut sz_size_t,
        match_lengths: *mut sz_size_t,
        matches_capacity: sz_size_t,
        matches_count: *mut sz_size_t,
        bytes_consumed: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_uncased_fold_best(
        source: *const c_void,
        source_length: sz_size_t,
        target: *mut c_void,
        target_length: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_norm_best(
        source: *const c_void,
        source_length: sz_size_t,
        form: Utf8NormalForm,
        target: *mut c_void,
        target_length: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_find_denormalized_best(
        source: *const c_void,
        source_length: sz_size_t,
        form: Utf8NormalForm,
        found: *mut *const c_void,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_uncased_needle_init_best(
        needle: *const c_void,
        needle_length: sz_size_t,
        prepared: *mut Utf8UncasedNeedle<'_>,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_uncased_search_best(
        haystack: *const c_void,
        haystack_length: sz_size_t,
        needle: *const Utf8UncasedNeedle<'_>,
        found: *mut *const c_void,
        match_length: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_uncased_order_best(
        a: *const c_void,
        a_length: sz_size_t,
        b: *const c_void,
        b_length: sz_size_t,
        ordering: *mut sz_ordering_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;

    pub(crate) fn sz_utf8_wordbreaks_best(
        text: *const c_void,
        length: sz_size_t,
        lengths: *mut sz_size_t,
        capacity: sz_size_t,
        count: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_graphemes_best(
        text: *const c_void,
        length: sz_size_t,
        lengths: *mut sz_size_t,
        capacity: sz_size_t,
        count: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_sentences_best(
        text: *const c_void,
        length: sz_size_t,
        lengths: *mut sz_size_t,
        capacity: sz_size_t,
        count: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_linebreaks_best(
        text: *const c_void,
        length: sz_size_t,
        lengths: *mut sz_size_t,
        capacity: sz_size_t,
        count: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;

    pub(crate) fn sz_equal_best(
        a: *const c_void,
        b: *const c_void,
        length: sz_size_t,
        equal: *mut sz_bool_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_order_best(
        a: *const c_void,
        a_length: sz_size_t,
        b: *const c_void,
        b_length: sz_size_t,
        ordering: *mut sz_ordering_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;

    pub(crate) fn sz_bytesum_best(
        text: *const c_void,
        length: sz_size_t,
        checksum: *mut u64,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_hash_best(
        text: *const c_void,
        length: sz_size_t,
        seed: u64,
        hash: *mut u64,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_hash_multiseed_best(
        text: *const c_void,
        length: sz_size_t,
        seeds: *const u64,
        seeds_count: sz_size_t,
        hashes: *mut u64,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_hash_state_init_best(
        state: *mut c_void,
        seed: u64,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_hash_state_update_best(
        state: *mut c_void,
        text: *const c_void,
        length: sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_hash_state_digest_best(
        state: *const c_void,
        hash: *mut u64,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_sha256_state_init_best(
        state: *mut c_void,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_sha256_state_update_best(
        state: *mut c_void,
        data: *const c_void,
        length: sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_sha256_state_digest_best(
        state: *const c_void,
        digest: *mut u8,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_sha256_multistate_update_best(
        states: *mut c_void,
        texts: *const _SzSequence,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_sha256_multistate_digest_best(
        states: *const c_void,
        states_count: sz_size_t,
        digests: *mut u8,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;

    pub(crate) fn sz_aes256_key_init_best(
        key: *mut c_void,
        secret: *const u8,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_ctr_xor_best(
        key: *const c_void,
        nonce: *const u8,
        byte_offset: u64,
        text: *const c_void,
        length: sz_size_t,
        output: *mut c_void,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_key_init_best(
        key: *mut c_void,
        secret: *const u8,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_encrypt_best(
        key: *const c_void,
        nonce: *const u8,
        associated: *const c_void,
        associated_length: sz_size_t,
        text: *const c_void,
        length: sz_size_t,
        output: *mut c_void,
        tag: *mut u8,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_decrypt_best(
        key: *const c_void,
        nonce: *const u8,
        associated: *const c_void,
        associated_length: sz_size_t,
        text: *const c_void,
        length: sz_size_t,
        output: *mut c_void,
        tag: *const u8,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_encryptor_init_best(
        encryptor: *mut c_void,
        key: *const c_void,
        nonce: *const u8,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_encryptor_associate_best(
        encryptor: *mut c_void,
        text: *const c_void,
        length: sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_encryptor_update_best(
        encryptor: *mut c_void,
        text: *const c_void,
        length: sz_size_t,
        output: *mut c_void,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_encryptor_digest_best(
        encryptor: *const c_void,
        tag: *mut u8,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_decryptor_init_best(
        decryptor: *mut c_void,
        key: *const c_void,
        nonce: *const u8,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_decryptor_associate_best(
        decryptor: *mut c_void,
        text: *const c_void,
        length: sz_size_t,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_decryptor_update_unverified_best(
        decryptor: *mut c_void,
        text: *const c_void,
        length: sz_size_t,
        output: *mut c_void,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_decryptor_verify_best(
        decryptor: *const c_void,
        tag: *const u8,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;

    pub(crate) fn sz_sequence_argsort_best(
        sequence: *const _SzSequence,
        top_count: sz_size_t,
        reverse: sz_bool_t,
        allocator: *const c_void,
        order: *mut SortedIdx,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_sequence_argsort_uncased_best(
        sequence: *const _SzSequence,
        top_count: sz_size_t,
        reverse: sz_bool_t,
        allocator: *const c_void,
        order: *mut SortedIdx,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;

    pub(crate) fn sz_sequence_intersect_best(
        first_sequence: *const _SzSequence,
        second_sequence: *const _SzSequence,
        allocator: *const c_void,
        seed: u64,
        intersection_count: *mut sz_size_t,
        first_positions: *mut SortedIdx,
        second_positions: *mut SortedIdx,
        capabilities: sz_capability_t,
        stream: *mut c_void,
    ) -> sz_status_t;

    // Cross-product engines. The capability group picks the CPU or one GPU vendor and the stream the
    // device; a null allocator resolves to the unified one of those capabilities.
    pub(crate) fn sz_levenshtein_engine_init(
        engine: *mut LevenshteinEngine,
        queries: *const _SzSequence,
        symbol: LevenshteinSymbol,
        capabilities: sz_capability_t,
        allocator: *const c_void,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_levenshtein_engine_free(engine: *mut LevenshteinEngine, stream: *mut c_void);
    pub(crate) fn sz_levenshtein_distances(
        engine: *mut LevenshteinEngine,
        candidates: *const _SzSequence,
        distances: *mut sz_size_t,
        distances_stride: sz_size_t,
        stream: *mut c_void,
    ) -> sz_status_t;

    pub(crate) fn sz_overlap_engine_init(
        engine: *mut OverlapEngine,
        queries: *const _SzSequence,
        window_widths: *const sz_size_t,
        window_widths_count: sz_size_t,
        candidates_budget: sz_size_t,
        capabilities: sz_capability_t,
        allocator: *const c_void,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_overlap_engine_free(engine: *mut OverlapEngine, stream: *mut c_void);
    pub(crate) fn sz_overlap_scores(
        engine: *mut OverlapEngine,
        candidates: *const _SzSequence,
        scores: *mut f32,
        scores_query_stride: sz_size_t,
        scores_candidate_stride: sz_size_t,
        stream: *mut c_void,
    ) -> sz_status_t;

    pub(crate) fn sz_substrings_engine_init(
        engine: *mut SubstringsEngine,
        needles: *const _SzSequence,
        case_sensitivity: CaseSensitivity,
        overlap_policy: SubstringsOverlapPolicy,
        hot_states: sz_size_t,
        matches_budget: sz_size_t,
        haystacks_budget: sz_size_t,
        capabilities: sz_capability_t,
        allocator: *const c_void,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_substrings_engine_free(engine: *mut SubstringsEngine, stream: *mut c_void);
    pub(crate) fn sz_substrings_counts(
        engine: *mut SubstringsEngine,
        haystacks: *const _SzSequence,
        counts: *mut sz_size_t,
        counts_stride: sz_size_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_substrings_find(
        engine: *mut SubstringsEngine,
        haystacks: *const _SzSequence,
        matches: *mut SubstringsMatch,
        matches_capacity: sz_size_t,
        matches_offsets: *mut sz_size_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_substrings_replace(
        engine: *mut SubstringsEngine,
        haystacks: *const _SzSequence,
        replacements: *const _SzSequence,
        target: *mut u8,
        target_capacity: sz_size_t,
        offsets: *mut sz_size_t,
        stream: *mut c_void,
    ) -> sz_status_t;
    pub(crate) fn sz_substrings_bm25_scores(
        engine: *mut SubstringsEngine,
        haystacks: *const _SzSequence,
        document_lengths: *const f32,
        parameters: *const Bm25Params,
        needle_weights: *const f32,
        scores: *mut f32,
        scores_stride: sz_size_t,
        stream: *mut c_void,
    ) -> sz_status_t;

}

/// Mirror of `sz_memory_allocator_t`, carried by value inside every engine so a release cannot be
/// handed the wrong allocator. The engines pass a null allocator and take the unified one of their
/// capabilities; `UnifiedAllocator` fills one to hand its blocks out, and `Sequence` to free its tape.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct _SzMemoryAllocator {
    pub(crate) allocate:
        Option<unsafe extern "C" fn(bytes: usize, handle: *mut c_void, stream: *mut c_void) -> *mut c_void>,
    pub(crate) free:
        Option<unsafe extern "C" fn(pointer: *mut c_void, bytes: usize, handle: *mut c_void, stream: *mut c_void)>,
    pub(crate) handle: *mut c_void,
}

/// Binds `items` into a `sz_sequence_t` that lives for the span of `call`, allocating nothing.
///
/// The accessors read the caller's slices in place, so a sequence borrows rather than copies and
/// must never outlive the call the C side makes through it.
pub(crate) fn with_sequence<Element, Return>(items: &[Element], call: impl FnOnce(&_SzSequence) -> Return) -> Return
where
    Element: AsRef<[u8]>,
{
    with_sequence_by(items, |item| item.as_ref(), call)
}

/// Binds `items` as [`with_sequence`] does, reading each one's bytes through `key`.
///
/// The C side calls `key` from inside `call`, where a panic cannot unwind and aborts the process.
pub(crate) fn with_sequence_by<Element, Key, Return>(
    items: &[Element],
    key: Key,
    call: impl FnOnce(&_SzSequence) -> Return,
) -> Return
where
    Key: Fn(&Element) -> &[u8],
{
    let view = (items, key);
    let sequence = _SzSequence {
        handle: &view as *const (&[Element], Key) as *const c_void,
        count: items.len(),
        get_start: Some(sequence_start::<Element, Key>),
        get_length: Some(sequence_length::<Element, Key>),
    };
    call(&sequence)
}

unsafe extern "C" fn sequence_start<Element, Key>(handle: *const c_void, index: usize) -> *const c_void
where
    Key: Fn(&Element) -> &[u8],
{
    let (items, key) = &*(handle as *const (&[Element], Key));
    key(&items[index]).as_ptr() as *const c_void
}

unsafe extern "C" fn sequence_length<Element, Key>(handle: *const c_void, index: usize) -> usize
where
    Key: Fn(&Element) -> &[u8],
{
    let (items, key) = &*(handle as *const (&[Element], Key));
    key(&items[index]).len()
}

/// Trait for unary string operations that only operate on `self` without needle parameters.
/// These operations include hash computation and byte sum calculation.
///
/// # Examples
///
/// Basic usage on a byte slice:
///
/// ```
/// use stringzilla::sz::StringZillableUnary;
///
/// let text = b"Hello";
/// assert_eq!(text.sz_bytesum(), 500);
/// ```
pub trait StringZillableUnary {
    /// Computes the bytesum value of unsigned bytes in a given string.
    /// This function is useful for verifying data integrity and detecting changes in
    /// binary data, such as files or network packets.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableUnary;
    ///
    /// let text = b"Hello";
    /// assert_eq!(text.sz_bytesum(), 500);
    /// ```
    fn sz_bytesum(&self) -> u64;

    /// Computes a 64-bit AES-based hash value for a given string.
    /// This function is designed to provide a high-quality hash value for use in
    /// hash tables, data structures, and cryptographic applications.
    /// Unlike the bytesum function, the hash function is order-sensitive.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableUnary;
    ///
    /// let s1 = b"Hello";
    /// let s2 = b"World";
    /// assert_ne!(s1.sz_hash(), s2.sz_hash());
    /// ```
    fn sz_hash(&self) -> u64;

    /// Returns a lazy UTF-8 character view with SIMD-accelerated operations.
    ///
    /// The view provides:
    /// - `.len()` for character count, computed on demand
    /// - `.offset_of(n)` for random access to Nth character offset
    /// - `.iter()` for efficient batched iteration over characters
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableUnary;
    ///
    /// let text = "Hello🌍";
    /// let view = text.sz_utf8_runes();
    ///
    /// // Lazy character count
    /// assert_eq!(view.len(), 6);
    ///
    /// // Random access (byte offset of Nth character)
    /// assert_eq!(view.offset_of(5), Some(5)); // 🌍 at byte 5
    ///
    /// // Efficient batched iteration
    /// let chars: Vec<char> = view.iter().collect();
    /// assert_eq!(chars, vec!['H', 'e', 'l', 'l', 'o', '🌍']);
    /// ```
    fn sz_utf8_runes(&self) -> Utf8View<'_>;

    /// Returns an iterator over lines split by UTF-8 newline characters.
    ///
    /// The iterator yields slices between newlines. Handles all Unicode newline characters
    /// including CRLF as a single delimiter.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableUnary;
    ///
    /// let text = "Hello\nWorld\r\nRust";
    /// let lines: Vec<&str> = text.sz_utf8_split_newlines()
    ///     .map(|line| std::str::from_utf8(line).unwrap())
    ///     .collect();
    /// assert_eq!(lines, vec!["Hello", "World", "Rust"]);
    /// ```
    fn sz_utf8_split_newlines(&self) -> Utf8SplitNewlines<'_>;

    /// Returns an iterator over the newline runs themselves, the separators.
    fn sz_utf8_newlines(&self) -> Utf8Newlines<'_>;

    /// Returns an iterator over segments split by UTF-8 whitespace characters.
    ///
    /// Handles all 25 Unicode "White_Space" characters; N delimiters yield N+1 segments. By default
    /// __empty segments are kept__, matching `str::split`, so runs of whitespace surface empty
    /// slices. Chain `.skip_empty()` on the returned iterator to recover the
    /// `str::split_whitespace` token behavior that drops empties.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableUnary;
    ///
    /// // KEEP (default): the double space between "Hello" and "World" yields an empty segment.
    /// let text = "Hello  World\tRust";
    /// let segments: Vec<&str> = text.sz_utf8_split_whitespaces()
    ///     .map(|segment| std::str::from_utf8(segment).unwrap())
    ///     .collect();
    /// assert_eq!(segments, vec!["Hello", "", "World", "Rust"]);
    ///
    /// // skip_empty: drops the empties to yield only the non-empty tokens.
    /// let tokens: Vec<&str> = text.sz_utf8_split_whitespaces()
    ///     .skip_empty()
    ///     .map(|token| std::str::from_utf8(token).unwrap())
    ///     .collect();
    /// assert_eq!(tokens, vec!["Hello", "World", "Rust"]);
    /// ```
    fn sz_utf8_split_whitespaces(&self) -> Utf8SplitWhitespaces<'_>;

    /// Returns an iterator over the whitespace runs themselves, the separators.
    fn sz_utf8_whitespaces(&self) -> Utf8Whitespaces<'_>;

    /// Returns an iterator splitting on any Unicode delimiter
    /// (punctuation/symbol/separator/whitespace).
    fn sz_utf8_split_delimiters(&self) -> Utf8SplitDelimiters<'_>;

    /// Returns an iterator over the delimiter runs themselves, the separators.
    fn sz_utf8_delimiters(&self) -> Utf8Delimiters<'_>;

    /// Returns an iterator over UAX-29 words, in order. Words tile the input contiguously.
    fn sz_utf8_wordbreaks(&self) -> Utf8Wordbreaks<'_>;

    /// Returns an iterator over UAX-29 grapheme clusters, in order. Clusters tile
    /// the input contiguously.
    fn sz_utf8_graphemes(&self) -> Utf8Graphemes<'_>;

    /// Returns an iterator over UAX-29 sentences, in order. Sentences tile the input contiguously.
    fn sz_utf8_sentences(&self) -> Utf8Sentences<'_>;

    /// Returns an iterator over UAX-14 line-break opportunities per Unicode TR14, in order.
    /// Linewrap segments tile the input contiguously, including soft break opportunities. For hard
    /// line splits only, use [`Self::sz_utf8_split_newlines`].
    fn sz_utf8_linebreaks(&self) -> Utf8Linebreaks<'_>;
}

impl<Source> StringZillableUnary for Source
where
    Source: AsRef<[u8]> + ?Sized,
{
    fn sz_bytesum(&self) -> u64 {
        bytesum(self)
    }

    fn sz_hash(&self) -> u64 {
        hash(self)
    }

    fn sz_utf8_runes(&self) -> Utf8View<'_> {
        Utf8View::new(self.as_ref())
    }

    fn sz_utf8_split_newlines(&self) -> Utf8SplitNewlines<'_> {
        Utf8SplitNewlines::new(self.as_ref())
    }

    fn sz_utf8_newlines(&self) -> Utf8Newlines<'_> {
        Utf8Newlines::new(self.as_ref())
    }

    fn sz_utf8_split_whitespaces(&self) -> Utf8SplitWhitespaces<'_> {
        Utf8SplitWhitespaces::new(self.as_ref())
    }

    fn sz_utf8_whitespaces(&self) -> Utf8Whitespaces<'_> {
        Utf8Whitespaces::new(self.as_ref())
    }

    fn sz_utf8_split_delimiters(&self) -> Utf8SplitDelimiters<'_> {
        Utf8SplitDelimiters::new(self.as_ref())
    }

    fn sz_utf8_delimiters(&self) -> Utf8Delimiters<'_> {
        Utf8Delimiters::new(self.as_ref())
    }

    fn sz_utf8_wordbreaks(&self) -> Utf8Wordbreaks<'_> {
        Utf8Wordbreaks::new(self.as_ref())
    }

    fn sz_utf8_graphemes(&self) -> Utf8Graphemes<'_> {
        Utf8Graphemes::new(self.as_ref())
    }

    fn sz_utf8_sentences(&self) -> Utf8Sentences<'_> {
        Utf8Sentences::new(self.as_ref())
    }

    fn sz_utf8_linebreaks(&self) -> Utf8Linebreaks<'_> {
        Utf8Linebreaks::new(self.as_ref())
    }
}

/// Prose fixtures shared by the tests of more than one UTF-8 domain module.
///
/// Realistic multi-script prose, with ASCII-source `\u{}` escapes and the rendered prose in
/// comments. Per-family segment counts are oracle-locked (ICU root / uniseg). Fixtures read by a
/// single module live beside the test that reads them instead.
#[cfg(test)]
pub(crate) mod fixtures {
    // Hotel review (German + Japanese): NFD cafe, NBSP-glued units, a sentence-ending abbreviation,
    // a CJK run.
    pub(crate) const PROSE_HOTEL_REVIEW: &str = concat!(
        "Last spring we strolled down M\u{fc}nchner Stra\u{df}e; the cafe\u{301} cortado cost 3,50\u{a0}",
        "\u{20ac} and was unreal. Dr. Vogel, our guide, swore it's the city's finest. Worth the detour?! ",
        "Absolutely \u{2014} and \u{6771}\u{4eac}\u{30bf}\u{30ef}\u{30fc} the next week, all 333\u{a0}m o",
        "f it, was breathtaking at dusk\u{2026}"
    );
    // Concert post (Korean + Japanese): conjoining L+V+T jamo, a Katakana run, an ideographic stop,
    // a 'p.m.' no-break.
    pub(crate) const PROSE_CONCERT_POST: &str = concat!(
        "\u{c624}\u{b298} \u{cf58}\u{c11c}\u{d2b8}, \u{c9c4}\u{c9dc} \u{bbf8}\u{cce4}\u{b2e4}!! \u{1112}",
        "\u{1161}\u{11ab}\u{ad6d} \u{d32c}\u{b4e4}\u{c774} \u{b2e4} \u{baa8}\u{c600}\u{ace0}, the staff b",
        "owed and said \u{c548}\u{b155}\u{d788} \u{ac00}\u{c138}\u{c694}. Setlist was pure \u{30cf}",
        "\u{30fc}\u{30c9}\u{30b3}\u{30a2}; \u{4eca}\u{65e5}\u{306f}\u{6700}\u{9ad8}\u{3060}\u{3063}",
        "\u{305f}\u{3002} We screamed \u{c0ac}\u{b791}\u{d574} till 11 p.m. sharp."
    );
    // News lede: 'U.S.A.' before a lowercase word without a break, curly quotes, thousands,
    // currency, a date range.
    pub(crate) const PROSE_NEWS_LEDE: &str = concat!(
        "The U.S.A. wasn't ready, analysts said. \u{201c}We lost 1,000 jobs,\u{201d} the mayor warned. ",
        "\u{201c}Recovery starts now.\u{201d} Filings spiked 2024/06\u{2013}2024/09, topping $1,000 per c",
        "laim. Will it hold?! No one knows for sure."
    );
    // Language lesson: a Greek final sigma, Cyrillic case pairs, a Croatian titlecase digraph, and
    // a fold-only match.
    #[allow(dead_code)] // used by the Python uncased prose test, not Rust
    pub(crate) const PROSE_LANGUAGE_LESSON: &str = concat!(
        "Greek lesson: \u{39f}\u{394}\u{39f}\u{3a3} becomes \u{3bf}\u{3b4}\u{3cc}\u{3c2} when lowercased,",
        " ending in a final \u{3c2}. Russian's easy too \u{2014} \u{41c}\u{41e}\u{421}\u{41a}\u{412}",
        "\u{410} \u{2194} \u{43c}\u{43e}\u{441}\u{43a}\u{432}\u{430}, no drama. Croatian has the digraph ",
        "\u{1c4}: titlecase \u{1c5}, lowercase \u{1c6}. Quiz \u{2014} does \u{201c}stra\u{df}e\u{201d} ma",
        "tch STRASSE? Yes, once you fold."
    );
    // RTL scripts: Hebrew gershayim, Arabic, a number-sign Prepend, an NFC niqqud reorder,
    // a Malayalam dot-reph.
    pub(crate) const PROSE_RTL_SCRIPTS: &str = concat!(
        "Hebrew acronyms take gershayim: \u{5e6}\u{5d4}\u{5f4}\u{5dc} and \u{5d0}\u{5e8}\u{5d4}\u{5f4}",
        "\u{5d1} aren't typos. Arabic flows right-to-left too \u{2014} \u{645}\u{631}\u{62d}\u{628}",
        "\u{627} \u{628}\u{627}\u{644}\u{639}\u{627}\u{644}\u{645} \u{2014} and finance text can carry th",
        "e number sign \u{600}\u{664}. Niqqud stacks marks: \u{5e9}\u{5c1}\u{5b8}\u{5dc}\u{5d5}\u{5b9}",
        "\u{5dd} must reorder under NFC. Malayalam even has a true prepend, the dot-reph \u{d4e}\u{d15}."
    );
}
