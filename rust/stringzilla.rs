//! Single-string operations and batch engines, both with SIMD acceleration.
//!
//! Provides fast string search, comparison, hashing, and manipulation functions optimized with SWAR
//! and SIMD instructions, plus the stateful cross-product engines - Levenshtein distances, window
//! overlaps and sketches, and multi-pattern search - that prepare a batch of queries once and reuse
//! it per round.
//!
//! File: rust/stringzilla.rs
//! Author: Ash Vardanian
//! Date: January 22, 2024

mod cipher;
mod compare;
mod find;
mod hash;
mod intersect;
mod levenshtein;
mod ngrams;
mod regex;
mod sort;
mod strings;
mod subdocuments;
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
pub use ngrams::*;
pub use regex::*;
pub use sort::*;
pub use strings::*;
pub use subdocuments::*;
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
unsafe extern "C" {

    pub(crate) fn sz_version_major() -> c_int;
    pub(crate) fn sz_version_minor() -> c_int;
    pub(crate) fn sz_version_patch() -> c_int;
    pub(crate) fn sz_status_name(status: sz_status_t) -> *const c_char;
    pub(crate) fn sz_capabilities_name(
        capabilities: sz_capability_t,
        buffer: *mut c_char,
        capacity: sz_size_t,
    ) -> sz_size_t;
    pub(crate) fn sz_capabilities_detected_cpu(capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_capabilities_compiled_cpu(capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_capabilities_enabled_cpu(capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_configure_thread_cpu(capabilities: sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_count_devices_cuda(count: *mut sz_size_t) -> sz_status_t;
    pub(crate) fn sz_capabilities_detected_cuda(ordinal: sz_size_t, capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_capabilities_compiled_cuda(ordinal: sz_size_t, capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_capabilities_enabled_cuda(ordinal: sz_size_t, capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_count_devices_rocm(count: *mut sz_size_t) -> sz_status_t;
    pub(crate) fn sz_capabilities_detected_rocm(ordinal: sz_size_t, capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_capabilities_compiled_rocm(ordinal: sz_size_t, capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_capabilities_enabled_rocm(ordinal: sz_size_t, capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_count_devices_metal(count: *mut sz_size_t) -> sz_status_t;
    pub(crate) fn sz_capabilities_detected_metal(ordinal: sz_size_t, capabilities: *mut sz_capability_t)
    -> sz_status_t;
    pub(crate) fn sz_capabilities_compiled_metal(ordinal: sz_size_t, capabilities: *mut sz_capability_t)
    -> sz_status_t;
    pub(crate) fn sz_capabilities_enabled_metal(ordinal: sz_size_t, capabilities: *mut sz_capability_t) -> sz_status_t;
    pub(crate) fn sz_stream_init_cuda(ordinal: sz_size_t, stream: *mut sz_stream_t) -> sz_status_t;
    pub(crate) fn sz_stream_free_cuda(stream: sz_stream_t) -> sz_status_t;
    pub(crate) fn sz_stream_init_rocm(ordinal: sz_size_t, stream: *mut sz_stream_t) -> sz_status_t;
    pub(crate) fn sz_stream_free_rocm(stream: sz_stream_t) -> sz_status_t;
    pub(crate) fn sz_stream_init_metal(ordinal: sz_size_t, stream: *mut sz_stream_t) -> sz_status_t;
    pub(crate) fn sz_stream_free_metal(stream: sz_stream_t) -> sz_status_t;

    pub(crate) fn sz_allocator_init_unified_best(
        allocator: *mut sz_allocator_t,
        capabilities: sz_capability_t,
    ) -> sz_status_t;
    pub(crate) fn sz_stream_synchronize_best(capabilities: sz_capability_t, stream: sz_stream_t) -> sz_status_t;

    pub(crate) fn sz_fill_random_best(
        text: *mut c_void,
        length: sz_size_t,
        nonce: u64,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;

    pub(crate) fn sz_strings_to_tape_best(
        strings: *const sz_strings_view_t,
        order: *const sz_size_t,
        count: sz_size_t,
        allocator: *mut sz_allocator_t,
        tape: *mut sz_strings_span_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_strings_to_rows_best(
        strings: *const sz_strings_view_t,
        order: *const sz_size_t,
        count: sz_size_t,
        rows: *const sz_strings_rows_t,
        longest: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_strings_escape_best(
        strings: *const sz_strings_view_t,
        order: *const sz_size_t,
        count: sz_size_t,
        kind: EscapeKind,
        allocator: *mut sz_allocator_t,
        tape: *mut sz_strings_span_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_strings_unescape_best(
        strings: *const sz_strings_view_t,
        order: *const sz_size_t,
        count: sz_size_t,
        kind: EscapeKind,
        allocator: *mut sz_allocator_t,
        tape: *mut sz_strings_span_t,
        malformed_index: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_strings_lookup_best(
        strings: *const sz_strings_view_t,
        order: *const sz_size_t,
        count: sz_size_t,
        lut: *const u8,
        allocator: *mut sz_allocator_t,
        tape: *mut sz_strings_span_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_strings_normalize_best(
        strings: *const sz_strings_view_t,
        order: *const sz_size_t,
        count: sz_size_t,
        form: NormalForm,
        allocator: *mut sz_allocator_t,
        tape: *mut sz_strings_span_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_strings_split_best(
        text: *const u8,
        length: sz_size_t,
        delimiters: *const Byteset,
        kind: SplitKind,
        allocator: *mut sz_allocator_t,
        tape: *mut sz_strings_span_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_strings_split_lines_best(
        text: *const u8,
        length: sz_size_t,
        allocator: *mut sz_allocator_t,
        tape: *mut sz_strings_span_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_strings_join_best(
        columns: *const sz_strings_view_t,
        columns_count: sz_size_t,
        rows: sz_size_t,
        frames: *const sz_string_view_t,
        between: sz_string_view_t,
        output: *mut u8,
        length: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;

    pub(crate) fn sz_find_best(
        haystack: *const c_void,
        haystack_length: sz_size_t,
        needle: *const c_void,
        needle_length: sz_size_t,
        found: *mut *const c_void,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_find_strings_best(
        strings: *const sz_strings_view_t,
        order: *const sz_size_t,
        count: sz_size_t,
        needle: *const u8,
        needle_length: sz_size_t,
        positions: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_rfind_best(
        haystack: *const c_void,
        haystack_length: sz_size_t,
        needle: *const c_void,
        needle_length: sz_size_t,
        found: *mut *const c_void,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_find_byteset_best(
        haystack: *const c_void,
        haystack_length: sz_size_t,
        byteset: *const c_void,
        found: *mut *const c_void,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_rfind_byteset_best(
        haystack: *const c_void,
        haystack_length: sz_size_t,
        byteset: *const c_void,
        found: *mut *const c_void,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;

    pub(crate) fn sz_utf8_count_best(
        text: *const c_void,
        length: sz_size_t,
        count: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_count_strings_best(
        strings: *const sz_strings_view_t,
        order: *const sz_size_t,
        count: sz_size_t,
        runes: *mut sz_size_t,
        malformed_index: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_seek_best(
        text: *const c_void,
        length: sz_size_t,
        n: sz_size_t,
        position: *mut *const c_void,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_decode_best(
        text: *const c_void,
        length: sz_size_t,
        runes: *mut u32,
        runes_capacity: sz_size_t,
        runes_count: *mut sz_size_t,
        bytes_consumed: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
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
        stream: sz_stream_t,
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
        stream: sz_stream_t,
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
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_uncased_fold_best(
        source: *const c_void,
        source_length: sz_size_t,
        target: *mut c_void,
        target_length: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_norm_best(
        source: *const c_void,
        source_length: sz_size_t,
        form: NormalForm,
        target: *mut c_void,
        target_length: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_find_denormalized_best(
        source: *const c_void,
        source_length: sz_size_t,
        form: NormalForm,
        found: *mut *const c_void,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_uncased_needle_init_best(
        needle: *const c_void,
        needle_length: sz_size_t,
        prepared: *mut Utf8UncasedNeedle<'_>,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_uncased_search_best(
        haystack: *const c_void,
        haystack_length: sz_size_t,
        needle: *const Utf8UncasedNeedle<'_>,
        found: *mut *const c_void,
        match_length: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_uncased_order_best(
        a: *const c_void,
        a_length: sz_size_t,
        b: *const c_void,
        b_length: sz_size_t,
        ordering: *mut sz_ordering_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;

    pub(crate) fn sz_utf8_wordbreaks_best(
        text: *const c_void,
        length: sz_size_t,
        lengths: *mut sz_size_t,
        capacity: sz_size_t,
        count: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_graphemes_best(
        text: *const c_void,
        length: sz_size_t,
        lengths: *mut sz_size_t,
        capacity: sz_size_t,
        count: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_sentences_best(
        text: *const c_void,
        length: sz_size_t,
        lengths: *mut sz_size_t,
        capacity: sz_size_t,
        count: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_utf8_linebreaks_best(
        text: *const c_void,
        length: sz_size_t,
        lengths: *mut sz_size_t,
        capacity: sz_size_t,
        count: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;

    pub(crate) fn sz_equal_best(
        a: *const c_void,
        b: *const c_void,
        length: sz_size_t,
        equal: *mut sz_bool_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_order_best(
        a: *const c_void,
        a_length: sz_size_t,
        b: *const c_void,
        b_length: sz_size_t,
        ordering: *mut sz_ordering_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;

    pub(crate) fn sz_bytesum_best(
        text: *const c_void,
        length: sz_size_t,
        checksum: *mut u64,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_hash_best(
        text: *const c_void,
        length: sz_size_t,
        seed: u64,
        hash: *mut u64,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_hash_strings_best(
        strings: *const sz_strings_view_t,
        order: *const sz_size_t,
        count: sz_size_t,
        seed: u64,
        hashes: *mut u64,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_hash_multiseed_best(
        text: *const c_void,
        length: sz_size_t,
        seeds: *const u64,
        seeds_count: sz_size_t,
        hashes: *mut u64,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_hash_state_init_best(
        state: *mut c_void,
        seed: u64,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_hash_state_update_best(
        state: *mut c_void,
        text: *const c_void,
        length: sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_hash_state_digest_best(
        state: *const c_void,
        hash: *mut u64,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_sha256_state_init_best(
        state: *mut c_void,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_sha256_state_update_best(
        state: *mut c_void,
        data: *const c_void,
        length: sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_sha256_state_digest_best(
        state: *const c_void,
        digest: *mut u8,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_sha256_multistate_update_best(
        states: *mut c_void,
        texts: *const sz_strings_view_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_sha256_multistate_digest_best(
        states: *const c_void,
        states_count: sz_size_t,
        digests: *mut u8,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;

    pub(crate) fn sz_aes256_key_init_best(
        key: *mut c_void,
        secret: *const u8,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_ctr_xor_best(
        key: *const c_void,
        nonce: *const u8,
        byte_offset: u64,
        text: *const c_void,
        length: sz_size_t,
        output: *mut c_void,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_key_init_best(
        key: *mut c_void,
        secret: *const u8,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
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
        stream: sz_stream_t,
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
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_encryptor_init_best(
        encryptor: *mut c_void,
        key: *const c_void,
        nonce: *const u8,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_encryptor_associate_best(
        encryptor: *mut c_void,
        text: *const c_void,
        length: sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_encryptor_update_best(
        encryptor: *mut c_void,
        text: *const c_void,
        length: sz_size_t,
        output: *mut c_void,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_encryptor_digest_best(
        encryptor: *const c_void,
        tag: *mut u8,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_decryptor_init_best(
        decryptor: *mut c_void,
        key: *const c_void,
        nonce: *const u8,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_decryptor_associate_best(
        decryptor: *mut c_void,
        text: *const c_void,
        length: sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_decryptor_update_unverified_best(
        decryptor: *mut c_void,
        text: *const c_void,
        length: sz_size_t,
        output: *mut c_void,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_aes256_gcm_decryptor_verify_best(
        decryptor: *const c_void,
        tag: *const u8,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;

    pub(crate) fn sz_sort_order_best(
        strings: *const sz_strings_view_t,
        top_count: sz_size_t,
        reverse: sz_bool_t,
        allocator: *const c_void,
        order: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_sort_uncased_order_best(
        strings: *const sz_strings_view_t,
        top_count: sz_size_t,
        reverse: sz_bool_t,
        allocator: *const c_void,
        order: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;

    pub(crate) fn sz_intersect_best(
        first_strings: *const sz_strings_view_t,
        second_strings: *const sz_strings_view_t,
        allocator: *const c_void,
        seed: u64,
        intersection_count: *mut sz_size_t,
        first_positions: *mut sz_size_t,
        second_positions: *mut sz_size_t,
        capabilities: sz_capability_t,
        stream: sz_stream_t,
    ) -> sz_status_t;

    // Cross-product engines. The capability group picks the CPU or one GPU vendor and the stream the
    // device; a null allocator resolves to the unified one of those capabilities.
    pub(crate) fn sz_levenshtein_engine_init(
        engine: *mut LevenshteinEngine,
        queries: *const sz_strings_view_t,
        symbol: LevenshteinSymbol,
        capabilities: sz_capability_t,
        allocator: *const c_void,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_levenshtein_engine_free(engine: *mut LevenshteinEngine, stream: sz_stream_t);
    pub(crate) fn sz_levenshtein_distances(
        engine: *mut LevenshteinEngine,
        candidates: *const sz_strings_view_t,
        symbol_counts: *const sz_size_t,
        order: *const sz_size_t,
        bound: sz_size_t,
        distances: *mut sz_size_t,
        distances_stride: sz_size_t,
        stream: sz_stream_t,
    ) -> sz_status_t;

    pub(crate) fn sz_ngrams_engine_init(
        engine: *mut NgramsEngine,
        queries: *const sz_strings_view_t,
        window_widths: *const sz_size_t,
        window_widths_count: sz_size_t,
        candidates_budget: sz_size_t,
        capabilities: sz_capability_t,
        allocator: *const c_void,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_ngrams_engine_free(engine: *mut NgramsEngine, stream: sz_stream_t);
    pub(crate) fn sz_ngrams_overlaps(
        engine: *mut NgramsEngine,
        candidates: *const sz_strings_view_t,
        radius: f32,
        shares: *mut f32,
        shares_query_stride: sz_size_t,
        shares_candidate_stride: sz_size_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_ngrams_sketches(
        engine: *mut NgramsEngine,
        texts: *const sz_strings_view_t,
        slots: sz_size_t,
        sketches: *mut u8,
        sketches_stride: sz_size_t,
        stream: sz_stream_t,
    ) -> sz_status_t;

    pub(crate) fn sz_substrings_engine_init(
        engine: *mut SubstringsEngine,
        needles: *const sz_strings_view_t,
        case_sensitivity: CaseSensitivity,
        overlap_policy: SubstringsOverlapPolicy,
        hot_states: sz_size_t,
        matches_budget: sz_size_t,
        haystacks_budget: sz_size_t,
        capabilities: sz_capability_t,
        allocator: *const c_void,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_substrings_engine_free(engine: *mut SubstringsEngine, stream: sz_stream_t);
    pub(crate) fn sz_substrings_counts(
        engine: *mut SubstringsEngine,
        haystacks: *const sz_strings_view_t,
        counts: *mut sz_size_t,
        counts_stride: sz_size_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_substrings_find(
        engine: *mut SubstringsEngine,
        haystacks: *const sz_strings_view_t,
        matches: *mut Match,
        matches_capacity: sz_size_t,
        matches_offsets: *mut sz_size_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_substrings_replace(
        engine: *mut SubstringsEngine,
        haystacks: *const sz_strings_view_t,
        replacements: *const sz_strings_view_t,
        target: *mut u8,
        target_capacity: sz_size_t,
        offsets: *mut sz_size_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    pub(crate) fn sz_substrings_bm25_scores(
        engine: *mut SubstringsEngine,
        haystacks: *const sz_strings_view_t,
        document_lengths: *const f32,
        parameters: *const Bm25Params,
        needle_weights: *const f32,
        scores: *mut f32,
        scores_stride: sz_size_t,
        stream: sz_stream_t,
    ) -> sz_status_t;

}

#[allow(non_camel_case_types)]
pub(crate) type sz_allocate_t =
    unsafe extern "C" fn(bytes: sz_size_t, handle: *mut c_void, stream: sz_stream_t) -> *mut c_void;
#[allow(non_camel_case_types)]
pub(crate) type sz_reallocate_t = unsafe extern "C" fn(
    pointer: *mut c_void,
    old_bytes: sz_size_t,
    new_bytes: sz_size_t,
    handle: *mut c_void,
    stream: sz_stream_t,
) -> *mut c_void;
#[allow(non_camel_case_types)]
pub(crate) type sz_free_t =
    unsafe extern "C" fn(pointer: *mut c_void, bytes: sz_size_t, handle: *mut c_void, stream: sz_stream_t);

/// Mirror of `sz_allocator_t`; C selects the allocation policy and each block keeps its allocator.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct sz_allocator_t {
    pub(crate) allocate: Option<sz_allocate_t>,
    pub(crate) reallocate: Option<sz_reallocate_t>,
    pub(crate) free: Option<sz_free_t>,
    pub(crate) handle: *mut c_void,
}

/// Mirror of `sz_string_view_t`, one of which an accessor writes per index it is asked for.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct sz_string_view_t {
    pub(crate) start: *const u8,
    pub(crate) length: sz_size_t,
}

/// Mirror of `sz_strings_get_t`, which the C side calls once per block of up to 64 indices.
#[allow(non_camel_case_types)]
pub(crate) type sz_strings_get_t = unsafe extern "C" fn(
    handle: *const c_void,
    indices: *const sz_size_t,
    count: sz_size_t,
    views: *mut sz_string_view_t,
);

/// Mirror of `sz_strings_layout_t`, naming the member of the `data` union to read a collection by.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum sz_strings_layout_t {
    U32Tape = 0,
    U64Tape = 1,
    Accessor = 2,
}

/// Mirror of the `accessor` member of the `sz_strings_view_t` union.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct sz_strings_accessor_t {
    pub(crate) get: sz_strings_get_t,
    pub(crate) handle: *const c_void,
}

/// Mirror of the `u32_tape` and `u64_tape` members of the `sz_strings_view_t` and
/// `sz_strings_span_t` unions, whose offsets are 32-bit or 64-bit as the `layout` beside them says.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct sz_strings_tape_t {
    pub(crate) data: *mut u8,
    pub(crate) offsets: *mut c_void,
}

/// Mirror of the `data` union of `sz_strings_view_t`.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union sz_strings_data_t {
    pub(crate) accessor: sz_strings_accessor_t,
    pub(crate) tape: sz_strings_tape_t,
}

/// Mirror of C's `sz_strings_view_t`, a borrowed collection the C side only reads, which a
/// [`StringsView`] wraps.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct sz_strings_view_t {
    pub(crate) layout: sz_strings_layout_t,
    pub(crate) count: sz_size_t,
    pub(crate) data: sz_strings_data_t,
}

/// Mirror of `sz_strings_span_t`, the tape every conversion writes, which is never an accessor.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct sz_strings_span_t {
    pub(crate) layout: sz_strings_layout_t,
    pub(crate) count: sz_size_t,
    pub(crate) tape: sz_strings_tape_t,
}

/// Mirror of `sz_strings_rows_t`, NUL-padded rows of `width` bytes, `stride` bytes apart.
#[allow(non_camel_case_types)]
#[repr(C)]
pub(crate) struct sz_strings_rows_t {
    pub(crate) data: *mut u8,
    pub(crate) stride: sz_size_t,
    pub(crate) width: sz_size_t,
}

// The C layouts: a 4-byte enum, then the count and two pointers, 32 bytes on 64-bit targets.
const _: () = {
    use core::mem::{offset_of, size_of};

    let word = size_of::<usize>();
    assert!(size_of::<sz_strings_layout_t>() == 4);
    assert!(offset_of!(sz_strings_view_t, count) == word && offset_of!(sz_strings_view_t, data) == 2 * word);
    assert!(size_of::<sz_strings_view_t>() == 4 * word && size_of::<sz_strings_span_t>() == 4 * word);
    assert!(offset_of!(sz_strings_span_t, tape) == 2 * word);
    assert!(size_of::<sz_allocator_t>() == 4 * word && size_of::<sz_strings_rows_t>() == 3 * word);
};

/// Binds `items` into a `sz_strings_view_t` that lives for the span of `call`, reading each one's
/// bytes in place through `key`, which a [`StringsView`] cannot carry.
///
/// The C side calls `key` from inside `call`, where a panic cannot unwind and aborts the process.
pub(crate) fn with_strings_by<Element, Key, Return>(
    items: &[Element],
    key: Key,
    call: impl FnOnce(&sz_strings_view_t) -> Return,
) -> Return
where
    Key: Fn(&Element) -> &[u8],
{
    let accessed = (items, key);
    let strings = sz_strings_view_t {
        layout: sz_strings_layout_t::Accessor,
        count: items.len(),
        data: sz_strings_data_t {
            accessor: sz_strings_accessor_t {
                get: strings_get::<Element, Key>,
                handle: &accessed as *const (&[Element], Key) as *const c_void,
            },
        },
    };
    call(&strings)
}

unsafe extern "C" fn strings_get<Element, Key>(
    handle: *const c_void,
    indices: *const sz_size_t,
    count: sz_size_t,
    views: *mut sz_string_view_t,
) where
    Key: Fn(&Element) -> &[u8],
{
    unsafe {
        let (items, key) = &*(handle as *const (&[Element], Key));
        for position in 0..count {
            let bytes = key(&items[*indices.add(position)]);
            views.add(position).write(sz_string_view_t { start: bytes.as_ptr(), length: bytes.len() });
        }
    }
}

/// Trait for unary string operations that only operate on `self` without needle parameters.
/// These operations include hash computation and byte sum calculation.
///
/// # Examples
///
/// Basic usage on a byte slice:
///
/// ```
/// use stringzilla::StringZillableUnary;
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
    /// use stringzilla::StringZillableUnary;
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
    /// use stringzilla::StringZillableUnary;
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
    /// use stringzilla::StringZillableUnary;
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
    /// use stringzilla::StringZillableUnary;
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
    /// use stringzilla::StringZillableUnary;
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
