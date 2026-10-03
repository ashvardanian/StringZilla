//! # StringZilla
//!
//! Fast string processing with SIMD and GPU acceleration.
//!
//! The crate is one module, `stringzilla`, aliased `sz`: single-string search, comparison, hashing,
//! sorting and UTF-8 segmentation, beside the stateful cross-product engines - `LevenshteinEngine`,
//! `OverlapEngine` and `SubstringsEngine` - which prepare a batch of queries once and score any
//! number of candidate batches against it.
//!
//! ## Features
//! - `std`: standard-library integration, such as `BuildSzHasher` for `HashMap`; without it the
//!   crate is `no_std`.
//! - `cuda`, `rocm`, `metal`: build the library through CMake with that GPU backend, so that
//!   vendor's devices count and make a `Stream`, on which engines, `Sequence` tapes and
//!   `UnifiedAllocator` memory are built; without the feature those report `Status::MissingGpu`.
//!
//! Every call dispatches on
//! [`Capabilities::cpu_enabled`](stringzilla::Capabilities::cpu_enabled), what this CPU runs and
//! this build compiled. Engine verbs queue on a [`Stream`](stringzilla::Stream) inside its
//! [`Stream::scope`](stringzilla::Stream::scope), which joins the stream before returning.
//!
//! File: rust/lib.rs
//! Author: Ash Vardanian

#![cfg_attr(not(feature = "std"), no_std)]

pub mod stringzilla;

// Convenience alias for the shorter name.
pub use stringzilla as sz;
