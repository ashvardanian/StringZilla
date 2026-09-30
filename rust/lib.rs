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
//!   vendor's devices count and the engines' `new_on` constructors can prepare a batch on one.
//!
//! Every call dispatches on the CPU's
//! [`Device::capabilities_enabled`](stringzilla::Device::capabilities_enabled), which
//! [`Device::capabilities_enable`](stringzilla::Device::capabilities_enable) narrows for the whole
//! process.
//!
//! File: rust/lib.rs
//! Author: Ash Vardanian

#![cfg_attr(not(feature = "std"), no_std)]

pub mod stringzilla;

// Convenience alias for the shorter name.
pub use stringzilla as sz;
