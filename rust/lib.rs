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
//! - `dynamic-dispatch`: compile every ISA tier and pick one at load through a dispatch table;
//!   without it the tier is resolved at compile time and baked in.
//! - `cuda`: compile the CUDA backend, which is what the engines' `new_on_gpu` constructors need.
//!
//! File: rust/lib.rs
//! Author: Ash Vardanian

#![cfg_attr(not(feature = "std"), no_std)]

pub mod stringzilla;

// Convenience alias for the shorter name.
pub use stringzilla as sz;
