#![cfg_attr(all(not(test), not(any(feature = "cpus", feature = "cuda", feature = "rocm"))), no_std)]
#![doc = r"
# StringZilla

Fast string processing library with SIMD and GPU acceleration.

This crate provides two main modules:
- `stringzilla` (alias `sz`): Single-string operations  
- `stringzillas` (alias `szs`): Multi-string parallel operations (requires features)

## Features
- `cpus`: Enable multi-threaded CPU backend
- `cuda`: Enable CUDA GPU backend  
- `rocm`: Enable ROCm GPU backend
"]

/// Core single-string operations with SIMD acceleration.
///
/// Provides fast string search, comparison, hashing, and manipulation
/// functions optimized with SWAR and SIMD instructions.
pub mod stringzilla;

/// High-performance parallel string algorithms with CPU/GPU acceleration.
///
/// Requires `cpus`, `cuda`, or `rocm` features. Provides:
/// - Levenshtein distances (binary and UTF-8)  
/// - Needleman-Wunsch global alignment
/// - Smith-Waterman local alignment
/// - Min-Hash fingerprinting
#[cfg(any(feature = "cpus", feature = "cuda", feature = "rocm"))]
pub mod stringzillas;

// Convenience aliases for shorter names
pub use stringzilla as sz;
#[cfg(any(feature = "cpus", feature = "cuda", feature = "rocm"))]
pub use stringzillas as szs;
