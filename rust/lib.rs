#![cfg_attr(not(feature = "std"), no_std)]
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

pub mod stringzilla;

#[cfg(any(feature = "cpus", feature = "cuda", feature = "rocm"))]
pub mod stringzillas;

// Convenience aliases for shorter names
pub use stringzilla as sz;
#[cfg(any(feature = "cpus", feature = "cuda", feature = "rocm"))]
pub use stringzillas as szs;
