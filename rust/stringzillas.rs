//! High-performance parallel string algorithms with CPU/GPU acceleration.
//!
//! Requires `cpus`, `cuda`, or `rocm` features. Provides:
//! - Levenshtein distances (binary and UTF-8)
//! - Needleman-Wunsch global alignment
//! - Smith-Waterman local alignment
//! - Min-Hash fingerprinting
//! - Multi-pattern substring search

mod device_scope;
mod fingerprints;
mod similarities;
mod substrings;
mod types;

pub use device_scope::*;
pub use fingerprints::*;
pub use similarities::*;
pub use substrings::*;
pub use types::*;

extern crate alloc;

// The `forkunion` crate's build script compiles and links the thread-pool runtime resolving the `fu_*`
// symbols behind the StringZillas engines; referencing the crate keeps that native library in the final
// link even though all calls happen on the C++ side.
use forkunion as _;

/// Device helpers shared by the tests of more than one engine domain.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::DeviceScope;

    /// Returns a CPU-backed `DeviceScope`, or `None` after logging why. `DeviceScope::default()`
    /// picks CPU cores, which should always succeed in CI, so a failure here signals an
    /// environment problem rather than a missing GPU - every skip is logged, never swallowed.
    pub(crate) fn device_or_skip(test_name: &str) -> Option<DeviceScope> {
        match DeviceScope::default() {
            Ok(device) => Some(device),
            Err(error) => {
                println!("Skipping {test_name} - device initialization failed: {error:?}");
                None
            }
        }
    }
}
