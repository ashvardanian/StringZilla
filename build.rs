use std::collections::HashMap;
use std::env;

fn main() {
    // Build stringzilla (always included, single-string operations)
    let serial_flags = build_stringzilla();

    // Build stringzillas (multi-string operations) if any feature is enabled
    if env::var("CARGO_FEATURE_CPUS").is_ok()
        || env::var("CARGO_FEATURE_CUDA").is_ok()
        || env::var("CARGO_FEATURE_ROCM").is_ok()
    {
        build_stringzillas(&serial_flags);
    }
}

/// Build the StringZilla C library with dynamic SIMD dispatching
/// and returns a dictionary of enabled compilation flags to be reused for
/// parallel backends (e.g., StringZillas).
fn build_stringzilla() -> HashMap<String, bool> {
    let mut flags = HashMap::<String, bool>::new();
    let mut build = cc::Build::new();
    build
        .file("c/stringzilla.c")
        .include("include")
        .warnings(false)
        .define("SZ_DYNAMIC_DISPATCH", "1")
        .define("SZ_AVOID_LIBC", "0")
        .define("SZ_DEBUG", "0")
        .flag("-O2")
        .flag("-std=c99") // Enforce C99 standard
        .flag_if_supported("-fdiagnostics-color=always")
        .flag_if_supported("-fPIC");

    // Cargo will set different environment variables that we can use to properly configure the build.
    // https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts
    // https://doc.rust-lang.org/reference/conditional-compilation.html#r-cfg.target_endian
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_endian = env::var("CARGO_CFG_TARGET_ENDIAN").unwrap_or_default();
    let target_bits = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap_or_default();

    // Set endian-specific macro
    if target_endian == "big" {
        build.define("SZ_IS_BIG_ENDIAN_", "1");
        flags.insert("SZ_IS_BIG_ENDIAN_".to_string(), true);
    } else {
        build.define("SZ_IS_BIG_ENDIAN_", "0");
        flags.insert("SZ_IS_BIG_ENDIAN_".to_string(), false);
    }

    if target_arch == "x86_64" && target_bits == "64" {
        build.define("SZ_IS_64BIT_X86_", "1");
        build.define("SZ_IS_64BIT_ARM_", "0");
        flags.insert("SZ_IS_64BIT_X86_".to_string(), true);
        flags.insert("SZ_IS_64BIT_ARM_".to_string(), false);
    } else if target_arch == "aarch64" && target_bits == "64" {
        build.define("SZ_IS_64BIT_X86_", "0");
        build.define("SZ_IS_64BIT_ARM_", "1");
        flags.insert("SZ_IS_64BIT_X86_".to_string(), false);
        flags.insert("SZ_IS_64BIT_ARM_".to_string(), true);
    } else {
        build.define("SZ_IS_64BIT_X86_", "0");
        build.define("SZ_IS_64BIT_ARM_", "0");
        flags.insert("SZ_IS_64BIT_X86_".to_string(), false);
        flags.insert("SZ_IS_64BIT_ARM_".to_string(), false);
    }

    // At start we will try compiling with all SIMD backends enabled
    // https://doc.rust-lang.org/reference/conditional-compilation.html#target_arch
    let flags_to_try = match target_arch.as_str() {
        "arm" | "aarch64" => vec![
            //
            "SZ_USE_SVE2_AES",
            "SZ_USE_SVE2",
            "SZ_USE_SVE",
            "SZ_USE_NEON_AES",
            "SZ_USE_NEON",
        ],
        "x86_64" => vec![
            //
            "SZ_USE_ICE",
            "SZ_USE_SKYLAKE",
            "SZ_USE_HASWELL",
            "SZ_USE_GOLDMONT",
            "SZ_USE_WESTMERE",
        ],
        _ => vec![],
    };
    for flag in flags_to_try.iter() {
        build.define(flag, "1");
        flags.insert(flag.to_string(), true);
    }

    // If that fails, we will try disabling them one by one
    if build.try_compile("stringzilla").is_err() {
        print!("cargo:warning=Failed to compile with all SIMD backends...");

        for flag in flags_to_try.iter() {
            build.define(flag, "0");
            flags.insert(flag.to_string(), false);
            if build.try_compile("stringzilla").is_ok() {
                break;
            }

            // Print the failed configuration
            println!(
                "cargo:warning=Failed to compile after disabling {}, trying next configuration...",
                flag
            );
        }
    }

    println!("cargo:rerun-if-changed=c/stringzilla.c");
    println!("cargo:rerun-if-changed=rust/stringzilla.rs");
    println!("cargo:rerun-if-changed=include/stringzilla/stringzilla.h");

    // Constituent parts:
    println!("cargo:rerun-if-changed=include/stringzilla/compare.h");
    println!("cargo:rerun-if-changed=include/stringzilla/find.h");
    println!("cargo:rerun-if-changed=include/stringzilla/hash.h");
    println!("cargo:rerun-if-changed=include/stringzilla/memory.h");
    println!("cargo:rerun-if-changed=include/stringzilla/similarities.h");
    println!("cargo:rerun-if-changed=include/stringzilla/small_string.h");
    println!("cargo:rerun-if-changed=include/stringzilla/sort.h");
    println!("cargo:rerun-if-changed=include/stringzilla/types.h");
    flags
}

fn build_stringzillas(serial_flags: &HashMap<String, bool>) {
    let mut build = cc::Build::new();
    let is_cpus = env::var("CARGO_FEATURE_CPUS").is_ok();
    let is_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let is_rocm = env::var("CARGO_FEATURE_ROCM").is_ok();

    build
        .include("include")
        .include("fork_union/include")
        .warnings(false)
        .define("SZ_DYNAMIC_DISPATCH", "1")
        .define("SZ_AVOID_LIBC", "0")
        .define("SZ_DEBUG", "0")
        .flag("-O2");

    // Nvidia GPU backend
    if is_cuda {
        build.cuda(true);
        build.file("c/stringzillas.cu");
        build.define("SZ_USE_CUDA", "1");
        build.define("SZ_USE_ROCM", "0");
        build.flag("-std=c++20");
        build.flag("--expt-relaxed-constexpr");
        build.flag("-arch=sm_90a");
    }
    // AMD GPU backend
    else if is_rocm {
        build.cpp(true);
        build.file("c/stringzillas.cu");
        build.define("SZ_USE_CUDA", "0");
        build.define("SZ_USE_ROCM", "1");
        build.flag("-std=c++20");
        // TODO: Add proper HIP/ROCm compiler support
    }
    // Multi-core CPU backend
    else if is_cpus {
        build.cpp(true);
        build.file("c/stringzillas.cpp");
        build.define("SZ_USE_CUDA", "0");
        build.define("SZ_USE_ROCM", "0");
        build.flag("-std=c++20");
    }

    // Common flags
    build
        .flag_if_supported("-fdiagnostics-color=always")
        .flag_if_supported("-fPIC");

    // Apply the same architecture-specific flags as determined for stringzilla
    for (flag, enabled) in serial_flags.iter() {
        if *enabled {
            build.define(flag, "1");
        } else {
            build.define(flag, "0");
        }
    }
    // Try compilation with fallback (similar to stringzilla approach)
    if build.try_compile("stringzillas").is_err() {
        println!("cargo:warning=Failed to compile stringzillas with selected backend");

        // Fallback: disable GPU features and try CPU-only
        build.define("SZ_USE_CUDA", "0");
        build.define("SZ_USE_ROCM", "0");

        if build.try_compile("stringzillas").is_err() {
            panic!("Failed to compile stringzillas even with CPU-only fallback");
        }
    }

    // StringZillas-specific rerun triggers
    println!("cargo:rerun-if-changed=c/stringzillas.cu");
    println!("cargo:rerun-if-changed=c/stringzillas.cuh");
    println!("cargo:rerun-if-changed=include/stringzillas/stringzillas.h");
    println!("cargo:rerun-if-changed=include/stringzillas/fingerprints.hpp");
    println!("cargo:rerun-if-changed=include/stringzillas/fingerprints.cuh");
    println!("cargo:rerun-if-changed=include/stringzillas/similarities.hpp");
    println!("cargo:rerun-if-changed=include/stringzillas/similarities.cuh");
    println!("cargo:rerun-if-changed=fork_union/include/fork_union.hpp");
}
