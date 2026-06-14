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
        .files([
            "c/stringzilla/runtime.c",
            "c/stringzilla/compare.c",
            "c/stringzilla/memory.c",
            "c/stringzilla/hash.c",
            "c/stringzilla/find.c",
            "c/stringzilla/sort.c",
            "c/stringzilla/intersect.c",
            "c/stringzilla/utf8_iterate.c",
            "c/stringzilla/utf8_case_fold.c",
            "c/stringzilla/utf8_norm.c",
            "c/stringzilla/utf8_case_insensitive.c",
        ])
        .include("include")
        .include("c/stringzilla") // for the same-directory `dispatch.h`
        .warnings(false)
        .define("SZ_DYNAMIC_DISPATCH", "1")
        .define("SZ_AVOID_LIBC", "0")
        .define("SZ_DEBUG", "0")
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
            "SZ_USE_SVE2AES",
            "SZ_USE_SVE2",
            "SZ_USE_SVE",
            "SZ_USE_NEONSHA",
            "SZ_USE_NEONAES",
            "SZ_USE_NEON",
        ],
        "x86_64" => vec![
            //
            "SZ_USE_ICELAKE",
            "SZ_USE_SKYLAKE",
            "SZ_USE_HASWELL",
            "SZ_USE_GOLDMONT",
            "SZ_USE_WESTMERE",
        ],
        "riscv64" => vec!["SZ_USE_RVV"],
        "loongarch64" => vec!["SZ_USE_LASX"],
        "powerpc64" => vec!["SZ_USE_POWERVSX"],
        _ => vec![],
    };

    // Check environment variables to allow users to disable specific backends
    // Example: SZ_USE_NEON=0 SZ_USE_SVE=0 cargo build
    for flag in flags_to_try.iter() {
        let enabled = match env::var(flag) {
            Ok(val) => val != "0" && val.to_lowercase() != "false",
            Err(_) => true, // Default to enabled if not specified
        };

        if enabled {
            build.define(flag, "1");
            flags.insert(flag.to_string(), true);
        } else {
            build.define(flag, "0");
            flags.insert(flag.to_string(), false);
            println!("cargo:warning=Disabled {} via environment variable", flag);
        }
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

    // Only re-run the C build when the C sources or headers change. The Rust source (`rust/stringzilla.rs`)
    // does not feed this compilation, so listing it here forced a full ~55s C rebuild on every Rust-only edit.
    println!("cargo:rerun-if-changed=c/stringzilla");
    println!("cargo:rerun-if-changed=include/stringzilla");

    // Rerun if SIMD backend environment variables change
    for flag in flags_to_try.iter() {
        println!("cargo:rerun-if-env-changed={}", flag);
    }

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
        // The per-capability providers pull in fork_union directly for `fu::basic_pool_t`; keep NUMA off so the
        // build does not require libnuma (matches the `stringzillas.cuh` default and the CMake build).
        .define("FU_ENABLE_NUMA", "0");

    // Nvidia GPU backend
    if is_cuda {
        build.cuda(true);
        // nvcc rejects host compilers newer than the version it supports (CUDA 12.x caps out at GCC 14). Honor the
        // standard CUDAHOSTCXX so the caller can point nvcc at a compatible host compiler, mirroring CMake's
        // CMAKE_CUDA_HOST_COMPILER. Example: `CUDAHOSTCXX=g++-14 cargo build --features cuda`.
        if let Ok(host_cxx) = env::var("CUDAHOSTCXX") {
            build.flag("-ccbin");
            build.flag(&host_cxx);
        }
        build.files([
            // C-API entry translation units.
            "c/stringzillas/runtime.cu",
            "c/stringzillas/levenshtein.cu",
            "c/stringzillas/needleman_wunsch.cu",
            "c/stringzillas/smith_waterman.cu",
            "c/stringzillas/fingerprints.cu",
            // Per-capability providers that emit each tier's kernels once (the entry TUs above only declare them).
            "c/stringzillas/levenshtein_cuda.cu",
            "c/stringzillas/levenshtein_kepler.cu",
            "c/stringzillas/levenshtein_hopper.cu",
            "c/stringzillas/needleman_wunsch_cuda.cu",
            "c/stringzillas/needleman_wunsch_hopper.cu",
            "c/stringzillas/smith_waterman_cuda.cu",
            "c/stringzillas/smith_waterman_hopper.cu",
        ]);
        build.define("SZ_USE_CUDA", "1");
        build.define("SZ_USE_ROCM", "0");
        build.flag("-std=c++20");
        build.flag("--expt-relaxed-constexpr");
        build.flag("-arch=sm_90a");
        // The kernels use the CUDA driver API (cuLaunchKernel, cuEventElapsedTime, cuFuncSetAttribute), so a binary
        // linking this crate needs libcuda in addition to the cudart that `build.cuda(true)` already links.
        println!("cargo:rustc-link-lib=dylib=cuda");
    }
    // AMD GPU backend
    else if is_rocm {
        build.cpp(true);
        build.files([
            "c/stringzillas/runtime.cu",
            "c/stringzillas/levenshtein.cu",
            "c/stringzillas/needleman_wunsch.cu",
            "c/stringzillas/smith_waterman.cu",
            "c/stringzillas/fingerprints.cu",
        ]);
        build.define("SZ_USE_CUDA", "0");
        build.define("SZ_USE_ROCM", "1");
        build.flag("-std=c++20");
        // TODO: Add proper HIP/ROCm compiler support
    }
    // Multi-core CPU backend
    else if is_cpus {
        build.cpp(true);
        build.files([
            // C-API entry translation units.
            "c/stringzillas/runtime.cpp",
            "c/stringzillas/levenshtein.cpp",
            "c/stringzillas/needleman_wunsch.cpp",
            "c/stringzillas/smith_waterman.cpp",
            "c/stringzillas/fingerprints.cpp",
            // Per-capability providers that emit each ISA's single-pair SIMD core once (the entry TUs above only
            // declare them). The off-platform files compile to empty objects via their internal SZ_USE_* guards.
            "c/stringzillas/levenshtein_serial.cpp",
            "c/stringzillas/levenshtein_icelake.cpp",
            "c/stringzillas/needleman_wunsch_serial.cpp",
            "c/stringzillas/needleman_wunsch_icelake.cpp",
            "c/stringzillas/needleman_wunsch_haswell.cpp",
            "c/stringzillas/needleman_wunsch_neon.cpp",
            "c/stringzillas/smith_waterman_serial.cpp",
            "c/stringzillas/smith_waterman_icelake.cpp",
            "c/stringzillas/smith_waterman_haswell.cpp",
            "c/stringzillas/smith_waterman_neon.cpp",
        ]);
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
    println!("cargo:rerun-if-changed=c/stringzillas");
    println!("cargo:rerun-if-changed=include/stringzillas");
    println!("cargo:rerun-if-changed=fork_union/include/fork_union.hpp");
}
