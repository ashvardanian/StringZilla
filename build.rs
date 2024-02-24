use std::env;

fn main() {
    let mut build = cc::Build::new();
    build
        .file("c/lib.c")
        .include("include")
        .warnings(false)
        .flag_if_supported("-std=c99")
        .flag_if_supported("-fPIC");

    // Cargo will set different environment variables that we can use to properly configure the build.
    // https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_endian = env::var("CARGO_CFG_TARGET_ENDIAN").unwrap_or_default();

    // To get the operating system we can use the TARGET environment variable.
    // To check the list of available targets, run `rustc --print target-list`.
    let target = env::var("TARGET").unwrap_or_default();

    if target.contains("linux") {
        build.flag_if_supported("-fdiagnostics-color=always");
        build.flag_if_supported("-O3");
        build.flag_if_supported("-pedantic");

        // Set architecture-specific flags and macros
        if target_arch == "x86_64" {
            build.define("SZ_USE_X86_AVX512", "1");
            build.define("SZ_USE_X86_AVX2", "1");
        } else {
            build.define("SZ_USE_X86_AVX512", "0");
            build.define("SZ_USE_X86_AVX2", "0");
        }

        if target_arch == "aarch64" {
            build.flag_if_supported("-march=armv8-a+simd");
            build.define("SZ_USE_ARM_SVE", "1");
            build.define("SZ_USE_ARM_NEON", "1");
        } else {
            build.define("SZ_USE_ARM_SVE", "0");
            build.define("SZ_USE_ARM_NEON", "0");
        }
    } else if target.contains("darwin") {
        build.flag_if_supported("-fcolor-diagnostics");
        build.flag_if_supported("-O3");
        build.flag_if_supported("-pedantic");

        if target_arch == "x86_64" {
            // Assuming no AVX-512 support for Darwin as per setup.py logic
            build.define("SZ_USE_X86_AVX512", "0");
            build.define("SZ_USE_X86_AVX2", "1");
        } else {
            build.define("SZ_USE_X86_AVX512", "0");
            build.define("SZ_USE_X86_AVX2", "0");
        }

        if target_arch == "aarch64" {
            build.define("SZ_USE_ARM_SVE", "0"); // Assuming no SVE support for Darwin
            build.define("SZ_USE_ARM_NEON", "1");
        } else {
            build.define("SZ_USE_ARM_SVE", "0");
            build.define("SZ_USE_ARM_NEON", "0");
        }
    } else if target.contains("windows") {
        // Set architecture-specific flags and macros
        if target_arch == "x86_64" {
            build.define("SZ_USE_X86_AVX512", "1");
            build.define("SZ_USE_X86_AVX2", "1");
        } else {
            build.define("SZ_USE_X86_AVX512", "0");
            build.define("SZ_USE_X86_AVX2", "0");
        }
    }

    // Set endian-specific macro
    if target_endian == "big" {
        build.define("SZ_DETECT_BIG_ENDIAN", "1");
    } else {
        build.define("SZ_DETECT_BIG_ENDIAN", "0");
    }

    build.compile("stringzilla");

    println!("cargo:rerun-if-changed=c/lib.c");
    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=include/stringzilla/stringzilla.h");
}
