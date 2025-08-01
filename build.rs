use std::env;

fn main() {
    let mut build = cc::Build::new();
    build
        .file("c/lib.c")
        .include("include")
        .warnings(false)
        .define("SZ_DYNAMIC_DISPATCH", "1")
        .define("SZ_AVOID_LIBC", "0")
        .define("SZ_DEBUG", "0")
        .flag("-O3")
        .flag("-std=c99") // Enforce C99 standard
        .flag_if_supported("-fdiagnostics-color=always")
        .flag_if_supported("-fPIC");

    // Cargo will set different environment variables that we can use to properly configure the build.
    // https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_endian = env::var("CARGO_CFG_TARGET_ENDIAN").unwrap_or_default();

    // Set endian-specific macro
    if target_endian == "big" {
        build.define("SZ_DETECT_BIG_ENDIAN", "1");
    } else {
        build.define("SZ_DETECT_BIG_ENDIAN", "0");
    }

    if target_arch == "x86_64" {
        build.define("SZ_IS_64BIT_X86_", "1");
        build.define("SZ_IS_64BIT_ARM_", "0");
    } else if target_arch == "aarch64" {
        build.define("SZ_IS_64BIT_X86_", "0");
        build.define("SZ_IS_64BIT_ARM_", "1");
    }

    // At start we will try compiling with all SIMD backends enabled
    let flags_to_try = match target_arch.as_str() {
        "arm" | "aarch64" => vec![
            //
            "SZ_USE_SVE2",
            "SZ_USE_SVE",
            "SZ_USE_NEON",
        ],
        _ => vec![
            //
            "SZ_USE_ICE",
            "SZ_USE_SKYLAKE",
            "SZ_USE_HASWELL",
        ],
    };
    for flag in flags_to_try.iter() {
        build.define(flag, "1");
    }

    // If that fails, we will try disabling them one by one
    if build.try_compile("stringzilla").is_err() {
        print!("cargo:warning=Failed to compile with all SIMD backends...");

        for flag in flags_to_try.iter() {
            build.define(flag, "0");
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
}
