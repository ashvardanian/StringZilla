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
        .include("include")
        .include("c/stringzilla") // for the same-directory `dispatch.h`
        .warnings(false)
        .define("SZ_DEBUG", "0")
        .flag("-std=c99") // Enforce C99 standard
        .flag_if_supported("-fdiagnostics-color=always")
        .flag_if_supported("-fPIC");
    for flag in no_builtin_flags() {
        build.flag(flag);
    }

    // Dispatch model, selected by the `dynamic-dispatch` feature:
    //  - ON  (default): compile the per-domain dispatch shims into a runtime table that picks the best ISA
    //    tier at load - mirrors CMake's `stringzilla_shared`. Flexible, one indirection per call.
    //  - OFF: compile a single amalgamation TU (`#include <stringzilla/stringzilla.h>`) that resolves each
    //    public function to one ISA tier at compile time and exports it via `SZ_EXPORT` (see `types.h`).
    //    No table/indirection; the tier is baked in (less portable, faster on some workloads).
    if env::var("CARGO_FEATURE_DYNAMIC_DISPATCH").is_ok() {
        build.define("SZ_DYNAMIC_DISPATCH", "1");
        build.files([
            "c/stringzilla/runtime.c",
            "c/stringzilla/compare.c",
            "c/stringzilla/memory.c",
            "c/stringzilla/hash.c",
            "c/stringzilla/find.c",
            "c/stringzilla/sort.c",
            "c/stringzilla/intersect.c",
            "c/stringzilla/utf8_runes.c",
            "c/stringzilla/utf8_tokens.c",
            "c/stringzilla/utf8_wordbreaks.c",
            "c/stringzilla/utf8_graphemes.c",
            "c/stringzilla/utf8_sentences.c",
            "c/stringzilla/utf8_linebreaks.c",
            "c/stringzilla/utf8_uncased_fold.c",
            "c/stringzilla/utf8_norm.c",
            "c/stringzilla/utf8_uncased.c",
        ]);
    } else {
        // One translation unit includes the umbrella header once; generated into `OUT_DIR` so there is no
        // checked-in source (same pattern as the relaxed-SIMD probe). Exactly one TU avoids duplicate symbols.
        build.define("SZ_DYNAMIC_DISPATCH", "0");
        build.define("SZ_EXPORT", "1");
        let amalgam_path = std::path::Path::new(&env::var("OUT_DIR").unwrap_or_default()).join("sz_stringzilla.c");
        std::fs::write(&amalgam_path, "#include <stringzilla/stringzilla.h>\n").expect("write amalgamation TU");
        build.file(&amalgam_path);
    }

    // Cargo will set different environment variables that we can use to properly configure the build.
    // https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts
    // https://doc.rust-lang.org/reference/conditional-compilation.html#r-cfg.target_endian
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_endian = env::var("CARGO_CFG_TARGET_ENDIAN").unwrap_or_default();
    let target_bits = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    let avoid_libc = target_os == "unknown" || target_os.is_empty();
    build.define("SZ_AVOID_LIBC", if avoid_libc { "1" } else { "0" });

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

    // WebAssembly selects SIMD through instruction flags (native ISAs use per-function `target`
    // attributes instead). `-msimd128` is the baseline; relaxed-SIMD is one tier up, enabled only when a
    // tiny probe confirms the toolchain emits the relaxed intrinsics and `SZ_USE_V128RELAXED` is not
    // forced off. The flags define `__wasm_simd128__` / `__wasm_relaxed_simd__`, which `types.h` maps to
    // `SZ_USE_V128` / `SZ_USE_V128RELAXED` (kept in `flags_to_try` below for the env override + fallback).
    let is_wasm = target_arch == "wasm32" || target_arch == "wasm64";
    let wasm_relaxed = is_wasm
        && env::var("SZ_USE_V128RELAXED").map_or(true, |v| v != "0" && v.to_lowercase() != "false")
        && wasm_relaxed_simd_supported();
    if is_wasm {
        build.flag("-msimd128");
        if wasm_relaxed {
            build.flag("-mrelaxed-simd");
        }
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
        "wasm32" | "wasm64" => {
            if wasm_relaxed {
                vec!["SZ_USE_V128RELAXED", "SZ_USE_V128"]
            } else {
                vec!["SZ_USE_V128"]
            }
        }
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
    // `SZ_USE_V128RELAXED` decides the wasm relaxed tier before `flags_to_try` is built, so when relaxed
    // is off it is absent from the loop above; track it (and its baseline `SZ_USE_V128`) explicitly so
    // toggling either between builds that share a target directory rebuilds the C objects.
    println!("cargo:rerun-if-env-changed=SZ_USE_V128");
    println!("cargo:rerun-if-env-changed=SZ_USE_V128RELAXED");

    flags
}

/// Probe whether the toolchain can emit WebAssembly relaxed-SIMD (`-mrelaxed-simd`). Compiles a tiny
/// snippet that uses the relaxed intrinsics the v128relaxed kernels depend on; success means the flag and
/// the `__wasm_relaxed_simd__` builtins are available. The snippet is embedded and written into `OUT_DIR`
/// (the `cc` crate compiles files, not in-memory strings), so there is no checked-in probe source.
fn wasm_relaxed_simd_supported() -> bool {
    let probe_source = concat!(
        "#include <wasm_simd128.h>\n",
        "v128_t sz_probe_swizzle(v128_t a, v128_t b) { return wasm_i8x16_relaxed_swizzle(a, b); }\n",
        "v128_t sz_probe_dot(v128_t a, v128_t b, v128_t c) {\n",
        "    return wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a, b, c);\n",
        "}\n",
    );
    let probe_path = std::path::Path::new(&env::var("OUT_DIR").unwrap_or_default()).join("sz_wasm_relaxed_probe.c");
    if std::fs::write(&probe_path, probe_source).is_err() {
        return false;
    }

    let mut probe = cc::Build::new();
    probe
        .file(&probe_path)
        .flag("-msimd128")
        .flag("-mrelaxed-simd")
        .warnings(false)
        .cargo_metadata(false);
    let supported = probe.try_compile("sz_wasm_relaxed_probe").is_ok();
    if !supported {
        println!("cargo:warning=WASM relaxed-SIMD unsupported by toolchain; using baseline SIMD128");
    }
    supported
}

/// MSVC host-compiler flags the StringZilla(s) C++ sources need on Windows: report `__cplusplus` accurately (so the
/// `sz_constexpr_if_cpp14` helpers are seen as constexpr, else C3615), use the standard-conforming preprocessor
/// (required by CCCL 3.x), and read the UTF-8 sources as UTF-8. Empty off the MSVC target env — a GCC/Clang host
/// needs none — so callers can loop unconditionally. The `cl`-direct backends apply these verbatim; the CUDA backend
/// forwards each through `-Xcompiler`. (`CARGO_CFG_TARGET_ENV`, not `cfg!`, is the target signal in a build script.)
fn msvc_cxx_flags() -> &'static [&'static str] {
    if matches!(env::var("CARGO_CFG_TARGET_ENV").as_deref(), Ok("msvc")) {
        &["/Zc:__cplusplus", "/Zc:preprocessor", "/utf-8"]
    } else {
        &[]
    }
}

/// Flags that stop the compiler from substituting its own builtins for the bytewise primitives StringZilla provides
/// — and, under `SZ_OVERRIDE_LIBC`, from lowering those implementations back into a self-recursive libc call. Mirrors
/// the "avoid builtin functions" block in CMakeLists.txt: MSVC disables intrinsic generation with `/Oi-`, GCC/Clang
/// disable the specific `mem*` builtins. Applied to every StringZilla(s) build; the CUDA backend forwards each
/// through `-Xcompiler`, exactly as it does the MSVC conformance flags.
fn no_builtin_flags() -> &'static [&'static str] {
    if matches!(env::var("CARGO_CFG_TARGET_ENV").as_deref(), Ok("msvc")) {
        &["/Oi-"]
    } else {
        &[
            "-fno-builtin-memcmp",
            "-fno-builtin-memchr",
            "-fno-builtin-memcpy",
            "-fno-builtin-memset",
        ]
    }
}

/// A fresh StringZillas build with the configuration shared by every backend: include paths, the dispatch/NUMA
/// defines, the C++20 standard, common flags, and the architecture flags inherited from the StringZilla build.
/// Each backend adds only its own sources (and, for CUDA, its nvcc flags) on top.
fn stringzillas_base_build(serial_flags: &HashMap<String, bool>) -> cc::Build {
    let mut build = cc::Build::new();
    build
        .include("include")
        .include("fork_union/include")
        .warnings(false)
        .define("SZ_DYNAMIC_DISPATCH", "1")
        .define("SZ_AVOID_LIBC", "0")
        .define("SZ_DEBUG", "0")
        // The per-capability providers pull in fork_union directly for `fu::basic_pool_t`; keep NUMA off so the
        // build does not require libnuma (matches the `stringzillas.cuh` default and the CMake build).
        .define("FU_ENABLE_NUMA", "0")
        .std("c++20")
        .flag_if_supported("-fdiagnostics-color=always")
        .flag_if_supported("-fPIC");
    // Apply the same architecture-specific flags as determined for stringzilla.
    for (flag, enabled) in serial_flags.iter() {
        build.define(flag, if *enabled { "1" } else { "0" });
    }
    build
}

/// Build the NVIDIA CUDA backend (the `.cu` sources via nvcc). Returns `Err` if the toolkit is missing or the
/// sources fail to compile; the driver-API link directive is emitted only on success.
fn try_build_stringzillas_cuda(serial_flags: &HashMap<String, bool>) -> Result<(), cc::Error> {
    let mut build = stringzillas_base_build(serial_flags);
    build.cuda(true).define("SZ_USE_CUDA", "1").define("SZ_USE_ROCM", "0");
    // nvcc rejects host compilers newer than it supports (CUDA 12.x caps at GCC 14); honor CUDAHOSTCXX so the caller
    // can point nvcc at a compatible host compiler, mirroring CMAKE_CUDA_HOST_COMPILER.
    if let Ok(host_cxx) = env::var("CUDAHOSTCXX") {
        build.flag("-ccbin").flag(&host_cxx);
    }
    // `.std()` only reaches the host compiler (`-Xcompiler -std:c++20`); nvcc's device frontend would otherwise
    // default to an older standard and `cudafe++` chokes on the C++20 device code (templated lambdas, designated
    // initializers), so set nvcc's own device standard too — as the CMake build does.
    build
        .flag("-std=c++20")
        .flag("--expt-relaxed-constexpr")
        .flag("-arch=sm_90a");
    // Forward the MSVC conformance flags and the no-builtin flags to the host compiler through nvcc.
    for flag in msvc_cxx_flags().iter().chain(no_builtin_flags()) {
        build.flag(format!("-Xcompiler={flag}"));
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
    build.try_compile("stringzillas")?;
    // Only demand libcuda once the build actually linked: the kernels use the driver API (cuLaunchKernel,
    // cuEventElapsedTime, cuFuncSetAttribute) on top of the cudart that `build.cuda(true)` already links.
    println!("cargo:rustc-link-lib=dylib=cuda");
    Ok(())
}

/// Build the AMD ROCm backend. TODO: wire up a real HIP/ROCm compiler — today this is a stub that fails to compile
/// the `.cu` sources with a plain C++ compiler, so callers fall back to the CPU-only backend.
fn try_build_stringzillas_rocm(serial_flags: &HashMap<String, bool>) -> Result<(), cc::Error> {
    let mut build = stringzillas_base_build(serial_flags);
    build.cpp(true).define("SZ_USE_CUDA", "0").define("SZ_USE_ROCM", "1");
    for flag in msvc_cxx_flags().iter().chain(no_builtin_flags()) {
        build.flag(flag);
    }
    build.files([
        "c/stringzillas/runtime.cu",
        "c/stringzillas/levenshtein.cu",
        "c/stringzillas/needleman_wunsch.cu",
        "c/stringzillas/smith_waterman.cu",
        "c/stringzillas/fingerprints.cu",
    ]);
    build.try_compile("stringzillas")
}

/// Build the CPU-only backend: the `.cpp` per-ISA single-pair cores (off-platform files compile to empty objects via
/// their internal `SZ_USE_*` guards). Used by the `cpus` feature AND as the fallback when a GPU toolkit is missing,
/// so it carries no `.cu` sources — only a host C++ compiler is needed.
fn try_build_stringzillas_cpus(serial_flags: &HashMap<String, bool>) -> Result<(), cc::Error> {
    let mut build = stringzillas_base_build(serial_flags);
    build.cpp(true).define("SZ_USE_CUDA", "0").define("SZ_USE_ROCM", "0");
    for flag in msvc_cxx_flags().iter().chain(no_builtin_flags()) {
        build.flag(flag);
    }
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
        "c/stringzillas/levenshtein_haswell.cpp",
        "c/stringzillas/levenshtein_neon.cpp",
        "c/stringzillas/levenshtein_rvv.cpp",
        "c/stringzillas/needleman_wunsch_serial.cpp",
        "c/stringzillas/needleman_wunsch_icelake.cpp",
        "c/stringzillas/needleman_wunsch_haswell.cpp",
        "c/stringzillas/needleman_wunsch_neon.cpp",
        "c/stringzillas/needleman_wunsch_rvv.cpp",
        "c/stringzillas/smith_waterman_serial.cpp",
        "c/stringzillas/smith_waterman_icelake.cpp",
        "c/stringzillas/smith_waterman_haswell.cpp",
        "c/stringzillas/smith_waterman_neon.cpp",
        "c/stringzillas/smith_waterman_rvv.cpp",
    ]);
    build.try_compile("stringzillas")
}

fn build_stringzillas(serial_flags: &HashMap<String, bool>) {
    println!("cargo:rerun-if-changed=c/stringzillas");
    println!("cargo:rerun-if-changed=include/stringzillas");
    println!("cargo:rerun-if-changed=fork_union/include/fork_union.hpp");

    // `cuda` and `rocm` both imply `cpus`. Try the requested GPU backend first; if it can't build (commonly: no GPU
    // toolkit on this machine), fall through to the CPU-only backend so the crate still works.
    let is_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let is_rocm = env::var("CARGO_FEATURE_ROCM").is_ok();
    let gpu_ok = if is_cuda {
        try_build_stringzillas_cuda(serial_flags).is_ok()
    } else if is_rocm {
        try_build_stringzillas_rocm(serial_flags).is_ok()
    } else {
        false
    };
    if gpu_ok {
        return;
    }
    if is_cuda || is_rocm {
        println!("cargo:warning=GPU backend unavailable; building CPU-only StringZillas instead");
    }
    try_build_stringzillas_cpus(serial_flags).expect("failed to compile CPU-only StringZillas");
}
