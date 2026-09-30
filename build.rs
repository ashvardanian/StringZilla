//! Cargo build script that builds the StringZilla C library through CMake, the one place that
//! probes which SIMD capabilities the compiler supports, as it does for every other binding.
//!
//! File: build.rs
//! Author: Ash Vardanian

use std::env;
use std::path::{Path, PathBuf};

/// The Cargo features that switch a CMake option, each passed as ON or OFF, never left to the host.
const FEATURE_OPTIONS: &[(&str, &str)] = &[
    ("CUDA", "STRINGZILLA_BUILD_CUDA"),
    ("ROCM", "STRINGZILLA_BUILD_ROCM"),
    ("METAL", "STRINGZILLA_BUILD_METAL"),
];

/// The CMake cache variables read from the environment, like `STRINGZILLA_TARGET_ARCH=native`, a
/// GPU architecture list, or a capability override such as `STRINGZILLA_TARGET_SVE2=0`.
const ENVIRONMENT_VARIABLES: &[&str] = &[
    "STRINGZILLA_TARGET_ARCH",
    "STRINGZILLA_CUDA_ARCHITECTURES",
    "STRINGZILLA_ROCM_ARCHITECTURES",
    "STRINGZILLA_TARGET_WESTMERE",
    "STRINGZILLA_TARGET_GOLDMONT",
    "STRINGZILLA_TARGET_HASWELL",
    "STRINGZILLA_TARGET_SKYLAKE",
    "STRINGZILLA_TARGET_ICELAKE",
    "STRINGZILLA_TARGET_NEON",
    "STRINGZILLA_TARGET_NEONAES",
    "STRINGZILLA_TARGET_NEONSHA",
    "STRINGZILLA_TARGET_SVE",
    "STRINGZILLA_TARGET_SVE2",
    "STRINGZILLA_TARGET_SVE2AES",
    "STRINGZILLA_TARGET_RVV",
    "STRINGZILLA_TARGET_RVVCRYPTO",
    "STRINGZILLA_TARGET_LOONGSONASX",
    "STRINGZILLA_TARGET_POWERVSX",
];

/// Rebuilds when a directory's contents change, recursing as Cargo watches only what it is told.
fn watch(directory: &Path) {
    println!("cargo:rerun-if-changed={}", directory.display());
    for entry in std::fs::read_dir(directory).into_iter().flatten().flatten() {
        let path = entry.path();
        match path.is_dir() {
            true => watch(&path),
            false => println!("cargo:rerun-if-changed={}", path.display()),
        }
    }
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let enabled = |feature: &str| env::var_os(format!("CARGO_FEATURE_{feature}")).is_some();

    // A directory holding `stringzilla_static` that CMake already built, like a parent project's
    // build tree or a release, skips the build here.
    println!("cargo:rerun-if-env-changed=STRINGZILLA_LIBRARY_DIR");
    let build = match env::var_os("STRINGZILLA_LIBRARY_DIR") {
        Some(directory) => PathBuf::from(directory),
        None => build_library(&manifest, enabled),
    };
    // Straight in the directory, or one configuration deeper on the multi-configuration generators.
    println!("cargo:rustc-link-search=native={}", build.display());
    println!("cargo:rustc-link-search=native={}", build.join("Release").display());
    println!("cargo:rustc-link-lib=static=stringzilla_static");
    println!("cargo:include={}", manifest.join("include").display());

    if enabled("CUDA") || enabled("ROCM") {
        // The runtimes come from wherever CMake found the toolkits it compiled the units with, and
        // a prebuilt archive without its build tree leaves them to the linker's own search path.
        let cache = std::fs::read_to_string(build.join("CMakeCache.txt")).unwrap_or_default();
        // The directory `levels` above a path CMake cached.
        let search = |key: &str, levels: usize| {
            let cached = cache.lines().find_map(|line| line.strip_prefix(key));
            if let Some(directory) = cached.and_then(|path| Path::new(path).ancestors().nth(levels)) {
                println!("cargo:rustc-link-search=native={}", directory.display());
            }
        };
        if enabled("CUDA") {
            search("CUDA_cudart_LIBRARY:FILEPATH=", 1);
            println!("cargo:rustc-link-lib=cudart");
        }
        if enabled("ROCM") {
            // HIP's package lives in `<rocm>/lib/cmake/hip`, two levels under `libamdhip64`.
            search("hip_DIR:PATH=", 2);
            println!("cargo:rustc-link-lib=amdhip64");
        }
        // The GPU units' host side is C++, reaching its runtime for guarded statics and unwinding.
        if env::var("CARGO_CFG_TARGET_ENV").unwrap() != "msvc" {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }
    if enabled("METAL") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}

/// Builds `stringzilla_static` through CMake and returns the directory holding it.
fn build_library(manifest: &Path, enabled: impl Fn(&str) -> bool) -> PathBuf {
    println!("cargo:rerun-if-changed={}", manifest.join("CMakeLists.txt").display());
    println!("cargo:rerun-if-changed={}", manifest.join("VERSION").display());
    for directory in ["cmake", "c", "include", "probes"] {
        watch(&manifest.join(directory));
    }

    let mut library = cmake::Config::new(manifest);
    library
        .profile("Release")
        .define("STRINGZILLA_BUILD_SHARED", "OFF")
        .build_target("stringzilla_static");
    for (feature, option) in FEATURE_OPTIONS {
        library.define(option, if enabled(feature) { "ON" } else { "OFF" });
    }
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let is_wasm = arch == "wasm32" || arch == "wasm64";

    // The crate reconfigures whenever Cargo reruns this script, so a variable set, changed or
    // dropped since the last build reaches the cache, where an unset one is removed to fall
    // back to the default.
    for variable in ENVIRONMENT_VARIABLES {
        println!("cargo:rerun-if-env-changed={variable}");
        // A WebAssembly module's kit follows the Rust target's features instead, below.
        if is_wasm && *variable == "STRINGZILLA_TARGET_ARCH" {
            continue;
        }
        match env::var(variable) {
            Ok(value) => library.define(variable, value),
            Err(_) => library.configure_arg(format!("-U{variable}")),
        };
    }

    if is_wasm {
        // An engine validates a module whole, so its one SIMD kit is the one
        // the Rust target declares.
        println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
        let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
        let declares = |feature: &str| target_features.split(',').any(|declared| declared == feature);
        let kit = match (declares("relaxed-simd"), declares("simd128")) {
            (true, _) => "v128relaxed",
            (false, true) => "v128",
            (false, false) => "serial",
        };
        library.define("STRINGZILLA_TARGET_ARCH", kit);
        // The checked-in toolchains name the SDK and its sysroot, which `cmake-rs` cannot.
        println!("cargo:rerun-if-env-changed=WASI_SDK_PATH");
        println!("cargo:rerun-if-env-changed=WASI_SDK_PREFIX");
        let target = env::var("TARGET").unwrap();
        match os.as_str() {
            "wasi" if target.ends_with("-threads") => {
                library.define(
                    "CMAKE_TOOLCHAIN_FILE",
                    manifest.join("cmake/toolchain-wasm32-wasi-threads.cmake"),
                );
            }
            "wasi" => {
                library.define(
                    "CMAKE_TOOLCHAIN_FILE",
                    manifest.join("cmake/toolchain-wasm32-wasi.cmake"),
                );
            }
            "unknown" | "none" => {}
            os => panic!("StringZilla has no CMake toolchain for WebAssembly on `{os}`, the target `{target}`"),
        }
    }
    // Without an OS there is no LibC either: `Generic` builds the library without one,
    // cross-compiling with the compiler `cmake-rs` picks.
    if os == "unknown" || os == "none" {
        library
            .define("CMAKE_SYSTEM_NAME", "Generic")
            .define("CMAKE_SYSTEM_PROCESSOR", &arch)
            .define("CMAKE_TRY_COMPILE_TARGET_TYPE", "STATIC_LIBRARY");
    }

    // `build_target` stops short of installing, so the archive sits where the generator left it.
    library.build().join("build")
}
