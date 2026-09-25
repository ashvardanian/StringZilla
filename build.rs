//! Cargo build script for the StringZilla C library, picking the SIMD tiers to compile in.
//!
//! File: build.rs
//! Author: Ash Vardanian

use std::env;

/// Builds the StringZilla C library, engines included, with the SIMD tiers this toolchain and
/// target allow.
fn main() {
    let mut build = cc::Build::new();
    build
        .include("include")
        .include("c/stringzilla") // for the same-directory `dispatch.h`
        .warnings(false)
        .define("STRINGZILLA_DEBUG", "0")
        .flag("-std=c99") // Enforce C99 standard
        .flag_if_supported("-fdiagnostics-color=always")
        .flag_if_supported("-fPIC");
    for flag in no_builtin_flags() {
        build.flag(flag);
    }

    // Dispatch model, selected by the `dynamic-dispatch` feature:
    //  - ON, the default: compile the per-domain dispatch shims into a runtime table that picks
    //    the best ISA tier at load - mirrors CMake's `stringzilla_shared`. Flexible, one
    //    indirection per call.
    //  - OFF: compile a single amalgamation TU, `#include <stringzilla/stringzilla.h>`, that
    //    resolves each public function to one ISA tier at compile time and exports it via
    //    `STRINGZILLA_EXPORT_`, as `types.h` describes. No table or indirection; the tier is baked
    //    in, which is less portable but faster on some workloads.
    if env::var("CARGO_FEATURE_DYNAMIC_DISPATCH").is_ok() {
        build.define("STRINGZILLA_RUNTIME_DISPATCH", "1");
        build.files([
            "c/stringzilla/runtime.c",
            "c/stringzilla/compare.c",
            "c/stringzilla/memory.c",
            "c/stringzilla/hash.c",
            "c/stringzilla/cipher.c",
            "c/stringzilla/find.c",
            "c/stringzilla/sort.c",
            "c/stringzilla/intersect.c",
            "c/stringzilla/levenshtein.c",
            "c/stringzilla/overlap.c",
            "c/stringzilla/substrings.c",
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
        // One translation unit includes the umbrella header once, generated into `OUT_DIR` so there
        // is no checked-in source, the same pattern as the relaxed-SIMD probe. Exactly one TU
        // avoids duplicate symbols.
        build.define("STRINGZILLA_RUNTIME_DISPATCH", "0");
        build.define("STRINGZILLA_EXPORT_", "1");
        let amalgam_path = std::path::Path::new(&env::var("OUT_DIR").unwrap_or_default()).join("sz_stringzilla.c");
        std::fs::write(&amalgam_path, "#include <stringzilla/stringzilla.h>\n").expect("write amalgamation TU");
        build.file(&amalgam_path);
    }

    // Cargo will set different environment variables that we can use to properly
    // configure the build.
    // https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts
    // https://doc.rust-lang.org/reference/conditional-compilation.html#r-cfg.target_endian
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_endian = env::var("CARGO_CFG_TARGET_ENDIAN").unwrap_or_default();
    let target_bits = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    let avoid_libc = target_os == "unknown" || target_os.is_empty();
    build.define("STRINGZILLA_WITH_LIBC", if avoid_libc { "0" } else { "1" });

    let is_64bit_x86 = target_arch == "x86_64" && target_bits == "64";
    let is_64bit_arm = target_arch == "aarch64" && target_bits == "64";
    build.define(
        "STRINGZILLA_ARCH_BIG_ENDIAN_",
        if target_endian == "big" { "1" } else { "0" },
    );
    build.define("STRINGZILLA_ARCH_X86_64_", if is_64bit_x86 { "1" } else { "0" });
    build.define("STRINGZILLA_ARCH_ARM64_", if is_64bit_arm { "1" } else { "0" });

    // SIMD tier selection - two probed facts per tier, shared with the CMake build through the same
    // checked-in `probes/` sources:
    //   COMPILE → can this toolchain emit it? Try-compiled from `probes/<arch>_<tier>.c`, which
    //             reuses the real kernels' `target` pragmas, intrinsics, and platform guards.
    //   RUN     → can this machine execute it? Learned by running `probes/run_capabilities.c` on
    //             native builds; cross builds fall back to the target description,
    //             `CARGO_CFG_TARGET_FEATURE`.
    // Dynamic dispatch, the default, enables the COMPILE set - the load-time table masks what the
    // CPU lacks - but only where the built library has real runtime detection, which
    // `probes/runtime_detection.c` answers; static dispatch bakes the best tier into every symbol
    // with no guard, so it enables COMPILE ∩ RUN. WebAssembly engines validate a module whole, so
    // there both models enable COMPILE ∩ the target description.
    let is_wasm = target_arch == "wasm32" || target_arch == "wasm64";
    let is_msvc = build.get_compiler().is_like_msvc();
    let dynamic_dispatch = env::var("CARGO_FEATURE_DYNAMIC_DISPATCH").is_ok();
    let target_features: std::collections::HashSet<String> = env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let runtime_detectable = dynamic_dispatch && probe_runtime_detection(avoid_libc);
    let machine_tokens = if dynamic_dispatch { None } else { machine_capabilities() };

    let mut tuned_beyond_description = false;
    let mut gated_by_description = false;
    for probe in isa_probes(&target_arch) {
        let compilable = probe_isa(probe);
        let described = probe.runs_on.iter().all(|f| target_features.contains(*f));
        let runnable = machine_tokens
            .as_ref()
            .map_or(described, |tokens| tokens.contains(probe.token));
        let default_on = compilable
            && if is_wasm {
                described
            } else if dynamic_dispatch {
                runtime_detectable || described
            } else {
                runnable
            };

        // Overrides: `STRINGZILLA_TARGET_NEON=0 STRINGZILLA_TARGET_SVE=1 cargo build`.
        // "0"/"false"/"off"/"no" disable, an empty value means unset, anything else force-enables -
        // past the RUN gate, as the deployment CPU may differ from this machine, but never past the
        // COMPILE gate, as an instruction the compiler refuses to produce cannot be linked.
        let env_value = env::var(probe.define)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(|v| !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "off" | "no"));
        let enabled = match env_value {
            Some(false) => {
                println!("cargo:warning=Disabled {} via environment variable", probe.define);
                false
            }
            Some(true) if !compilable => {
                println!(
                    "cargo:warning={} requested via environment variable, but this toolchain cannot compile {}; disabling",
                    probe.define, probe.probe_file
                );
                false
            }
            Some(true) => {
                if !dynamic_dispatch && !runnable {
                    println!(
                        "cargo:warning={} forced on beyond what this target or machine supports; static dispatch may SIGILL at runtime",
                        probe.define
                    );
                }
                true
            }
            None => {
                // Only warn when the drop is surprising - the target description claims the tier,
                // yet the toolchain cannot build it. Undescribed tiers skip silently (e.g. SVE on
                // Apple targets, where the probe itself rules the hardware out).
                if described && !compilable {
                    println!(
                        "cargo:warning=The target declares support, but this toolchain cannot compile {}; building without {}",
                        probe.probe_file, probe.define
                    );
                }
                if !dynamic_dispatch && compilable {
                    tuned_beyond_description |= default_on && !described; // the run probe widened the set
                    gated_by_description |= !default_on && machine_tokens.is_none();
                    // cross → description too weak
                }
                default_on
            }
        };

        build.define(probe.define, if enabled { "1" } else { "0" });
        if enabled {
            // Only the WebAssembly tiers carry any: their SIMD is a whole-module flag, not
            // a per-function pragma.
            for flag in if is_msvc { probe.msvc_flags } else { probe.gcc_flags } {
                build.flag(flag);
            }
        }
    }
    if tuned_beyond_description {
        println!(
            "cargo:warning=Static dispatch: tiers beyond the declared target features were enabled because \
             this machine supports them. The binary is tuned to this machine and is NOT portable to older \
             CPUs; pin `-C target-feature=…` or set `STRINGZILLA_TARGET_X=0` for portable builds."
        );
    }
    if gated_by_description {
        println!(
            "cargo:warning=Static dispatch: some SIMD tiers were disabled because the target description \
             does not advertise them. Build with `RUSTFLAGS=\"-C target-cpu=native\"` (or `-C \
             target-feature=+…`) to bake in the best tier for the deployment machine."
        );
    }

    // The compile probes already rejected anything this toolchain cannot build, so failures here
    // are real bugs that deserve a loud error, not a silent fallback to a slower tier.
    build.compile("stringzilla");

    // Only re-run the C build when the C sources, headers, or probes change. The Rust module trees
    // under `rust/` do not feed this compilation, so listing them here forced a full ~55s C rebuild
    // on every Rust-only edit.
    println!("cargo:rerun-if-changed=c/stringzilla");
    println!("cargo:rerun-if-changed=include/stringzilla");
    println!("cargo:rerun-if-changed=probes");

    // Re-run when any tier override changes, and when the target's advertised features change: an
    // edit to `-C target-cpu` or `-C target-feature` surfaces here as `CARGO_CFG_TARGET_FEATURE`.
    for probe in isa_probes(&target_arch) {
        println!("cargo:rerun-if-env-changed={}", probe.define);
    }
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
}

/// One SIMD tier's probe row. The `probe_file` sources under `probes/` are shared with the CMake
/// build and reuse the real kernels' `target` pragmas and intrinsics, so try-compiling one answers
/// "can this toolchain emit this tier?" honestly - including ICE-prone toolchains, which only trip
/// when SIMD values actually cross function boundaries.
struct IsaProbe {
    /// The `STRINGZILLA_TARGET_*` macro handed to the C build - also the env var that force-enables
    /// or disables the tier.
    define: &'static str,
    /// The checked-in probe program, `probes/<arch>_<tier>.c`.
    probe_file: &'static str,
    /// Extra GCC and Clang flags the probe and the real build need; only the wasm tiers use any -
    /// native tiers carry their ISA in per-function `target` pragmas over baseline flags.
    gcc_flags: &'static [&'static str],
    /// Extra MSVC flags; none today - MSVC compiles any x86 intrinsic regardless of `/arch`, and
    /// the SVE probes `#error` under MSVC by design, as it has no SVE pragmas or intrinsics.
    msvc_flags: &'static [&'static str],
    /// This tier's token in `probes/run_capabilities.c` output - the library's
    /// own capability names.
    token: &'static str,
    /// Cumulative Rust `target_feature` tokens a CPU needs to RUN this tier, mirroring the
    /// nesting that `types.h` closes downward: NEON ⊂ SVE ⊂ SVE2 and WESTMERE ⊂ HASWELL ⊂
    /// SKYLAKE ⊂ ICELAKE, while Goldmont and the Arm crypto tiers are orthogonal. This is the
    /// fallback RUN answer when the machine cannot be probed, as in cross builds, and the only
    /// answer on WebAssembly.
    runs_on: &'static [&'static str],
}

const ARM_PROBES: &[IsaProbe] = &[
    IsaProbe {
        define: "STRINGZILLA_TARGET_SVE2AES",
        probe_file: "probes/arm_sve2aes.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "sve2aes",
        runs_on: &["neon", "sve", "sve2", "sve2-aes"],
    },
    IsaProbe {
        define: "STRINGZILLA_TARGET_SVE2",
        probe_file: "probes/arm_sve2.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "sve2",
        runs_on: &["neon", "sve", "sve2"],
    },
    IsaProbe {
        define: "STRINGZILLA_TARGET_SVE",
        probe_file: "probes/arm_sve.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "sve",
        runs_on: &["neon", "sve"],
    },
    IsaProbe {
        define: "STRINGZILLA_TARGET_NEONSHA",
        probe_file: "probes/arm_neonsha.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "neonsha",
        runs_on: &["neon", "sha2"],
    },
    IsaProbe {
        define: "STRINGZILLA_TARGET_NEONAES",
        probe_file: "probes/arm_neonaes.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "neonaes",
        runs_on: &["neon", "aes"],
    },
    IsaProbe {
        define: "STRINGZILLA_TARGET_NEON",
        probe_file: "probes/arm_neon.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "neon",
        runs_on: &["neon"],
    },
];

const X86_PROBES: &[IsaProbe] = &[
    IsaProbe {
        define: "STRINGZILLA_TARGET_ICELAKE",
        probe_file: "probes/x86_icelake.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "icelake",
        // The Ice Lake kernels split across two orthogonal AVX-512 sub-extensions: the
        // hash/intersect cores need VNNI (`_mm512_dpbusds_epi32`), the find/UTF-8 cores need VBMI2
        // (compress/expand) - so the RUN description must require both, matching the runtime
        // detector's icelake bit.
        runs_on: &[
            "sse4.2",
            "aes",
            "avx2",
            "avx512f",
            "avx512vl",
            "avx512bw",
            "avx512vbmi",
            "avx512vbmi2",
            "avx512vnni",
            "vaes",
        ],
    },
    IsaProbe {
        define: "STRINGZILLA_TARGET_SKYLAKE",
        probe_file: "probes/x86_skylake.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "skylake",
        runs_on: &["sse4.2", "aes", "avx2", "avx512f", "avx512vl", "avx512bw"],
    },
    IsaProbe {
        define: "STRINGZILLA_TARGET_HASWELL",
        probe_file: "probes/x86_haswell.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "haswell",
        runs_on: &["sse4.2", "aes", "avx2"],
    },
    IsaProbe {
        define: "STRINGZILLA_TARGET_GOLDMONT",
        probe_file: "probes/x86_goldmont.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "goldmont",
        // Matches the kernel's own pragma (`sse3,ssse3,sse4.1,sha`) - Goldmont is the orthogonal
        // SHA-NI tier, not part of the AVX nesting, and its kernels never touch SSE4.2.
        runs_on: &["sse3", "ssse3", "sse4.1", "sha"],
    },
    IsaProbe {
        define: "STRINGZILLA_TARGET_WESTMERE",
        probe_file: "probes/x86_westmere.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "westmere",
        runs_on: &["sse4.2", "aes"],
    },
];

const WASM_PROBES: &[IsaProbe] = &[
    IsaProbe {
        define: "STRINGZILLA_TARGET_V128RELAXED",
        probe_file: "probes/wasm_v128relaxed.c",
        gcc_flags: &["-msimd128", "-mrelaxed-simd"],
        msvc_flags: &[],
        token: "v128relaxed",
        runs_on: &["simd128", "relaxed-simd"],
    },
    IsaProbe {
        define: "STRINGZILLA_TARGET_V128",
        probe_file: "probes/wasm_v128.c",
        gcc_flags: &["-msimd128"],
        msvc_flags: &[],
        token: "v128",
        runs_on: &["simd128"],
    },
];

const RISCV_PROBES: &[IsaProbe] = &[
    IsaProbe {
        define: "STRINGZILLA_TARGET_RVVCRYPTO",
        probe_file: "probes/riscv_rvvcrypto.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "rvvcrypto",
        runs_on: &["v", "zvkned", "zvknhb"],
    },
    IsaProbe {
        define: "STRINGZILLA_TARGET_RVV",
        probe_file: "probes/riscv_rvv.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "rvv",
        runs_on: &["v"],
    },
];

const LOONGARCH_PROBES: &[IsaProbe] = &[IsaProbe {
    define: "STRINGZILLA_TARGET_LASX",
    probe_file: "probes/loongarch_lasx.c",
    gcc_flags: &[],
    msvc_flags: &[],
    token: "lasx",
    runs_on: &["lasx"],
}];

const POWER_PROBES: &[IsaProbe] = &[IsaProbe {
    define: "STRINGZILLA_TARGET_POWERVSX",
    probe_file: "probes/power_vsx.c",
    gcc_flags: &[],
    msvc_flags: &[],
    token: "powervsx",
    runs_on: &["vsx"],
}];

/// Candidate SIMD tiers for a target architecture, newest first.
/// https://doc.rust-lang.org/reference/conditional-compilation.html#target_arch
fn isa_probes(target_arch: &str) -> &'static [IsaProbe] {
    match target_arch {
        "arm" | "aarch64" => ARM_PROBES,
        "x86_64" => X86_PROBES,
        "wasm32" | "wasm64" => WASM_PROBES,
        "riscv64" => RISCV_PROBES,
        "loongarch64" => LOONGARCH_PROBES,
        "powerpc64" => POWER_PROBES,
        _ => &[],
    }
}

/// Try to compile one tier's probe program with the target toolchain; `true` means the toolchain
/// can emit that tier. Drives the discovered compiler directly instead of `cc::Build::try_compile`
/// so a failed probe stays quiet - probe failures are expected outcomes on old GCCs and ICE-prone
/// toolchains, and `cc` would otherwise relay the compiler's crash spew as a wall of
/// `cargo:warning` lines. Flags are appended verbatim, never through `flag_if_supported`, so a
/// dropped flag can never make a probe pass dishonestly. The optimization level is deliberately
/// left unpinned: `cc` inherits the profile's `OPT_LEVEL`, and toolchain breakage can be
/// opt-level-specific - Homebrew clang 20 ICEs emitting SVE at `-O0` but not at `-O1` and above -
/// so the probe must compile exactly the way the real kernels will.
fn probe_isa(probe: &IsaProbe) -> bool {
    let out_dir = match env::var("OUT_DIR") {
        Ok(dir) => std::path::PathBuf::from(dir),
        Err(_) => return false,
    };
    let name = probe.probe_file.replace("probes/", "sz_probe_").replace(".c", "");

    let mut build = cc::Build::new();
    build.cargo_metadata(false).warnings(false);
    let tool = build.get_compiler();
    let mut command = tool.to_command();
    if tool.is_like_msvc() {
        command
            .current_dir(&out_dir)
            .arg("/nologo")
            .arg("/c")
            .arg(
                std::path::Path::new(probe.probe_file)
                    .canonicalize()
                    .unwrap_or_else(|_| probe.probe_file.into()),
            )
            .arg(format!("/Fo{}", out_dir.join(format!("{name}.obj")).display()));
        for flag in probe.msvc_flags {
            command.arg(flag);
        }
    } else {
        command
            .arg("-std=c99") // the real build enforces C99 (see `main`), so probes must too
            .arg("-c")
            .arg(probe.probe_file)
            .arg("-o")
            .arg(out_dir.join(format!("{name}.o")));
        for flag in probe.gcc_flags {
            command.arg(flag);
        }
    }
    command.output().map(|result| result.status.success()).unwrap_or(false)
}

/// Compile-probe `probes/runtime_detection.c`: `true` means the library built for this target
/// performs real runtime capability detection, so its load-time dispatch table will mask tiers the
/// CPU lacks and dynamic dispatch may safely enable everything the toolchain can emit. The header
/// owns the answer through `STRINGZILLA_HAS_RUNTIME_DETECTION_`, defined next to the detectors,
/// which is why this is a compile probe against it rather than a platform list here. Works for
/// cross targets - nothing runs. `STRINGZILLA_WITH_LIBC` must match the real build: detectability
/// hinges on it where detection reads the auxiliary vector.
fn probe_runtime_detection(avoid_libc: bool) -> bool {
    let out_dir = match env::var("OUT_DIR") {
        Ok(dir) => std::path::PathBuf::from(dir),
        Err(_) => return false,
    };
    let manifest_dir = match env::var("CARGO_MANIFEST_DIR") {
        Ok(dir) => std::path::PathBuf::from(dir),
        Err(_) => return false,
    };
    let source = manifest_dir.join("probes").join("runtime_detection.c");
    let include_dir = manifest_dir.join("include");
    let avoid_libc_define = format!("STRINGZILLA_WITH_LIBC={}", if avoid_libc { "0" } else { "1" });

    let mut build = cc::Build::new();
    build.cargo_metadata(false).warnings(false);
    let tool = build.get_compiler();
    let mut command = tool.to_command();
    if tool.is_like_msvc() {
        command
            .current_dir(&out_dir)
            .arg("/nologo")
            .arg("/c")
            .arg(&source)
            .arg(format!("/I{}", include_dir.display()))
            .arg(format!("/D{avoid_libc_define}"))
            .arg(format!(
                "/Fo{}",
                out_dir.join("sz_probe_runtime_detection.obj").display()
            ));
    } else {
        command
            .arg("-std=c99")
            .arg(format!("-I{}", include_dir.display()))
            .arg(format!("-D{avoid_libc_define}"))
            .arg("-c")
            .arg(&source)
            .arg("-o")
            .arg(out_dir.join("sz_probe_runtime_detection.o"));
    }
    command.output().map(|result| result.status.success()).unwrap_or(false)
}

/// Compile `probes/run_capabilities.c` into a host executable, run it, and return the capability
/// tokens the build machine reports, like `{"serial", "neon", "neonaes"}`. Returns `None` when
/// cross-compiling, as with `HOST != TARGET` the binary could not run here, or when any step fails:
/// no compiler, a crashed or signal-killed probe, empty output. Callers then fall back to the
/// target description. The `cc` crate only produces objects and archives, so the discovered
/// compiler is driven directly to link an executable; `to_command()` carries the MSVC environment,
/// `INCLUDE`, `LIB`, and `PATH`, that `cl` needs.
fn machine_capabilities() -> Option<std::collections::HashSet<String>> {
    let host = env::var("HOST").ok()?;
    let target = env::var("TARGET").ok()?;
    if host != target {
        return None;
    }
    let out_dir = std::path::PathBuf::from(env::var("OUT_DIR").ok()?);
    let manifest_dir = std::path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").ok()?);
    let source = manifest_dir.join("probes").join("run_capabilities.c");
    let include_dir = manifest_dir.join("include");

    let mut probe = cc::Build::new();
    // `-O0` is fine here, unlike in `probe_isa`: this TU is serial-only, with no SIMD codegen at
    // stake, and only needs to execute correctly, so favor the fastest compile.
    probe.cargo_metadata(false).warnings(false).opt_level(0);
    let tool = probe.get_compiler();
    let exe = out_dir.join(if tool.is_like_msvc() {
        "sz_run_capabilities.exe"
    } else {
        "sz_run_capabilities"
    });
    let mut command = tool.to_command();
    if tool.is_like_msvc() {
        command.current_dir(&out_dir); // `cl` drops the intermediate `.obj` into the working directory
        command
            .arg("/nologo")
            .arg(&source)
            .arg(format!("/I{}", include_dir.display()))
            .arg(format!("/Fe{}", exe.display()));
    } else {
        command
            .arg(&source)
            .arg(format!("-I{}", include_dir.display()))
            .arg("-o")
            .arg(&exe);
    }
    let compiled = command.output().ok()?;
    if !compiled.status.success() {
        return None;
    }
    let ran = std::process::Command::new(&exe).output().ok()?;
    if !ran.status.success() {
        return None;
    }
    let stdout = String::from_utf8(ran.stdout).ok()?;
    let tokens: std::collections::HashSet<String> = stdout
        .trim()
        .split(',')
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
        .collect();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens)
    }
}

/// Flags that stop the compiler from substituting its own builtins for the bytewise primitives
/// StringZilla provides — and, under `STRINGZILLA_OVERRIDE_LIBC`, from lowering those
/// implementations back into a self-recursive LibC call. Mirrors the "avoid builtin functions"
/// block in CMakeLists.txt: MSVC disables intrinsic generation with `/Oi-`, GCC and Clang disable
/// the specific `mem*` builtins.
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
