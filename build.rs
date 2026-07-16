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

    // SIMD tier selection - two probed facts per tier, shared with the CMake build through the same
    // checked-in `probes/` sources:
    //   COMPILE → can this toolchain emit it? Try-compiled from `probes/<arch>_<tier>.c`, which reuses
    //             the real kernels' `target` pragmas, intrinsics, and platform guards.
    //   RUN     → can this machine execute it? Learned by running `probes/run_capabilities.c` on native
    //             builds; cross builds fall back to the target description (`CARGO_CFG_TARGET_FEATURE`).
    // Dynamic dispatch (default) enables the COMPILE set - the load-time table masks what the CPU lacks -
    // but only where the built library has real runtime detection (`probes/runtime_detection.c`); static
    // dispatch bakes the best tier into every symbol with no guard, so it enables COMPILE ∩ RUN.
    let is_wasm = target_arch == "wasm32" || target_arch == "wasm64";
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
            && if dynamic_dispatch {
                runtime_detectable || described
            } else {
                runnable
            };

        // Overrides: `SZ_USE_NEON=0 SZ_USE_SVE=1 cargo build`. "0"/"false"/"off"/"no" disable, an empty
        // value means unset, anything else force-enables - past the RUN gate (the deployment CPU may
        // differ from this machine), never past the COMPILE gate (an instruction the compiler refuses to
        // produce cannot be linked).
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
                // Only warn when the drop is surprising - the target description claims the tier, yet the
                // toolchain cannot build it. Undescribed tiers skip silently (e.g. SVE on Apple targets,
                // where the probe itself rules the hardware out).
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
        flags.insert(probe.define.to_string(), enabled);
    }
    if tuned_beyond_description {
        println!(
            "cargo:warning=Static dispatch: tiers beyond the declared target features were enabled because \
             this machine supports them. The binary is tuned to this machine and is NOT portable to older \
             CPUs; pin `-C target-feature=…` or set `SZ_USE_X=0` for portable builds."
        );
    }
    if gated_by_description {
        println!(
            "cargo:warning=Static dispatch: some SIMD tiers were disabled because the target description \
             does not advertise them. Build with `RUSTFLAGS=\"-C target-cpu=native\"` (or `-C \
             target-feature=+…`) to bake in the best tier for the deployment machine."
        );
    }

    // WebAssembly selects SIMD through instruction flags rather than per-function `target` attributes, so
    // mirror the enabled tiers onto the compiler invocation; the flags define `__wasm_simd128__` /
    // `__wasm_relaxed_simd__`, which `types.h` maps back to `SZ_USE_V128` / `SZ_USE_V128RELAXED`.
    if is_wasm {
        if *flags.get("SZ_USE_V128RELAXED").unwrap_or(&false) {
            build.flag("-msimd128").flag("-mrelaxed-simd");
        } else if *flags.get("SZ_USE_V128").unwrap_or(&false) {
            build.flag("-msimd128");
        }
    }

    // The compile probes already rejected anything this toolchain cannot build, so failures here are real
    // bugs that deserve a loud error, not a silent fallback to a slower tier.
    build.compile("stringzilla");

    // Only re-run the C build when the C sources, headers, or probes change. The Rust source
    // (`rust/stringzilla.rs`) does not feed this compilation, so listing it here forced a full ~55s C
    // rebuild on every Rust-only edit.
    println!("cargo:rerun-if-changed=c/stringzilla");
    println!("cargo:rerun-if-changed=include/stringzilla");
    println!("cargo:rerun-if-changed=probes");

    // Re-run when any tier override changes, and when the target's advertised features change (an edit to
    // `-C target-cpu` / `-C target-feature` surfaces here as `CARGO_CFG_TARGET_FEATURE`).
    for probe in isa_probes(&target_arch) {
        println!("cargo:rerun-if-env-changed={}", probe.define);
    }
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");

    flags
}

/// One SIMD tier's probe row. The `probe_file` sources under `probes/` are shared with the CMake build
/// (same convention as the sibling NumKong project) and reuse the real kernels' `target` pragmas and
/// intrinsics, so try-compiling one answers "can this toolchain emit this tier?" honestly - including
/// ICE-prone toolchains, which only trip when SIMD values actually cross function boundaries.
struct IsaProbe {
    /// The `SZ_USE_*` macro handed to the C build - also the env var that force-enables/disables the tier.
    define: &'static str,
    /// The checked-in probe program, `probes/<arch>_<tier>.c`.
    probe_file: &'static str,
    /// Extra GCC/Clang flags the probe (and the real build) need; only the wasm tiers use any - native
    /// tiers carry their ISA in per-function `target` pragmas over baseline flags.
    gcc_flags: &'static [&'static str],
    /// Extra MSVC flags; none today - MSVC compiles any x86 intrinsic regardless of `/arch`, and the SVE
    /// probes `#error` under MSVC by design (no SVE pragmas or intrinsics there).
    msvc_flags: &'static [&'static str],
    /// This tier's token in `probes/run_capabilities.c` output - the library's own capability names.
    token: &'static str,
    /// Cumulative Rust `target_feature` tokens a CPU needs to RUN this tier, mirroring the nesting that
    /// `types.h` closes downward (NEON ⊂ SVE ⊂ SVE2; WESTMERE ⊂ HASWELL ⊂ SKYLAKE ⊂ ICELAKE; Goldmont and
    /// the Arm crypto tiers are orthogonal). This is the fallback RUN answer when the machine cannot be
    /// probed (cross builds); empty means "the build flags are the description" (the wasm tiers).
    runs_on: &'static [&'static str],
}

const ARM_PROBES: &[IsaProbe] = &[
    IsaProbe {
        define: "SZ_USE_SVE2AES",
        probe_file: "probes/arm_sve2aes.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "sve2aes",
        runs_on: &["neon", "sve", "sve2", "sve2-aes"],
    },
    IsaProbe {
        define: "SZ_USE_SVE2",
        probe_file: "probes/arm_sve2.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "sve2",
        runs_on: &["neon", "sve", "sve2"],
    },
    IsaProbe {
        define: "SZ_USE_SVE",
        probe_file: "probes/arm_sve.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "sve",
        runs_on: &["neon", "sve"],
    },
    IsaProbe {
        define: "SZ_USE_NEONSHA",
        probe_file: "probes/arm_neonsha.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "neonsha",
        runs_on: &["neon", "sha2"],
    },
    IsaProbe {
        define: "SZ_USE_NEONAES",
        probe_file: "probes/arm_neonaes.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "neonaes",
        runs_on: &["neon", "aes"],
    },
    IsaProbe {
        define: "SZ_USE_NEON",
        probe_file: "probes/arm_neon.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "neon",
        runs_on: &["neon"],
    },
];

const X86_PROBES: &[IsaProbe] = &[
    IsaProbe {
        define: "SZ_USE_ICELAKE",
        probe_file: "probes/x86_icelake.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "icelake",
        // The Ice Lake kernels split across two orthogonal AVX-512 sub-extensions: the hash/intersect
        // cores need VNNI (`_mm512_dpbusds_epi32`), the find/UTF-8 cores need VBMI2 (compress/expand) -
        // so the RUN description must require both, matching the runtime detector's icelake bit.
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
        define: "SZ_USE_SKYLAKE",
        probe_file: "probes/x86_skylake.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "skylake",
        runs_on: &["sse4.2", "aes", "avx2", "avx512f", "avx512vl", "avx512bw"],
    },
    IsaProbe {
        define: "SZ_USE_HASWELL",
        probe_file: "probes/x86_haswell.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "haswell",
        runs_on: &["sse4.2", "aes", "avx2"],
    },
    IsaProbe {
        define: "SZ_USE_GOLDMONT",
        probe_file: "probes/x86_goldmont.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "goldmont",
        // Matches the kernel's own pragma (`sse3,ssse3,sse4.1,sha`) - Goldmont is the orthogonal SHA-NI
        // tier, not part of the AVX nesting, and its kernels never touch SSE4.2.
        runs_on: &["sse3", "ssse3", "sse4.1", "sha"],
    },
    IsaProbe {
        define: "SZ_USE_WESTMERE",
        probe_file: "probes/x86_westmere.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "westmere",
        runs_on: &["sse4.2", "aes"],
    },
];

const WASM_PROBES: &[IsaProbe] = &[
    IsaProbe {
        define: "SZ_USE_V128RELAXED",
        probe_file: "probes/wasm_v128relaxed.c",
        gcc_flags: &["-msimd128", "-mrelaxed-simd"],
        msvc_flags: &[],
        token: "v128relaxed",
        runs_on: &[],
    },
    IsaProbe {
        define: "SZ_USE_V128",
        probe_file: "probes/wasm_v128.c",
        gcc_flags: &["-msimd128"],
        msvc_flags: &[],
        token: "v128",
        runs_on: &[],
    },
];

const RISCV_PROBES: &[IsaProbe] = &[
    IsaProbe {
        define: "SZ_USE_RVVCRYPTO",
        probe_file: "probes/riscv_rvvcrypto.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "rvvcrypto",
        runs_on: &["v", "zvkned", "zvknhb"],
    },
    IsaProbe {
        define: "SZ_USE_RVV",
        probe_file: "probes/riscv_rvv.c",
        gcc_flags: &[],
        msvc_flags: &[],
        token: "rvv",
        runs_on: &["v"],
    },
];

const LOONGARCH_PROBES: &[IsaProbe] = &[IsaProbe {
    define: "SZ_USE_LASX",
    probe_file: "probes/loongarch_lasx.c",
    gcc_flags: &[],
    msvc_flags: &[],
    token: "lasx",
    runs_on: &["lasx"],
}];

const POWER_PROBES: &[IsaProbe] = &[IsaProbe {
    define: "SZ_USE_POWERVSX",
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

/// Try to compile one tier's probe program with the target toolchain; `true` means the toolchain can emit
/// that tier. Drives the discovered compiler directly instead of `cc::Build::try_compile` so a FAILED
/// probe stays quiet - probe failures are expected outcomes (old GCCs, ICE-prone toolchains), and `cc`
/// would otherwise relay the compiler's crash spew as a wall of `cargo:warning` lines. Flags are appended
/// verbatim (never `flag_if_supported`) so a dropped flag can never make a probe pass dishonestly. The
/// optimization level is deliberately NOT pinned: `cc` inherits the profile's `OPT_LEVEL`, and toolchain
/// breakage can be opt-level-specific (Homebrew clang 20 ICEs emitting SVE at `-O0` but not `-O1`+), so
/// the probe must compile exactly the way the real kernels will.
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
            .arg("-std=c99") // the real build enforces C99 (see `build_stringzilla`), so probes must too
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

/// Compile-probe `probes/runtime_detection.c`: `true` means the library built for this target performs
/// real runtime capability detection, so its load-time dispatch table will mask tiers the CPU lacks and
/// dynamic dispatch may safely enable everything the toolchain can emit. The answer is owned by the
/// header (`SZ_CAPABILITIES_RUNTIME_DETECTABLE_`, defined next to the detectors), which is why this is a
/// compile probe against it rather than a platform list here. Works for cross targets - nothing runs.
/// `SZ_AVOID_LIBC` must match the real build: detectability hinges on it where detection reads the
/// auxiliary vector.
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
    let avoid_libc_define = format!("SZ_AVOID_LIBC={}", if avoid_libc { "1" } else { "0" });

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

/// Compile `probes/run_capabilities.c` into a host executable, run it, and return the capability tokens
/// the build machine reports (e.g. `{"serial", "neon", "neonaes"}`). Returns `None` when cross-compiling
/// (`HOST != TARGET` - the binary could not run here) or when any step fails: no compiler, a crashed or
/// signal-killed probe, empty output. Callers then fall back to the target description. The `cc` crate
/// only produces objects and archives, so the discovered compiler is driven directly to link an
/// executable; `to_command()` carries the MSVC environment (`INCLUDE`/`LIB`/`PATH`) that `cl` needs.
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
    // `-O0` is fine here, unlike in `probe_isa`: this TU is serial-only (no SIMD codegen at stake) and
    // only needs to execute correctly, so favor the fastest compile.
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
    build.flag("-std=c++20").flag("--expt-relaxed-constexpr");
    // Coverage-parity with the CMake build and setup.py: Ampere (sm_80) and Hopper (sm_90) real SASS plus forward PTX.
    // Those two link loose objects and give the Hopper-only DPX TUs a narrower sm_90 set; the `cc` crate emits one
    // static archive (per-tier splitting would need non-portable linker grouping), so every TU shares this union set -
    // the Hopper providers' extra sm_80 cubin is their guarded scalar fallback, which `--compress-all` erases.
    for gencode in [
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_90,code=sm_90",
        "-gencode=arch=compute_90,code=compute_90",
    ] {
        build.flag(gencode);
    }
    build.flag("-Xfatbin=--compress-all");
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
