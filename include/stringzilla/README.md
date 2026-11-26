# StringZilla Kernels

This directory holds the kernels behind the [C](../../c/README.md) and [C++](../../cpp/README.md) interfaces, which document how to call them.
Per-family hubs `find.h`, `hash.h`, `sort.h`, `compare.h`, `intersect.h`, `strings.h`, `small_string.h`, `levenshtein.h`, `ngrams.h`, and `substrings.h` declare the family's dispatch points and its kernels, each capability's under its `STRINGZILLA_TARGET_*` macro, from the `c/` header root.
The kernels are defined under the matching subdirectories of the `include/` root — `find/haswell.h`, `hash/icelake.h`, `strings/neon.h`, and so on — and the C++ headers live in the `cpp/` root; a build passes all three as include paths.
The library compiles the kernels once, one capability per unit in `src/capabilities/`, and header-only builds include them into every translation unit.

Each family's design notes, building blocks and measurements live beside its kernels: [`cipher`](cipher/README.md), [`compare`](compare/README.md), [`find`](find/README.md), [`hash`](hash/README.md), [`intersect`](intersect/README.md), [`levenshtein`](levenshtein/README.md), [`ngrams`](ngrams/README.md), [`regex`](regex/README.md), [`sort`](sort/README.md), [`strings`](strings/README.md), [`subdocuments`](subdocuments/README.md), [`substrings`](substrings/README.md), [`utf8_graphemes`](utf8_graphemes/README.md), [`utf8_linebreaks`](utf8_linebreaks/README.md), [`utf8_norm`](utf8_norm/README.md), [`utf8_runes`](utf8_runes/README.md), [`utf8_sentences`](utf8_sentences/README.md), [`utf8_tokens`](utf8_tokens/README.md), [`utf8_uncased`](utf8_uncased/README.md), [`utf8_uncased_fold`](utf8_uncased_fold/README.md), [`utf8_wordbreaks`](utf8_wordbreaks/README.md).

## Build Knobs

The behavior of the core is controlled entirely by preprocessor macros, so you can tune it without touching the sources.
The most important ones, mirrored from `stringzilla.h`:

| Macro                                |  Default | Effect                                   |
| :----------------------------------- | -------: | :--------------------------------------- |
| `STRINGZILLA_HEADER_ONLY`            |      `0` | Inline kernels instead of the library    |
| `STRINGZILLA_DEBUG`                  |      `0` | Debug assertions and logging             |
| `STRINGZILLA_WITH_LIBC`              |      `1` | `0` builds freestanding, without libc    |
| `STRINGZILLA_ALLOW_MISALIGNED_LOADS` | platform | Unaligned word loads in SWAR fallbacks   |
| `STRINGZILLA_SWAR_THRESHOLD`         |     `24` | Length below which scalar loops are used |
| `STRINGZILLA_DEFAULT_ALIGNMENT`      | platform | Default alignment and padding policy     |

`STRINGZILLA_HEADER_ONLY` chooses between the two builds: with `0`, the default, the dispatch points live in the library, which compiles every capability and picks one by the mask each call passes; with `1` the kernels your compiler flags enable inline into your translation unit, and the dispatch points report `sz_missing_library_k`.
`STRINGZILLA_WITH_LIBC=0` builds freestanding without the C standard library, which disables the default `malloc`-based allocator and the `offsetof` static checks.
`STRINGZILLA_ALLOW_MISALIGNED_LOADS` allows unaligned word loads in the SWAR fallbacks where the platform permits.

Each ISA backend's kernels sit under their own macro, which defaults to what the compiler's target flags enable and may be set to 0 to drop them; a 1 without those flags declares kernels the library defines, so it links only against the library:
`STRINGZILLA_TARGET_WESTMERE` for SSE4.2 + AES-NI, `STRINGZILLA_TARGET_GOLDMONT` for SHA-NI, `STRINGZILLA_TARGET_HASWELL` for AVX2, `STRINGZILLA_TARGET_SKYLAKE` for AVX-512 F/BW/VL, `STRINGZILLA_TARGET_ICELAKE` for AVX-512 VBMI + VAES, then `STRINGZILLA_TARGET_NEON`, `STRINGZILLA_TARGET_NEONAES`, `STRINGZILLA_TARGET_NEONSHA`, `STRINGZILLA_TARGET_SVE`, `STRINGZILLA_TARGET_SVE2`, and `STRINGZILLA_TARGET_SVE2AES` on ARM, `STRINGZILLA_TARGET_V128` and `STRINGZILLA_TARGET_V128RELAXED` for WebAssembly SIMD128, `STRINGZILLA_TARGET_RVV` and `STRINGZILLA_TARGET_RVVCRYPTO` for the RISC-V Vector extension, `STRINGZILLA_TARGET_LOONGSONASX` for LoongArch, and `STRINGZILLA_TARGET_POWERVSX` for IBM Power.

The umbrella header also exposes the version triple as `STRINGZILLA_VERSION_MAJOR`/`_MINOR`/`_PATCH`, with matching `sz_version_major()`, `sz_version_minor()`, and `sz_version_patch()` accessors.

## Compilers and Platforms

The headers require only C 99 and C++ 20, so any reasonably modern compiler builds the serial and SWAR baseline.
The per-ISA SIMD kernels and the project's own tests are CI-validated with these toolchains:

| Toolchain | Recommended Versions |
| :-------- | :------------------- |
| GCC       | 13 or newer          |
| Clang     | 18 or newer          |
| MSVC      | Visual Studio 2026   |
| NVCC      | CUDA 13+             |

The RVV kernels need Clang 21 or newer, as older Clang hides the 64-bit vector types from a unit compiled without `v`, and an older compiler builds the libraries without them.
On macOS, prefer Homebrew Clang over Apple Clang; on Windows, MinGW with GCC works alongside MSVC.
NVCC with CUDA 13 builds the device backends of the engine families, reached by building an engine with the mask `sz_capabilities_enabled_cuda` reports for a device.
HIP-Clang builds the same sources for AMD GPUs, reached through `sz_capabilities_enabled_rocm`.
Built with `STRINGZILLA_METAL_ARCH`, the three engines also run on Apple GPUs of the Apple7 family and newer, through the same constructors and verbs.

StringZilla also __compiles to WebAssembly__: the `wasm32` toolchain targets `wasm32-wasip1`, and `STRINGZILLA_MARCH` names the module's one SIMD capability, `serial`, `v128` or the default `v128relaxed`, which enables the `STRINGZILLA_TARGET_V128` and `STRINGZILLA_TARGET_V128RELAXED` kernels listed above.

## Compilation Settings and Debugging

__`STRINGZILLA_DEBUG`__:

> For maximal performance, the C library does not perform any bounds checking in Release builds.
> In C++, bounds checking happens only in places where the STL `std::string` would do it.
> If you want to enable more aggressive bounds-checking, define `STRINGZILLA_DEBUG` before including the header.
> If not explicitly set, it will be inferred from the build type.

__`STRINGZILLA_TARGET_GOLDMONT`, `STRINGZILLA_TARGET_WESTMERE`, `STRINGZILLA_TARGET_HASWELL`, `STRINGZILLA_TARGET_SKYLAKE`, `STRINGZILLA_TARGET_ICELAKE`, `STRINGZILLA_TARGET_NEON`, `STRINGZILLA_TARGET_NEONAES`, `STRINGZILLA_TARGET_NEONSHA`, `STRINGZILLA_TARGET_SVE`, `STRINGZILLA_TARGET_SVE2`, `STRINGZILLA_TARGET_SVE2AES`, `STRINGZILLA_TARGET_V128`, `STRINGZILLA_TARGET_V128RELAXED`, `STRINGZILLA_TARGET_RVV`, `STRINGZILLA_TARGET_LOONGSONASX`, `STRINGZILLA_TARGET_POWERVSX`__:

> Each says whether a unit has one SIMD family's kernels; set one to 0 to drop a family for compatibility or benchmarking purposes.
> In header-only use the defaults are inferred from the compiler's predefined macros under your own `-march` flags; LASX and POWER9 need your `-mlasx` and `-mcpu=power9`, as their intrinsics headers hide without them.
> Setting one to 1 without those flags declares the family's kernels without defining them, which links only against the library that holds them.
> The CMake build, which every binding goes through, resolves them from the `probes/` sources instead — compiling each capability's probe, which calls one of its kernels, to learn whether the toolchain builds them — so the libraries carry every compilable capability.
> The same names work as CMake cache options (`-D STRINGZILLA_TARGET_SVE2=0`) and as Cargo environment variables (`STRINGZILLA_TARGET_SVE2=0 cargo build`); an explicit `1` keeps a capability only where its probe compiles.

__`STRINGZILLA_TARGET_CUDA`, `STRINGZILLA_TARGET_ROCM`, `STRINGZILLA_TARGET_METAL`__:

> Compile the GPU kernels of one vendor: CUDA wherever the unit is compiled as CUDA, by nvcc or by Clang, ROCm wherever it is compiled as HIP, and Metal wherever the build links Metal and Foundation, as `STRINGZILLA_METAL_ARCH` does.
> The Metal kernels travel as embedded source and compile on the device at first use.
> Each defaults to 1 wherever its vendor is present, and setting it to 0 keeps the host API while dropping that vendor's kernels.

__`STRINGZILLA_HEADER_ONLY`__:

> By default, the headers declare the StringZilla library, which compiles every capability once and dispatches among them at runtime, as the shared library and every binding do.
> Set it to 1 to inline the kernels your compiler flags enable instead, calling them by name; the dispatch points and finders then report `sz_missing_library_k`.

__`STRINGZILLA_ALLOW_MISALIGNED_LOADS`__:

> Default is platform-dependent: enabled on x86 (where unaligned accesses are fast), disabled on others by default.
> When enabled, many byte-level operations use word-sized loads, which can significantly accelerate the serial (SWAR) backend.
> Consider enabling it explicitly if you are targeting platforms that support fast unaligned loads.

__`STRINGZILLA_WITH_LIBC`__:

> Both the library and the C headers can be built without LibC.
> This may affect the type resolution system on obscure hardware platforms.

__`STRINGZILLA_WITH_STL`__:

> When using the C++ interface one can disable implicit conversions from `std::string` to `sz::string_t` and back.
> If not needed, the `<string>` and `<string_view>` headers will be excluded, reducing compilation time.

__`STRINGZILLA_BUILD_SHARED`, `STRINGZILLA_BUILD_TEST`, `STRINGZILLA_BUILD_BENCH`, `STRINGZILLA_MARCH`__ for CMake users:

> When compiling the tests and benchmarks, you can explicitly set the target hardware architecture.
> It's synonymous to GCC's `-march` flag and is used to enable/disable the appropriate instruction sets.
> You can also disable the shared library build, if you don't need it.
