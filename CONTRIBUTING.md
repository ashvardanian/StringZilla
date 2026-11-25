# Contributing to StringZilla

Thank you for coming here!
It's always nice to have third-party contributors 🤗
Depending on the type of contribution, you may need to follow different steps.

---

## Directory Tree

```
c/stringzilla/                 The C interface: one .h per kernel family declaring its dispatch points, kernels and finder, beside types.h and capabilities.h
include/stringzilla/           The GPU runtimes, like cuda.cuh, rocm.cuh and metal.h, and the device prelude types.cuh and types.metal
include/stringzilla/*/         Kernels, one file per CPU capability — serial, haswell, neon, rvv, etc. — plus the engines' GPU simt.cuh and metal.* sources
cpp/stringzilla/               The C++ interface over the C one: stringzilla.hpp and its types.hpp
src/stringzilla.c              Library exports shared by every family — versions, statuses, capability queries, the kernel finder
src/capabilities/*.c           Library units, one per CPU capability, each defining that capability's kernels once
src/dispatch/                  Library units, one per kernel family, with its capability lists, dispatch points and finder
src/dispatch.h                 The capability lists' shape and the kernel pick the dispatch units share, internal to the library
src/capabilities/cuda.cu       The engines' cuda kernels and the CUDA device exports, under STRINGZILLA_CUDA_ARCH
src/capabilities/ampere.cu     The engines' ampere kernels, for the codes the build carries from compute capability 8.0
src/capabilities/hopper.cu     The engines' hopper kernels, for the codes the build carries from compute capability 9.0
src/capabilities/blackwell.cu  The engines' blackwell kernels, for the codes the build carries from compute capability 10.0
src/capabilities/rocm.hip      The engines' rocm kernels and the ROCm device exports, under STRINGZILLA_ROCM_ARCH
src/capabilities/cdna3.hip     The engines' cdna3 kernels, for gfx942 where the build carries it
src/capabilities/cdna4.hip     The engines' cdna4 kernels, for gfx950 where the build carries it
src/capabilities/cdna5.hip     The engines' cdna5 kernels, for the targets the build carries among gfx1250 and gfx1251
src/capabilities/metal.c       The engines' metal kernels and the Metal device exports, under STRINGZILLA_METAL_ARCH
src/parallel.h, .c             Tile-parallel runs on each platform's thread pool, compiled into the Python and Node extensions only
probes/                        ISA probe sources, one per capability mirroring src/capabilities/, each calling one of its kernels as the library compiles it
test/                          C++ and Python tests — see test/README.md
bench/                         C++ benchmarks — see bench/README.md
python/                        CPython extension, one unit per kernel family, with python/stringzilla/stringzilla.h as its private header
rust/                          Rust crate, one module per kernel family under rust/stringzilla/, re-exported by rust/stringzilla.rs
javascript/                    Node.js native addon
swift/                         Swift package sources and tests
golang/                        Go cgo bindings
java/                          Java bindings
csharp/                        C# bindings
cmake/                         ISA probe modules, package config templates, and cross-compilation toolchain files
```

For minimal test coverage, check the following scripts:

- `test/main_cpu.cpp` - runs the family suites over the dispatch points and the C++ wrappers, and `test/cross_<arch>.cpp` hold every capability's kernels to the serial ones, built as `stringzilla_test`, `stringzilla_shared_test` and `stringzilla_cpu_header_test`.
- `test/cross_cuda.cu`, `test/cross_rocm.hip` and `test/cross_metal.cpp` - vendor kernels against serial answers, in the same linked test runners through `test/cross_device.hpp`.
- `test/*.py` - tests the Python API against native strings, split per kernel family - `string_types.py`, `find.py`, `sort.py`, `hash.py`, `cipher.py`, `uncased.py`, `utf8_*.py` - with shared helpers in `base.py` and `utf8_helpers.py`.
- `test/main.js`.

At the C++ level all benchmarks also validate the results against the serial kernels and the STL baselines, serving as tests on real-world data.
They have the broadest coverage of the library, and are the most important to keep up-to-date:

- `bench/main.cpp` - runs every family file, then every `bench/cross_<arch>.cpp`, built as `stringzilla_bench`, and header-only as `stringzilla_cpu_header_bench`.
- `bench/token.cpp`, `bench/find.cpp`, `bench/sort.cpp`, `bench/container.cpp`, and the other family files - the dispatch points against the STL and LibC baselines.
- `bench/cross_<arch>.cpp` - every capability's kernels against the serial ones, through the adapters in `bench/cross.hpp`.
- `bench/cross_cuda.cu`, `bench/cross_rocm.hip`, and `bench/cross_metal.cpp` - each GPU vendor's engines against the dispatched CPU entry.

## Benchmarking Datasets

It's not always easy to find good datasets for benchmarking strings workloads.
I use several ASCII and UTF-8 international datasets, all of them mirrored on the HuggingFace dataset hub, in the [StringKilla](https://huggingface.co/datasets/ashvardanian/StringKilla) repository.
You can download them using the following commands:

```sh
wget --no-clobber -O utf8.txt https://huggingface.co/datasets/ashvardanian/StringKilla/resolve/main/utf8.txt?download=true
wget --no-clobber -O leipzig1M.txt https://huggingface.co/datasets/ashvardanian/StringKilla/resolve/main/leipzig1M.txt?download=true
wget --no-clobber -O enwik9.txt https://huggingface.co/datasets/ashvardanian/StringKilla/resolve/main/enwik9.txt?download=true
wget --no-clobber -O xlsum.csv https://huggingface.co/datasets/ashvardanian/StringKilla/resolve/main/xlsum.csv?download=true
```

For bioinformatics workloads, I use the following datasets with increasing string lengths:

```sh
wget --no-clobber -O acgt_100.txt https://huggingface.co/datasets/ashvardanian/StringKilla/resolve/main/acgt_100.txt?download=true
wget --no-clobber -O acgt_1k.txt https://huggingface.co/datasets/ashvardanian/StringKilla/resolve/main/acgt_1k.txt?download=true
wget --no-clobber -O acgt_10k.txt https://huggingface.co/datasets/ashvardanian/StringKilla/resolve/main/acgt_10k.txt?download=true
wget --no-clobber -O acgt_100k.txt https://huggingface.co/datasets/ashvardanian/StringKilla/resolve/main/acgt_100k.txt?download=true
wget --no-clobber -O acgt_1m.txt https://huggingface.co/datasets/ashvardanian/StringKilla/resolve/main/acgt_1m.txt?download=true
wget --no-clobber -O acgt_10m.txt https://huggingface.co/datasets/ashvardanian/StringKilla/resolve/main/acgt_10m.txt?download=true
```

## IDE Integrations

The project was originally developed in VS Code, and contains a set of configuration files for that IDE under `.vscode/`.

- `tasks.json` - build tasks for CMake.
- `launch.json` - debugger launchers for CMake.
- `extensions.json` - recommended extensions for VS Code, including:
  - `ms-vscode.cpptools-themes` - C++ language support.
  - `ms-vscode.cmake-tools`, `cheshirekow.cmake-format` - CMake integration.
  - `ms-python.python`, `ms-python.black-formatter` - Python language support.
  - `yzhang.markdown-all-in-one` - formatting Markdown.
  - `aaron-bond.better-comments` - color-coded comments.

## Code Styling

The project uses `.clang-format` to enforce a consistent code style.
Modern IDEs, like VS Code, can be configured to automatically format the code on save.

- East const over const West.
  Write `char const*` instead of `const char*`.
- For color-coded comments start the line with `!` for warnings or `?` for questions.
- Sort the includes: standard libraries, third-party libraries, and only then internal project headers.
- List in a `@par Statuses:` block only the statuses a function's contract adds, while resource failures, cancellation, deferral and defects are covered once by the categories in the `sz_status_t` docblock.

Every all-caps name starts with the full project name, `STRINGZILLA_`.
A trailing `_` marks a name as internal: it may change in any release, and nothing outside this repository may define or test it.
A name without it is a public contract, either a switch you may set or a value you may read.

| Family                                      | Form                                 | Example                              |
| :------------------------------------------ | :----------------------------------- | :----------------------------------- |
| ISA tier, backend, GPU generation           | `STRINGZILLA_TARGET_<TIER>`          | `STRINGZILLA_TARGET_HASWELL`         |
| Header-only build, without runtime dispatch | `STRINGZILLA_HEADER_ONLY`            |                                      |
| Debug-only checks                           | `STRINGZILLA_DEBUG`                  |                                      |
| Optional facility                           | `STRINGZILLA_WITH_[<OS>_]<FACILITY>` | `STRINGZILLA_WITH_LIBC`              |
| Permission for a liberty                    | `STRINGZILLA_ALLOW_<LIBERTY>`        | `STRINGZILLA_ALLOW_MISALIGNED_LOADS` |
| Architecture fact                           | `STRINGZILLA_ARCH_<ARCH>_`           | `STRINGZILLA_ARCH_X8664_`            |
| Capability whose code compiles here         | `STRINGZILLA_ARCH_<ARCH>_<TIER>_`    | `STRINGZILLA_ARCH_X8664_HASWELL_`    |
| Operating-system fact                       | `STRINGZILLA_OS_<OS>_`               | `STRINGZILLA_OS_LINUX_`              |
| Toolchain fact                              | `STRINGZILLA_HAS_<FEATURE>_`         | `STRINGZILLA_HAS_CLANG_EVEX512_`     |

Architectures are spelled as one token each, `X8664`, `X8632`, `ARM64`, `RISCV64`, `PPC64`, `LOONGARCH64`, `S390X` and `WASM`, and GPU platforms `CUDA` and `ROCM`.
A GPU platform holds beside the host's architecture in both compiler passes, so it never follows a CPU architecture in an `#elif` chain.
Metal has no compiler macro on the host side, so its host API is a switch the build sets where it links Metal and Foundation.
Every name in these families is always defined, as 0 or 1, and tested with `#if`.
`defined(...)` appears only in a `!defined` default and in a guard that an `#error` follows.
Only the Platform Identity region of `c/stringzilla/types.h` reads operating-system predefines like `__linux__`.
Compiler predefines like `__clang__`, `__GNUC__` and `_MSC_VER` are read raw where they matter.

C names put the capability last, `sz_<family>_<what>_<capability>`, like `sz_find_haswell`.
Only a role suffix may follow it: `_t` for a type, `_k` for a constant, `_kernel_` for a GPU entry point, or the trailing `_` of an internal name, like `sz_levenshtein_u64x1_sweep_serial_` and `sz_overlap_scores_blackwell_kernel_`.
A dispatch point puts `best` in the capability's place, like `sz_strings_lookup_best`, and the kernels it picks from are its twins, like `sz_strings_lookup_haswell`.

Code that several capabilities share belongs to one of three layers, named in the capability's place:

- `serial`, portable and capability-neutral, in each family's `serial.h`.
- `simt`, the single C source that CUDA and HIP both compile, in each family's `simt.cuh`.
- `metal`, what every Metal tier shares, in each family's `metal.h` and the `metal.metal` shaders it embeds.

A constant identical across the GPU layers is defined once in `serial.h` with the adjective `gpu`, like `sz_levenshtein_gpu_warp_lanes_k`.
One whose value differs per layer takes its layer instead, like `sz_substrings_tally_slot_bits_simt_k` and `sz_substrings_tally_slot_bits_metal_k`.

Every capability has two flags, CPU and GPU alike.
`STRINGZILLA_ARCH_<ARCH>_<CAPABILITY>_` says the capability's code compiles in this unit, and never reads a target.
`c/stringzilla/types.h` detects it: for a CPU capability from the unit's own flags, and for a GPU generation from compiling as its vendor; only this library's build raises or zeroes a CPU flag, with its probe's verdict.
A base's flag also holds wherever a flag built on it holds, so Haswell's building blocks compile wherever Skylake's do.
`STRINGZILLA_TARGET_<CAPABILITY>` says this unit has the capability's kernels, by default a literal from the same detection on the CPU, the vendor flag for `cuda` and `rocm`, and 0 for a GPU generation.
Every capability header has one shape: its architecture or vendor guard, its `STRINGZILLA_ARCH_<ARCH>_<CAPABILITY>_` guard, its building blocks, then its kernels under `STRINGZILLA_TARGET_<CAPABILITY>`.
Building blocks, the `STRINGZILLA_DEVICE` steps other families and products call, are `static __forceinline__` device functions, which a device never emits unless a kernel calls them.
A table one building block alone reads lives inside it, so only units calling the block emit it, and a table only kernels name sits with them and reaches the steps through a pointer, like `sz_utf8_norm_tables_cuda_`.
`STRINGZILLA_TARGET_<CAPABILITY>` guards only what a launch reaches, as a device keeps every kernel a unit defines: kernels, tables only kernels read, entry-point tables, host launchers and `STRINGZILLA_API` entries, in regions after every building block.
No kernel sits under an `||` of targets: a body several tiers launch is a `STRINGZILLA_DEVICE` building block, in the oldest of their files or in `simt.cuh` and `kernels.cuh` for both vendors, and each tier's file wraps it in its own kernel, like `sz_regex_bound_ampere_kernel_` over `sz_regex_bound_simt_`, which its entry-point table points at.
Within the guards, `extern "C"` opens after the includes and before the pragma push, which wraps the building blocks and the target region; a header of kernels alone pushes it first inside the target region instead, with a comment on that `#if`, as an empty push region fails under -Werror.
The root `cuda.cuh` and `rocm.cuh` hold the runtime services and the baseline lane steps, and `strings/<generation>.cuh`, like `strings/ampere.cuh` and `strings/cdna5.cuh`, the steps a generation added.
An instruction lives in the lowest family that needs it, in its generation's file, and the families above include that file.

The GPU capability groups are `cuda`, `rocm` and `metal`, beside the CPU's `cpu`.
A later generation in a group takes its own bit and unit, like `sz_cap_blackwell_k` and `src/capabilities/blackwell.cu`, which every device of that generation and after reports beside the baseline.
A ROCm generation bit, like `sz_cap_cdna3_k`, is reported only on the targets that have its DPP form.
Each word names its group's baseline bit, like `sz_cap_cuda_k`, its functions and its library unit, like `src/capabilities/cuda.cu`, and no symbol or source path names a vendor, like `nvidia`, `amd` or `apple`.
A function that touches a device is either a producer or a consumer:

- A __producer__ reports a device's capabilities or opens a stream on it.
  Device queries end with their group. Producers are the only functions that take a device's `ordinal`: `sz_count_devices_cuda(&count)`, `sz_capabilities_enabled_cuda(ordinal, &capabilities)` and `sz_stream_init_cuda(ordinal, &stream)`.
- A __consumer__ takes the `capabilities` it picks from and, where it queues work, a trailing `sz_stream_t stream`, but never an ordinal, since the stream names its device: `sz_stream_synchronize_best(capabilities, stream)`.
  It ends in `best` like any dispatch point, and its twins end in their capability, like `sz_stream_synchronize_cuda(stream)`.

A null stream is the default stream of the default device: the calling thread's current device on CUDA and ROCm, and the system default device on Metal.
On the CPU the stream must be null.

No verb allocates.
A GPU verb takes any memory its device reaches, and the runtime checks that per call.
Unified memory comes only from an explicit `copy`, or from `UnifiedAllocator` in Rust.
A GPU verb runs the capabilities the caller names.

Each of these words has one meaning across the library:

| Word                                   | Meaning                                                                          | Appears as                                                                     |
| :------------------------------------- | :------------------------------------------------------------------------------- | :----------------------------------------------------------------------------- |
| `cpu`, `cuda`, `rocm`, `metal`         | A capability group and its baseline bit                                          | Device-query, twin and kernel suffix, `src/capabilities/<group>.*`             |
| `simt`                                 | The single C source CUDA and HIP both compile                                    | `<family>/simt.cuh`, the suffixes `_simt_`, `_simt_t`, `_simt_k`               |
| `metal`, as a layer                    | What every Metal tier shares                                                     | `<family>/metal.h`, `<family>/metal.metal`, the suffix `_metal_`               |
| `gpu`                                  | Adjective for every GPU group                                                    | `sz_cap_gpus_k`, `sz_missing_gpu_k`, `sz_levenshtein_gpu_warp_lanes_k`         |
| `device`                               | A processor kernels run on, which a stream belongs to                            | `sz_count_devices_cuda`, `sz_device_memory_mismatch_k`, `sz_device_enter_simt_` |
| `ordinal`                              | A device's index within its group, as its runtime numbers it                     | Producer parameters only                                                       |
| `stream`                               | A `cudaStream_t`, `hipStream_t` or `id<MTLCommandQueue>`, which names its device | The trailing `void *stream` of every consumer                                  |
| `queue`                                | A tile work queue, never a stream                                                | `sz_tile_queue_t`                                                              |
| `unified`                              | Memory both the host and the stream's device address                             | `sz_allocator_init_unified_best`                                               |
| `tape`                                 | Strings' bytes back to back, and the `count + 1` offsets into them               | What `sz_strings_to_tape_best` writes                                          |
| A capability, like `haswell` or `cuda` | One bit of a mask                                                                | The last token before the role suffix                                          |
| `kernel`                               | A GPU entry point                                                                | `_kernel_`, right after the capability                                         |

For C++ code:

- Explicitly use `std::` or `sz::` namespaces over global `memcpy`, `uint64_t`, etc.
- Mark every library interface `noexcept`, except `__global__` CUDA functions, stream output and conversions into STL types, whose failures belong to the STL.
- Return a `status_t` or an `expected` from every fallible interface, never throw, and clamp positions past the end like Python's slices.
- Avoid C-style variadic arguments in favor of templates.
- Avoid C-style casts in favor of `static_cast`, `reinterpret_cast`, and `const_cast`, except for places where a C function is called.
- Use lower-case names for everything, except settings/conditions macros.
  Function-like macros, that take arguments, should be lowercase as well.
- Macros expand expressions and never define functions.
- In templates prefer `typename` over `class`.

For Python code:

- Use lower-case names for functions and variables.

## Lean State

A type holds only what it cannot recompute, and names each part for what it holds.
These rules are shared by every sibling library, and the pre-commit hook enforces each rule that names a check.

- Members are named for what they hold, never for their type.
  Write `request->call` or `call->kept`, never `call->call` or `file.file`.
  Checked by `check_self_repeat_access`.
- Derived state is a method, not a field.
  A count, mask or capacity that follows from other members is computed when asked, so it can never disagree with them.
- One fact, one constant.
  Every `_k` constant states its derivation in its docblock, or says "measured" and where, and no other constant restates it.
- An owner that never moves holds its parts by value.
  A pointer is for a part that outlives its owner, moves apart from it, or is shared beyond it.
- Empty allocators take no bytes.
  An allocator member is declared `[[no_unique_address]]`, so a container over a stateless allocator is no larger than its own state.
  Checked by `check_empty_member_bytes`.
- No wrapper without its own state or invariant.
  A type that only forwards to one member is a name for that member, so use an alias or a function instead.
- Concepts are `a_`/`an_` noun phrases.
  A concept names what satisfies it, as `an_allocator` or `a_task_callback`: `an_` before a vowel sound, singular, with no `is_` or `has_` prefix and no `_concept` or `_t` suffix.
  Checked by `check_concept_names`.

## C++ and C

The primary C implementation and the C++ wrapper are built with CMake.
StringZilla builds with GCC 13, Clang 18, Apple Clang 16, MSVC 19.40 or newer.
On Ubuntu 24.04 the default packages are enough:

```bash
sudo apt-get install cmake g++ gcc
```

On Linux, after that, if you want to compile the minimal set of tests, use the presets CI uses:

```bash
cmake --preset release
cmake --build --preset release --target stringzilla_test
ctest --preset release
```

`cmake --list-presets` shows the rest - `debug`, `cuda`, `rocm`, `metal`, `swift`, the `linux_<arch>` cross builds, and WASI.
Machine-specific settings, like a CUDA host compiler, belong in an untracked `CMakeUserPresets.json`.
Without a preset, a configure builds the libraries alone, as every binding's build does: `-D STRINGZILLA_BUILD_TEST=1` and `-D STRINGZILLA_BUILD_BENCH=1` add the suites, and `-D STRINGZILLA_CUDA_ARCH=all`, `STRINGZILLA_ROCM_ARCH` or `STRINGZILLA_METAL_ARCH` the GPUs.

Every binding builds the libraries through `CMakeLists.txt`, so CMake is the one place that probes which capabilities the toolchain builds.
Each capability's probe, `probes/<capability>.c`, calls one of its kernels, compiled header-only at the baseline flags as the library compiles it.
Each capability's headers scope its kernels with `#pragma clang attribute` or `#pragma GCC target`, on every platform.
LASX and POWER9 scope per function only under GCC 15 or newer, whose target pragma defines `__loongarch_asx` and `__POWER9_VECTOR__` inside the region, so `types.h` opens `lasxintrin.h` and `altivec.h` there; Clang's headers stay closed without `-mlasx` or `-mcpu=power9`, so a Clang build leaves both off.
The libraries compile every capability the toolchain builds and dispatch by runtime detection, so one artifact runs on any CPU of its architecture.
A monoversioned build, `-D STRINGZILLA_MULTIVERSIONED=OFF`, keeps only the capabilities whose predefined macros `STRINGZILLA_MARCH` implies, as `probes/implied.c` reads them through `types.h`, and folds each dispatch table to its best kernel.
The `xeon4_hopper`, `xeon6_blackwell`, `venice_mi450`, `grace_hopper` and `vera_rubin` presets build that way for one machine and its one GPU code, spelling every capability as 1 or 0 rather than trusting the macros.
`-D STRINGZILLA_TARGET_<CAPABILITY>=0` drops a capability, and `=1` keeps one only where its probe compiles, in a monoversioned build vouching for what no compiler macro reports.
Units calling capability kernels link `stringzilla::cpu_capabilities_compiled`, the build-tree twin of `sz_capabilities_compiled_cpu()`: every CPU capability the toolchain compiles.
`stringzilla::header` carries no verdict, so a header-only consumer enables what its own flags name, and the runtime mask picks the compiled capabilities the CPU runs.
Each capability's flags for a unit compiled whole for it are cached as `stringzilla_target_<capability>_flags`.

Six words name a capability's states, from configure to dispatch:

| Word | Question | When | Scope |
| --- | --- | --- | --- |
| __probe__ | Can the toolchain build this capability? | configure: `try_compile` of `probes/<capability>.c`, then compile: `STRINGZILLA_ARCH_<ARCH>_<CAPABILITY>_` | build |
| __target__ | Is this capability's unit in the build? | compile: `STRINGZILLA_TARGET_<CAPABILITY>` | build |
| __implied__ | What do the baseline flags already guarantee? | compile, from predefined macros, and at runtime where the OS cannot be asked | build |
| __compiled__ | Does the binary carry this capability's code for this processor? | runtime: `sz_capabilities_compiled_<group>` | CPU: the build; GPU: each device |
| __detected__ | Does this processor offer the capability? | runtime: `sz_capabilities_detected_<group>` | processor |
| __enabled__ | What may dispatch pick? | runtime: `sz_capabilities_enabled_<group>`, detected and compiled | processor |

GPU `compiled` is per device, as only GPU binaries carry code per architecture: each GPU unit exports one of its kernels, and `sz_capabilities_compiled_cuda` or `sz_capabilities_compiled_rocm` asks the runtime whether it loads that kernel on the device, through `cudaFuncGetAttributes` or `hipFuncGetAttributes`.
A monoversioned build names one machine: one CPU and at most one GPU architecture per vendor, shared by every GPU in it.
It compiles the best capability for that machine and folds every dispatch row to it, so link-time optimization drops every other kernel.
On any other machine a call answers `sz_missing_kernel_k`, never a crash and never a slower kernel in silence.
Sibling GPU capabilities form no chain, so with `STRINGZILLA_MULTIVERSIONED=OFF` a configure listing two CUDA or two ROCm architectures fails.

The baseline is each architecture's floor: `-march=x86-64`, `-march=armv8-a`, `-march=rv64gc`, `-mcpu=power8`, and `-march=loongarch64`.
WebAssembly is the exception, as an engine validates a module whole, so its one capability, `STRINGZILLA_MARCH`, reaches every unit.
The RVV kernels need Clang 21 or newer; an older compiler probes those capabilities as 0.
The `linux_arm64` and `linux_riscv64` presets cross-compile with Clang, through `cmake/toolchain-<arch>-llvm.cmake`, and `linux_ppc64le` and `linux_loongarch64` with GCC 15, through `cmake/toolchain-<arch>-gnu.cmake`, as only GCC scopes POWER9 and LASX per function; each runs the tests under QEMU.
Each toolchain emulates the richest CPU by default, so the tests reach every capability, and its `<ARCH>_QEMU_CPU` variable, like `-D PPC_QEMU_CPU=power8`, runs them on the floor instead.

On macOS it's recommended to use Homebrew and install Clang, as opposed to "Apple Clang".
Replacing the default compiler is not recommended, as it may break the system, but you can pass it as an environment variable:

```bash
brew install llvm
cmake -D CMAKE_BUILD_TYPE=Release -D STRINGZILLA_BUILD_TEST=1 \
    -D CMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
    -D CMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
    -B build_release
cmake --build build_release --config Release --parallel
```

On Windows you can build with either MSVC (Visual Studio) or MinGW (GCC).
Pick one.
For MSVC (Developer Prompt):

```bat
cmake -B build_release -G "Visual Studio 18 2026" -A x64 -D STRINGZILLA_BUILD_TEST=1 -D CMAKE_BUILD_TYPE=Release
cmake --build build_release --config Release --parallel
build_release\\Release\\stringzilla_test.exe
```

For MinGW (MSYS2):

```bash
pacman -S --needed --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake make
cmake -G "MinGW Makefiles" -B build_release -D STRINGZILLA_BUILD_TEST=1 -D CMAKE_BUILD_TYPE=Release
cmake --build build_release --config Release --parallel
./build_release/stringzilla_test.exe
```

### Benchmarking

For benchmarks, you can use the following commands:

```bash
cmake -D STRINGZILLA_BUILD_BENCH=1 -B build_release
cmake --build build_release --config Release --parallel # Produces the following targets:
build_release/stringzilla_bench                     # - every family's dispatch points, then every capability's kernels
build_release/stringzilla_cpu_header_bench              # - the kernels header-only, with the helpers the library keeps private
```

Each family times its dispatch points, like `sz_find_best`, against the standard baselines, like `find<std::strstr>`.
Each capability's kernels, like `sz_find_haswell`, are timed against the serial ones, like `sz_find_serial`, and skipped on a CPU that lacks the capability.
The header-only twin has no dispatch points to time, but it reaches the private helpers, like the SIMD sorts of pgrams and the Levenshtein and overlap steps.

The same `stringzilla_bench` runs CPU rows once and then all selected GPU backends enabled by
`STRINGZILLA_CUDA_ARCH`, `STRINGZILLA_ROCM_ARCH`, and `STRINGZILLA_METAL_ARCH`.
CUDA and ROCm can coexist in one build. `STRINGWARS_DEVICES=cuda:0,rocm:1` selects vendor-qualified
ordinals; unset, each available compiled vendor runs device zero. Explicit unavailable devices are errors.
The tests follow the same rule with `stringzilla_test` and `STRINGZILLA_DEVICES`.

GPU rows are named after the kernel they time, like `sz_levenshtein_distances_cuda:0:q1024:w16:sorted` or `sz_overlap_scores_metal:0:w5`, which is what `STRINGWARS_FILTER` matches.

All of them support customization via environment variables.
Let's say you want to benchmark large-batch DNA edit distances:

```sh
cmake -D STRINGZILLA_BUILD_BENCH=1 -B build_release
cmake --build build_release --config Release --target stringzilla_bench --parallel
STRINGWARS_FILTER="sz_levenshtein_distances" STRINGWARS_DATASET="acgt_1k.txt" build_release/stringzilla_bench
STRINGWARS_DEVICES=cuda:0 STRINGWARS_FILTER="sz_levenshtein_distances_cuda" STRINGWARS_DATASET="acgt_100k.txt" build_release/stringzilla_bench

STRINGWARS_FILTER="sz_levenshtein_distances_cuda:0:q[0-9]+:w" STRINGWARS_DATASET="acgt_1k.txt" build_release/stringzilla_bench
STRINGZILLA_STRESS=0 STRINGWARS_FILTER="sz_levenshtein_distances_cuda:0:q1024:w16" STRINGWARS_DATASET="acgt_100k.txt" build_release/stringzilla_bench
```

The benchmark harness reads these environment variables, and a flag overrides each:

| Variable                        | Flag                  | Default     | Meaning                                                                              |
| :------------------------------ | :-------------------- | :---------- | :----------------------------------------------------------------------------------- |
| `STRINGWARS_DEVICES`            | `--devices`           | unset       | Comma-separated `backend:ordinal` devices, like `cuda:0,rocm:1`                      |
| `STRINGWARS_DATASET`            | `--dataset`           | per corpus  | Corpus file, instead of `leipzig1M.txt` or `xlsum.csv`                               |
| `STRINGWARS_TOKENS`             | `--tokens`            | per corpus  | `file`, `lines`, `words`, or an N-gram length like `64`                              |
| `STRINGWARS_BYTES`              | `--bytes`             | whole file  | Bytes read from the corpus, like `4096`, `64KB` or `1GB`; UTF-8 families read `64MB` |
| `STRINGWARS_UNIQUE`             | `--unique`            | `0`         | Sorts the tokens and drops duplicates first                                          |
| `STRINGWARS_FILTER`             | `--filter`            | unset       | ECMAScript regex over benchmark names                                                |
| `STRINGWARS_SEED`               | `--seed`              | `42`        | Seed for every random generator, or `random` to draw one, which a run prints         |
| `STRINGWARS_WARMUP`             | `--warmup`            | `1s`        | Untimed run before each benchmark, like `200ms` or `1s`                              |
| `STRINGWARS_TIME_LIMIT`         | `--time-limit`        | `10s`       | Timed run of each benchmark, like `200ms` or `10s`; `1s` in debug builds             |
| `STRINGWARS_BATCH_PER_CORE`     | `--batch-per-core`    | per backend | Candidates per call per CPU core, CUDA or ROCm multiprocessor, or Metal GPU core     |
| `STRINGZILLA_STRESS`            | `--stress`            | `1`         | Checks each backend against its baseline before timing                               |
| `STRINGZILLA_STRESS_TIME_LIMIT` | `--stress-time-limit` | time limit  | Time of each stress check, like `200ms` or `10s`                                     |
| `STRINGZILLA_STRESS_DIR`        | `--stress-dir`        | `.tmp`      | Directory for stress-check failure logs                                              |
| `STRINGZILLA_STRESS_LIMIT`      | `--stress-limit`      | `2`         | The stress-check failure that stops the run, so `1` stops at the first               |

For a fast inner loop, scope to one backend on a small dataset, cap the dataset size, skip the stress phase, and use short runs:

```bash
STRINGWARS_FILTER='sz_find' STRINGWARS_DATASET=leipzig1M.txt \
    STRINGWARS_BYTES=64KB STRINGWARS_BATCH_PER_CORE=1024 \
    STRINGZILLA_STRESS=0 STRINGWARS_TIME_LIMIT=1s \
    build_release/stringzilla_bench
```

Throughput is a time-bounded measurement: absolute GB/s drifts ±10-15% on a loaded machine, while the ratio between two backends in the _same_ run stays stable.
Compare A/B within one run; raise `STRINGWARS_TIME_LIMIT` and use a quiet machine when you need stable absolute numbers.
The work itself is deterministic: the tokens keep their file order, and a given `STRINGWARS_SEED` always draws the same samples from them.

Each family's benchmarks live in an identically named file in the `bench/` directory, and their kernel adapters in `bench/cross.hpp`.
All of them feature file-level documentation, and are designed to be self-explanatory.
You can easily log their descriptions until the first `*/` with the following `sed` and `awk` commands:

```sh
sed '/\*\//q' bench/strings.cpp
awk '/\*\// { exit } { print }' bench/strings.cpp
```

### Benchmarking Hardware-Specific Optimizations

Running on modern hardware, you may want to compile the code for older generations to compare the relative performance.
The assumption would be that newer ISA extensions would provide better performance.
On x86_64, you can use the following commands to compile for Sandy Bridge, Haswell, and Sapphire Rapids:

```bash
cmake -D CMAKE_BUILD_TYPE=Release -D STRINGZILLA_BUILD_BENCH=1 \
    -D STRINGZILLA_MARCH="ivybridge" -B build_release/ivybridge && \
    cmake --build build_release/ivybridge --config Release --parallel
cmake -D CMAKE_BUILD_TYPE=Release -D STRINGZILLA_BUILD_BENCH=1 \
    -D STRINGZILLA_MARCH="haswell" -B build_release/haswell && \
    cmake --build build_release/haswell --config Release --parallel
cmake -D CMAKE_BUILD_TYPE=Release -D STRINGZILLA_BUILD_BENCH=1 \
    -D STRINGZILLA_MARCH="sapphirerapids" -B build_release/sapphirerapids && \
    cmake --build build_release/sapphirerapids --config Release --parallel
```

### Benchmarking Compiler-Specific Optimizations

Alternatively, you may want to compare the performance of the code compiled with different compilers.
On x86_64, you may want to compare GCC, Clang, and ICX.

```bash
cmake -D CMAKE_BUILD_TYPE=Release -D STRINGZILLA_BUILD_BENCH=1 -D STRINGZILLA_BUILD_SHARED=1 \
    -D CMAKE_CXX_COMPILER=g++ -D CMAKE_C_COMPILER=gcc \
    -B build_release/gcc && cmake --build build_release/gcc --config Release --parallel
cmake -D CMAKE_BUILD_TYPE=Release -D STRINGZILLA_BUILD_BENCH=1 -D STRINGZILLA_BUILD_SHARED=1 \
    -D CMAKE_CXX_COMPILER=clang++-18 -D CMAKE_C_COMPILER=clang-18 \
    -B build_release/clang && cmake --build build_release/clang --config Release --parallel
```

### Profiling

To simplify tracing and profiling, build with symbols using the `RelWithDebInfo` configuration.
Here is an example for profiling the hashing benchmarks of `stringzilla_bench`.

```bash
cmake -D STRINGZILLA_BUILD_BENCH=1 \
    -D STRINGZILLA_BUILD_TEST=1 \
    -D STRINGZILLA_BUILD_SHARED=1 \
    -D CMAKE_BUILD_TYPE=RelWithDebInfo \
    -B build_profile
cmake --build build_profile --config Release --target stringzilla_bench --parallel

# Check that the debugging symbols are there with your favorite tool
readelf --sections build_profile/stringzilla_bench | grep debug
objdump -h build_profile/stringzilla_bench | grep debug

# Profile
sudo env STRINGWARS_DATASET=./leipzig1M.txt STRINGWARS_FILTER='sz_hash' perf record -g build_profile/stringzilla_bench
sudo perf report
```

### Testing in Docker

It might be a good idea to check the compatibility against the most popular Linux distributions.
Docker is the goto-choice for that.

#### Alpine

Alpine is one of the most popular Linux distributions for containers, due to its size.
The base image is only ~3 MB, and it's based on musl libc, which is different from glibc.

```bash
sudo docker run -it --rm -v "$(pwd)":/workspace/StringZilla alpine:latest /bin/ash
cd /workspace/StringZilla
apk add --update make cmake g++ gcc
cmake -D STRINGZILLA_BUILD_TEST=1 -D CMAKE_BUILD_TYPE=Debug -B build_debug
cmake --build build_debug --config Debug --parallel
build_debug/stringzilla_test
```

#### Intel Clear Linux

Clear Linux is a distribution optimized for Intel hardware, and is known for its performance.
It has rolling releases, and is based on `glibc`.
It might be a good choice for compiling with Intel oneAPI compilers.

```bash
sudo docker run -it --rm -v "$(pwd)":/workspace/StringZilla clearlinux:latest /bin/bash
cd /workspace/StringZilla
swupd update
swupd bundle-add c-basic dev-utils
cmake -D STRINGZILLA_BUILD_TEST=1 -D CMAKE_BUILD_TYPE=Debug -B build_debug
cmake --build build_debug --config Debug --parallel
build_debug/stringzilla_test
```

For benchmarks:

```bash
cmake -D STRINGZILLA_BUILD_TEST=1 -D STRINGZILLA_BUILD_BENCH=1 -B build_release
cmake --build build_release --config Release --parallel
```

#### Amazon Linux

For CentOS-based __Amazon Linux 2023__:

```bash
sudo docker run -it --rm -v "$(pwd)":/workspace/StringZilla amazonlinux:2023 bash
cd /workspace/StringZilla
yum install -y make cmake3 gcc g++
cmake3 -D STRINGZILLA_BUILD_TEST=1 -D CMAKE_BUILD_TYPE=Debug \
    -D CMAKE_CXX_COMPILER=g++ -D CMAKE_C_COMPILER=gcc -D STRINGZILLA_MARCH="ivybridge" \
    -B build_debug
cmake3 --build build_debug --config Debug --target stringzilla_test
build_debug/stringzilla_test
```

---

Don't forget to clean up Docker afterwards.

```bash
docker system prune -a --volumes
```

### Cross Compilation

Unlike GCC, LLVM handles cross compilation very easily.
You just need to pass the right `TARGET_ARCH` and `BUILD_ARCH` to CMake.
The [list includes](https://packages.ubuntu.com/search?keywords=crossbuild-essential&searchon=names):

- `crossbuild-essential-amd64` for 64-bit x86
- `crossbuild-essential-arm64` for 64-bit Arm
- `crossbuild-essential-armhf` for 32-bit ARM hard-float
- `crossbuild-essential-armel` for 32-bit ARM soft-float (emulates `float`)
- `crossbuild-essential-riscv64` for RISC-V
- `crossbuild-essential-powerpc` for PowerPC
- `crossbuild-essential-s390x` for IBM Z
- `crossbuild-essential-mips` for MIPS
- `crossbuild-essential-ppc64el` for PowerPC 64-bit little-endian

Here is an example for cross-compiling for Arm64 on an x86_64 machine:

```sh
sudo apt-get update
sudo apt-get install -y clang lld make crossbuild-essential-arm64 crossbuild-essential-armhf
export CC="clang"
export CXX="clang++"
export AR="llvm-ar"
export NM="llvm-nm"
export RANLIB="llvm-ranlib"
export TARGET_ARCH="aarch64-linux-gnu" # Or "x86_64-linux-gnu"
export BUILD_ARCH="arm64" # Or "amd64"

cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER_TARGET=${TARGET_ARCH} \
    -D CMAKE_CXX_COMPILER_TARGET=${TARGET_ARCH} \
    -D CMAKE_SYSTEM_NAME=Linux \
    -D CMAKE_SYSTEM_PROCESSOR=${BUILD_ARCH} \
    -B build_artifacts
cmake --build build_artifacts --config Release --parallel
```

### WebAssembly

Two toolchain files under `cmake/` cover WebAssembly, and both need the [wasi-sdk](https://github.com/WebAssembly/wasi-sdk/releases) and [Wasmtime](https://wasmtime.dev).
`toolchain-wasm32-wasi.cmake` builds `wasm32-wasip1` modules, single-threaded, for the single-string core.
`toolchain-wasm32-wasi-threads.cmake` builds `wasm32-wasip1-threads` modules over one shared memory, for a caller that shards work across threads itself; nothing in the library needs them, and the CI wasm jobs use the single-threaded file above.
Point either file at the SDK with `-DWASI_SDK_PREFIX=...` or the `WASI_SDK_PATH` environment variable.

```sh
export WASI_SDK_PATH=~/wasi-sdk
cmake -B build_wasm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-wasi.cmake \
    -DSTRINGZILLA_BUILD_TEST=1 -DSTRINGZILLA_BUILD_SHARED=0 -DCMAKE_BUILD_TYPE=Release
cmake --build build_wasm --target stringzilla_test
ctest --test-dir build_wasm # runs each .wasm under Wasmtime
```

A module carries one SIMD capability, `STRINGZILLA_MARCH`, so the relaxed-SIMD `v128relaxed` build above is a separate artifact from the strict `v128` one and the SIMD-free `serial` one.
Pass `-DSTRINGZILLA_MARCH=v128` or `=serial`, or set the environment variable, for the others; CMake turns the capability into `-msimd128` and `-mrelaxed-simd` for every unit, so no other opcode reaches the binary.
Shared libraries stay off in both configurations, since WASI has no dynamic loader.

## Test Suites

### Principles

Tests reach the library through its public surface: the C ABI in `c/stringzilla/`, `stringzilla.hpp`, the capability kernels the public headers declare by name, and the `pub` items of the Python module and the Rust crate.
The few private includes are listed under [Private Surface Waivers](#private-surface-waivers), each with what the public surface lacks.
A kernel is never its own oracle: each capability is held to the serial kernels or to known answers, and the inputs are never drawn through the kernels under test.
Oracles never compile out, so a test checks with `expect_true` and its siblings, never `assert`, and every test unit opens with `#undef NDEBUG // ! A test's oracle must stay live in every build`.
Benchmarks never include `test/`, each tree keeps its own harness, and neither uses a test or benchmark framework.
One test binary carries every vendor's GPU shard, and the devices it checks are picked at runtime.
The harness owns stdout, while test bodies write their diagnostics to stderr.
Settings are read once, and a value that does not parse prints one line naming it and exits with 1.
Skips are honest: a filter that selects no case fails the run, a case that cannot run says why, and a test missing a dependency names it.

The C++ harness holds one mutable global, `test_run`, the run in progress: its settings, the case under way, the count of failed checks with the last one's file and line, and the passed, failed and skipped cases.
Checks fire on worker threads and inside helpers as well as in a case's own body, so `run_test_case` diffs the failed-check count around each case rather than threading a tally through every call.
A debugger reads it from a core dump as `p 'ashvardanian::stringzilla::test::test_run'`, and the crash handlers print the case in progress before the backtrace.

### Running

```bash
cmake --preset debug             # Clang, with AddressSanitizer
cmake --build --preset debug
ctest --preset debug
```

The build produces three binaries over the same suites:

```bash
build_debug/stringzilla_test            # Dispatch points, C++ wrappers, and every kernel of the static library
build_debug/stringzilla_shared_test     # Dispatch points alone, through the shared library
build_debug/stringzilla_cpu_header_test # Kernels compiled header-only, and the dispatch-point stubs
```

The `cuda`, `rocm` and `metal` presets add each vendor's GPU checks to `stringzilla_test`.
`STRINGZILLA_CUDA_ARCH` and `STRINGZILLA_ROCM_ARCH` may both be set in one configure, which links both vendors' shards into the same binary.
There is no separate SIMD-disabled target: `-D STRINGZILLA_TARGET_<CAPABILITY>=0` at configure time leaves a capability out, for example `-D STRINGZILLA_TARGET_SKYLAKE=0 -D STRINGZILLA_TARGET_ICELAKE=0` for x86 without AVX-512.
`-D STRINGZILLA_MARCH=<name>`, like `x86-64-v3` or `native`, raises the libraries' floor in place of the portable one, without narrowing the capabilities.
Address Sanitizers have a hard time with masked load and store instructions in AVX-512 and SVE.

A run prints the library and the machine, then its settings, one verdict line per case, and a summary:

```text
StringZilla 6.0.0
- Compiled for: serial,neon,neonaes,neonsha,sve,sve2,sve2aes
- This machine: serial,neon,neonaes,neonsha,sve,sve2,sve2aes
- Seed: 42, scale: 1, filter: none
- test_hash_unit ... ok (0.01 s)
- check_find_unit_(find_haswell) ... skipped: needs sz_cap_haswell_k
412 passed, 0 failed, 37 skipped, seed 42
```

A failed check prints where it failed, with both operands for `expect_equal`, and the case goes on, so one run lists every mismatch it reaches.
A case with a failed check prints its verdict and a `rerun:` line on stderr, naming the binary, the seed, and a filter that selects only that case, then the run moves on:

```text
  check failed at /path/to/StringZilla/test/utf8_runes.cpp:54
    expected 6, got 5
- test_utf8_runes_unit ... FAILED: 1 failed check(s)
  rerun: build_debug/stringzilla_test --seed 42 --filter '^test_utf8_runes_unit$'
```

The run exits with 1 when a case failed, a check failed outside any case, or the filter selected nothing, and with 3 when `STRINGZILLA_NO_SKIPS=1` refused a skip.
`STRINGZILLA_TRAP_ON_FAILURE=1` raises `SIGTRAP` at the first failed check instead, which stops a debugger there, or dumps a core without one:

```bash
STRINGZILLA_TRAP_ON_FAILURE=1 STRINGZILLA_FILTER='^test_utf8_runes_unit$' gdb -ex run --args build_debug/stringzilla_test
```

A library assertion or a crash stops the run with `*** SIGSEGV in case '<label>' - backtrace follows ***` and a backtrace where the C library has one.
Useful breakpoints are `__asan::ReportGenericError` for illegal memory accesses, `__builtin_unreachable` for unexpected code paths, and `sz_assert_failure_` for StringZilla's logic assertions.

The scalable-vector backends must stay correct at every hardware vector length, and the CI `test_cross_qemu` matrix sweeps them all.
The same sweep runs locally with user-mode QEMU - on an Arm host that covers NEON-only dispatch plus SVE at 128/256/512 bits, and cross-compilers unlock the RISC-V, x86, LoongArch, and POWER backends too:

```bash
sudo apt install qemu-user gcc-x86-64-linux-gnu gcc-riscv64-linux-gnu gcc-loongarch64-linux-gnu gcc-powerpc64le-linux-gnu

# Sweep SVE vector lengths on the native Arm binary (sve-max-vq is VL/128)
for vq in 1 2 4; do
  STRINGZILLA_FILTER='^test_utf8_' STRINGZILLA_SCALE=0.1 qemu-aarch64 -cpu max,sve-max-vq=$vq build_release/stringzilla_test
done

# NEON-only dispatch (otherwise SVE2 always wins and NEON is never exercised)
qemu-aarch64 -cpu max,sve=off build_release/stringzilla_test

# Cross-compile a single-TU probe against another backend and run it emulated
x86_64-linux-gnu-gcc -O2 -mavx2 -mbmi -mbmi2 -mpopcnt -DSTRINGZILLA_TARGET_HASWELL=1 -Ic -Iinclude probe.c -o probe -static
qemu-x86_64 -cpu max ./probe
```

QEMU's TCG cannot execute AVX-512, so the Ice Lake and Skylake backends are compile-checked only under emulation and need real x86 hardware to run.

CppCheck and Clang-Tidy are too noisy for CI, but may be useful locally over an exported compilation database:

```bash
sudo apt install cppcheck clang-tidy-11
cmake -B build_artifacts -D CMAKE_BUILD_TYPE=RelWithDebInfo -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -D STRINGZILLA_BUILD_BENCH=1 -D STRINGZILLA_BUILD_TEST=1
cppcheck --project=build_artifacts/compile_commands.json --enable=all
clang-tidy-11 -p build_artifacts
```

The Python suite runs as the CI runs it, after installing the `test` group, and `test-oracles` for the reference implementations.
NumPy is not a required dependency, since StringZilla covers more platforms than NumPy does, so the tests that use it skip without it, as do the suites cross-checking against an oracle that is absent:

```bash
uv pip install --group test --group test-oracles                 # the suite, NumPy and the oracles
uv pip install uniseg grapheme pysbd pyicu                        # more oracles for the UTF-8 suites
uv run --no-project python -X faulthandler -m pytest -x           # how the CI runs it
uv run --no-project python -m pytest test/doctests.py             # the docstring examples alone
```

The header pytest prints names the module it imported, `StringZilla <version> from <path>`, and ends with the same `- Seed: 42, scale: 1, filter: none` line as the C++ runner.
The Rust tests run as `cargo test`, once with the default features and once with `--no-default-features`, as the [Rust](#rust) section shows.

### Environment

Each setting is an environment variable, and in the C++ runners a `--name` flag of the same name overrides it, as `--scale 0.1` or `--filter='^hash\.'`:

| Variable                      | Flag                | Default | Meaning                                                                                 |
| :---------------------------- | :------------------ | :------ | :-------------------------------------------------------------------------------------- |
| `STRINGZILLA_SEED`            | `--seed`            | `42`    | Any unsigned 64-bit seed, or `random` to draw one, which a run prints                   |
| `STRINGZILLA_SCALE`           | `--scale`           | `1`     | Multiplier for every baseline iteration count, like `0.1` or `10`                       |
| `STRINGZILLA_FILTER`          | `--filter`          | unset   | Regex searched in C++ case labels                                                       |
| `STRINGZILLA_DEVICES`         | `--devices`         | unset   | C++ only: comma-separated `backend:ordinal` devices, like `cuda:0,rocm:1`               |
| `STRINGZILLA_TRAP_ON_FAILURE` | `--trap-on-failure` | `0`     | C++ only: raises `SIGTRAP` at the first failed check, to stop there under a debugger    |
| `STRINGZILLA_NO_SKIPS`        | `--no-skips`        | `0`     | C++ only: exits with 3 when any case skipped, for runners that must cover everything    |
| `STRINGZILLA_IN_QEMU`         | -                   | `0`     | Python only: marks the run as emulated in the pytest header                             |
| `STRINGZILLA_EXPECT_SIMD`     | -                   | `1`     | Python only: `0` skips the check that a SIMD machine's baseline was compiled in         |

A switch is off when unset, empty or `0`, and on otherwise.
Python reads only the variables, beside `pytest -k` and pytest's own options, and Rust reads only `STRINGZILLA_SEED`, filtering through `cargo test <name>`.

`STRINGZILLA_SEED` keys every case's own stream by the case's name, as `stream_key(seed, name, index)`, identical in C++, Python and Rust.
A case's inputs therefore never depend on which cases ran before it, so `STRINGZILLA_SEED` together with `STRINGZILLA_FILTER` is a faithful reproduction recipe.

Each case has its own baseline iteration count tuned for its operation complexity, and `STRINGZILLA_SCALE` is the suite's only tuning knob.
The intended operating points are a short default run for local iteration, `10x` for nightly CI, and `100x` for a stress farm.
Sizing is relative rather than absolute: each family gets a comparable share of the budget, so no single case dominates a run on any machine.

```bash
STRINGZILLA_SEED=random build_debug/stringzilla_test                                     # A fresh seed, which the run prints
STRINGZILLA_FILTER='^test_utf8_' STRINGZILLA_SCALE=0.1 build_debug/stringzilla_test           # Fast inner loop over the UTF-8 cases
STRINGZILLA_SEED=12345 STRINGZILLA_SCALE=10 build_debug/stringzilla_test                 # A nightly stress run
STRINGZILLA_FILTER=utf8 uv run --no-project python -X faulthandler -m pytest -x          # A filter over pytest node ids
```

### Writing a Test

A C++ case is a `void` function, or one returning `case_result_t` when it may skip, that `main` runs as `run_test_case("<label>", function, arguments...)`.
It takes no settings or context, as the run is `test_run`, and it skips only by returning `case_result_t::skipped(reason)`, with a reason like `needs sz_cap_haswell_k` or `no CUDA device`.
A label names the code under test: the function's own name, as `test_hash_unit`, or a call with the runtime choice's exact constants, as `check_find_unit_(find_haswell)`, so `STRINGZILLA_FILTER='^test_utf8_'` runs one family and `STRINGZILLA_FILTER='_neon\)$'` one capability.

It checks with functions from `test/harness.hpp`, each of which counts a failed check at the caller's line, lets the case go on, and returns whether it held:

- `expect_true(held)`, for any condition;
- `expect_equal(actual, wanted)`, which prints both operands when they differ;
- `expect_near(actual, wanted, tolerance)`, holding when `|actual - wanted| <= tolerance * max(1, |actual|, |wanted|)`;
- `expect_throws<std::out_of_range>([&] { return text.substr(100); })`, whose subject is the failure.

`expect_true`, `expect_equal` and `expect_near` take an optional message last, a format string and its arguments, as `expect_equal(found, wanted, "needle {:?} at {}", needle, offset)`.
Where the step after a failed check would index or dereference past what it checked, the case stops on the check's answer, as in `if (!expect_equal(found.size(), wanted.size())) return;`.
A case draws from `split_mix_generator_t generator {case_stream_key()}`, keyed by the run's seed and the case's label, sizes its loops through `scaled_iterations(baseline)`, which scales and floors at 1, and walks an exhaustive space by `sweep_stride(coverage)`, so that a default run samples it evenly and a `10x` run covers it.
A wait polls through `spin_until(predicate)`, which gives up after `wait_limit_k`, so a hang fails the case rather than the CI job.

Every kernel family names its drivers with a fixed set of tiers, and there are no others - fold a new case into whichever tier already owns that kind of assertion:

| Tier          | Asserts against                                                                         | Scales |
| :------------ | :-------------------------------------------------------------------------------------- | :----- |
| `unit`        | hand-written known answers only, never another backend's                                | no     |
| `rules`       | one motif per rule of a published spec, UAX-29 or UAX-14, plus a coverage gate          | no     |
| `equivalence` | a reference against a candidate over generated corpora                                  | yes    |
| `safety`      | survival only - in bounds, stated refusals, no crash on malformed input                 | yes    |
| `all`         | nothing of its own: it drives the tiers above over the family's dispatch points         | yes    |

The load-bearing rule is that `unit` may never derive its expectations by calling another backend: if the reference is wrong, a derived expectation is wrong with it, and the test passes anyway.
Likewise `safety` never asserts content correctness - only that the kernel stayed inside its buffer and returned.

A Python test is `test_<subject>_<behaviour>`, read as a sentence, with a one-line docstring saying what holds.
It draws from the `seed`, `rng` or `np_rng` fixture, never from a literal seed.
The seed is `stream_key(STRINGZILLA_SEED, name, step)`, where `name` is the test function and its parameters other than the `BACKEND_AXES`, here `capabilities`, and `step` is the pytest-repeat step, so every CPU mask sees the same inputs while other parameters and repeats draw apart.
A differential test takes the `capabilities` fixture or `backend_results`, which sweep every CPU mask against the serial one, and a test of a finalizer takes `unraisable`, which collects what the interpreter could not raise.
Axis names follow one vocabulary: `count` for the strings in a batch and `capabilities` for the kernel backend, while `length` stays a string's length; parametrize names are tuples, and marks sit above the parametrizes.
`MAX_CASES_PER_TEST` in `test/base.py` caps how many cases one test function expands to, its backend axes aside, and raising it is a decision for review.
A test that needs an optional dependency says so where it is decided, as `@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")` or `pytest.importorskip("uniseg.wordbreak", reason="uniseg is not installed")`, and a build or device it needs, through a mark of `test/base.py` like `@requires_cuda`.
Anything only a run can find out, like a library that does not load, skips inside a fixture.

A Rust test is named as a behaviour sentence without a `test_` prefix, like `capability_names_match_the_c_table`, and calls the crate's `pub` API.
The pinned region of `rust/testing.rs` gives it `skipped(reason)`, which a test that cannot run here returns, `case_stream_key(index)` for its seeds, `ScratchDirectory` and `block_on`, and the helpers several modules' tests share live below that region.
`test/base.py`, `test/conftest.py` and `rust/testing.rs` each hold a pinned `Shared` region, which the pre-commit hook keeps identical to its canonical copy, so edit around it.

### Benchmarks

Each `bench/harness.hpp` carries the same pinned settings readers as `test/harness.hpp`, so its variables parse, refuse and take `--name` flags the same way, under the `STRINGWARS_` prefix that [StringWars](https://github.com/ashvardanian/StringWars) shares.
Those names, and the `STRINGZILLA_STRESS*` variables of the stress phase, stay as they are; the [Benchmarking](#benchmarking) section lists them.
A benchmark row is named after the kernel it times, like `sz_find_haswell` or `sz_levenshtein_distances_cuda:0:q1024:w16:sorted`, which is what `STRINGWARS_FILTER` matches, and a run prints the same machine header and `- Name: value` settings echo as the tests.

### Devices

`test/cross_cuda.cu`, `test/cross_rocm.hip` and `test/cross_metal.cpp` are one shard per vendor, each compiled for that vendor alone, and every shard links into `stringzilla_test` and `stringzilla_shared_test`.
A vendor the build lacks compiles its shard as plain C++ into a stub, so the runner always links.
The checks themselves live once, in `test/cross_device.hpp`, templated over `device_kernels`, and a shard only binds its vendor's kernels to them.

A vendor switch, `STRINGZILLA_ARCH_CUDA_`, `STRINGZILLA_ARCH_ROCM_` or `STRINGZILLA_ARCH_METAL_`, is a fact about the unit, so each takes its own `#if`, never an `#elif` after another vendor's, nor a negation beside it.
A header that assumes a single vendor says so, as `include/stringzilla/types.cuh` does with `#if STRINGZILLA_ARCH_CUDA_ + STRINGZILLA_ARCH_ROCM_ > 1` and an `#error`.
Anything in a shared header whose meaning depends on the vendor takes internal linkage or the vendor's name, since the shards link into one binary, where two inline definitions of one name would silently merge into one.

`STRINGZILLA_DEVICES=cuda:0,rocm:1` picks the devices to check by `backend:ordinal`, where the ordinal counts within its vendor.
Unset, the runner visits device zero of each compiled vendor that answers, and an explicitly named device that is unavailable ends the run with 1.
A GPU case on a machine without that vendor's device reports `skipped: no CUDA device`, and the Python and Rust suites skip theirs the same way.
The benchmarks pick theirs the same way, through `STRINGWARS_DEVICES`.

### Private Surface Waivers

These test files include a header the public surface does not cover, each for what that surface lacks:

- `test/cross.hpp` includes `stringzilla/utf8_graphemes/serial.h`, `stringzilla/utf8_norm/serial.h` and `stringzilla/utf8_wordbreaks/serial.h` for the per-position rule oracles `sz_utf8_is_grapheme_boundary_serial`, `sz_utf8_is_word_boundary_serial` and `sz_rune_is_word_char`, and the decomposition bound `sz_utf8_norm_decomp_max_k`, which the C ABI does not export.
- `test/parallel.cpp` includes `src/parallel.h`, the tile scheduler the Python and Node extensions share, which no public header declares.

The Python and Rust tests need no waiver.

## CUDA

`STRINGZILLA_CUDA_ARCH` adds `src/capabilities/cuda.cu` to `stringzilla_static` and `stringzilla_shared`: the engines' `cuda` kernels, compiled from each family's `simt.cuh`, and the CUDA device exports: the producers `sz_count_devices_cuda`, `sz_capabilities_detected_cuda` and `sz_stream_init_cuda`, and the twins behind the `_best` dispatch points in `strings.h`, like `sz_allocator_init_unified_cuda` and `sz_strings_to_tape_cuda`.
`STRINGZILLA_CUDA_ARCH=all` compiles each GPU unit once, as PTX at the lowest code its capability needs: `compute_75` for the CUDA baseline, `compute_80` for Ampere's warp reductions and asynchronous copies, `compute_90` for Hopper's clusters, and `compute_100` for Blackwell's cluster launch control.
A named code, like `100f`, or `native` compiles as SASS alone.
`stringzilla_test` includes CUDA checks from `test/cross_cuda.cu` alongside the CPU checks and other enabled vendors:

```sh
cmake -D CMAKE_BUILD_TYPE=Debug -D STRINGZILLA_BUILD_TEST=1 -D STRINGZILLA_CUDA_ARCH=native -B build_debug
cmake --build build_debug --config Debug --target stringzilla_test --parallel
```

```sh
cmake -D CMAKE_BUILD_TYPE=Release -D STRINGZILLA_BUILD_TEST=1 -D STRINGZILLA_CUDA_ARCH=native -B build_release
cmake --build build_release --config Release --target stringzilla_test --parallel
```

```sh
cuda-gdb ./build_debug/stringzilla_test
cuda-memcheck ./build_debug/stringzilla_test
```

## Metal and ROCm

`STRINGZILLA_METAL_ARCH` adds `src/capabilities/metal.c` to the same libraries on Apple platforms: the engines' `metal` kernels, whose `metal.metal` shaders travel as embedded source and compile on the device at first use, and the Metal device exports, the producers and the `_metal` twins behind the `strings.h` dispatch points, like `sz_stream_synchronize_metal`.
`STRINGZILLA_ROCM_ARCH` adds `src/capabilities/rocm.hip` the same way, with the `rocm` kernels and the `sz_rocm_*` device exports, and every vendor asked for sits side by side in one library.
`src/capabilities/cdna3.hip` joins it where the targets include gfx942, `src/capabilities/cdna4.hip` where they include gfx950, and `src/capabilities/cdna5.hip` where they include gfx1250 or gfx1251, each compiled for those targets alone.

The Metal backends of all three engine families build into `stringzilla_test` on Apple platforms, which needs `STRINGZILLA_METAL_ARCH` set and checks them against the serial answers:

```sh
cmake --preset metal
cmake --build --preset metal
ctest --preset metal
```

`STRINGZILLA_ROCM_ARCH` adds the ROCm checks from `test/cross_rocm.hip` to the same `stringzilla_test`; CUDA, ROCm and Metal share `test/cross_device.hpp`.
It builds beside the CUDA test when both are set:

```sh
cmake --preset rocm
cmake --build --preset rocm
ctest --preset rocm
```

## Python

Python bindings are implemented using pure CPython, so you wouldn't need to install SWIG, PyBind11, or any other third-party library.
Still, you need a virtual environment, and it's recommended to use `uv` to create one.

```bash
uv venv --python 3.12                   # or your preferred Python version
source .venv/bin/activate               # to activate the virtual environment
uv pip install . --force-reinstall      # to build locally from source
```

The build goes through `CMakeLists.txt`: scikit-build-core turns `STRINGZILLA_BUILD_PYTHON` on and builds the `stringzilla_python` target, the extension over `stringzilla_static`, in `build_python/<wheel tag>`.
The ISA probes pick the SIMD capabilities exactly as they do for the C library.
CMake options pass through `-C`:

```bash
uv pip install . -C cmake.build-type=Debug                                               # debug asserts on
uv pip install . -C cmake.define.STRINGZILLA_CUDA_ARCH=all                               # CUDA backends too
STRINGZILLA_MARCH=native uv pip install .                                                # tuned for this machine
```

To check the installed version and capabilities, try:

```bash
uv run --no-project python -c "import stringzilla as sz; print(sz.__version__, repr(sz.capabilities_enabled_cpu()))"
```

To clean up code before pushing:

```bash
uv pip install ruff mypy bandit flake8
uv run --no-project ruff check test/*.py --fix
uv run --no-project mypy test/*.py --ignore-missing-imports
uv run --no-project bandit test/*.py -s B101
uv run --no-project flake8 test/*.py --max-line-length=120
```

### Testing

The Python suite, its fixtures and its settings are described with the other suites under [Test Suites](#test-suites).

### Packaging

Source distributions carry every file Git doesn't ignore, minus the `sdist.exclude` list in `pyproject.toml`.

```bash
uv pip install build
uv build --sdist --out-dir dist
```

Before you ship, please make sure the `cibuilwheel` packaging works and tests pass on other platforms.
Don't forget to use the right [CLI arguments][cibuildwheel-cli] to avoid overloading your Docker runtime.

```bash
cibuildwheel
cibuildwheel --platform linux                   # works on any OS and builds all Linux backends
cibuildwheel --platform linux --archs x86_64    # 64-bit x86, the most common on desktop and servers
cibuildwheel --platform linux --archs aarch64   # 64-bit Arm for mobile devices, Apple M-series, and AWS Graviton
cibuildwheel --platform linux --archs i686      # 32-bit Linux
cibuildwheel --platform linux --archs s390x     # emulating big-endian IBM Z
cibuildwheel --platform macos                   # works only on macOS
cibuildwheel --platform windows                 # works only on Windows
```

You may need root privileges for multi-architecture builds:

```bash
sudo $(which cibuildwheel) --platform linux
```

To avoid QEMU issues on SVE, have the emulator hide it, as `QEMU_CPU=max,sve=off` does on AArch64, and set `STRINGZILLA_IN_QEMU=1` so the PyTest header reports the run as emulated:

```bash
CIBW_ENVIRONMENT_LINUX="STRINGZILLA_IN_QEMU=1 QEMU_CPU=max,sve=off" sudo -E $(which cibuildwheel) --platform linux --archs aarch64
```

On Windows and macOS, to avoid frequent path resolution issues, you may want to use:

```bash
python -m cibuildwheel --platform windows
```

All together, for one version of Python, OS, hardware platform:

```bash
CIBW_BUILD=cp312-* CIBW_ARCHS_LINUX=x86_64 cibuildwheel --platform linux
CIBW_BUILD=cp312-* CIBW_ARCHS_MACOS=arm64 python3 -m cibuildwheel --platform macos
$env:CIBW_BUILD = "cp312-*"; $env:CIBW_ARCHS_WINDOWS = "AMD64"; python -m cibuildwheel --platform windows
```

[cibuildwheel-cli]: https://cibuildwheel.readthedocs.io/en/stable/options/#command-line

If you want to run benchmarks against third-party implementations, check out the [`ashvardanian/StringWars`](https://github.com/ashvardanian/StringWars/) repository.

## JavaScript

The addon is the `stringzilla_node` target, which `cmake-js` builds through `CMakeLists.txt` over `stringzilla_static`.
`npm run prebuild` builds it into `build_node/` and stages it under `prebuilds/`, where the loader finds it before any published `@stringzilla/*` package, which `--omit=optional` keeps out:

```bash
npm install --ignore-scripts --omit=optional
npm run prebuild
npm test
```

Log capabilities:

```bash
npm link stringzilla
node --input-type=module -e "import('stringzilla').then(m=>console.log(m.default.Device.cpu().capabilitiesEnabled()))"
```

Check files that would be included in the package:

```bash
npm pack --dry-run
```

## Swift

SwiftPM links the C library prebuilt, as the `StringZillaC` XCFramework on Apple platforms and as an SE-0482 artifact bundle elsewhere, which the `swift` preset builds for the host.
`STRINGZILLA_SWIFT_ARTIFACT` points `Package.swift` at it, relative to the package root; without it, SwiftPM downloads the release's:

```bash
cmake --preset swift
cmake --build --preset swift
STRINGZILLA_SWIFT_ARTIFACT=build_swift/StringZillaC.xcframework swift test # StringZillaC.artifactbundle off Apple platforms
```

Each build adds its slice or variant to the artifact in `STRINGZILLA_SWIFT_DIRECTORY`, so the release fills one XCFramework and one bundle from a build per target, as `.github/workflows/build-swift.yml` does.

Running Swift on Linux requires a couple of extra steps - [`swift.org/install` page](https://www.swift.org/install).
Alternatively, on Linux, the official Swift Docker image can be used for builds and tests:

```bash
sudo docker run --rm -v "$PWD:/workspace" -w /workspace swift:6.4 /bin/bash -cl "swift build -c release --static-swift-stdlib && swift test -c release"
```

To format the code on Linux:

```bash
sudo docker run --rm -v "$PWD:/workspace" -w /workspace swift:6.4 /bin/bash -c "swift format . -i -r --configuration .swift-format"
```

## Rust

StringZilla's Rust crate supports both `std` and `no_std` builds.
Other options include:

- `std`, on by default: enables standard library support.
- `cuda`: the CUDA backend for NVIDIA GPUs, which implies `std`.
- `rocm`: the AMD counterpart, which implies `std`.
- `metal`: the Metal backend for Apple GPUs, which implies `std`.

Each GPU feature lets `Stream::new` make a stream on that vendor's devices, which every engine constructor takes.
`build.rs` builds `stringzilla_static` through CMake, with every capability the toolchain can emit, picked per call by the capability mask, and each GPU feature sets its `STRINGZILLA_*_ARCH` to `all`, or to the code the environment names.
It forwards `STRINGZILLA_MARCH`, `STRINGZILLA_MULTIVERSIONED` and the `STRINGZILLA_TARGET_<CAPABILITY>` overrides from the environment, rebuilding when one changes.
`STRINGZILLA_LIBRARY_DIR=<directory>` links an archive CMake already built there instead:

```bash
cmake --preset release_shared && cmake --build --preset release_shared --target stringzilla_static
STRINGZILLA_LIBRARY_DIR=$PWD/build_release_shared cargo test
```

```bash
cargo test --no-default-features                # verify `no_std` build
cargo test                                      # default tests with `std`
cargo test --features cuda                      # for the Nvidia GPU engines
cargo test --features metal                     # with the Metal backend
```

If you need to isolate a failing test:

```bash
export RUST_BACKTRACE=full
cargo test -- --test-threads=1 --nocapture
```

To polish code before pushing:

```bash
cargo clippy --lib                  # check the library code
cargo clippy --lib -- -D warnings   # to fail on warnings
cargo clean && cargo build --lib    # to force a clean build
```

If you are updating the package contents, you can validate the list of included files using the following command:

```bash
cargo package --list --allow-dirty
```

If you want to run benchmarks against third-party implementations, check out the [`ashvardanian/StringWars`](https://github.com/ashvardanian/StringWars/) repository.

## Go

The binding links the static library, so first build it and copy it next to the Go sources:

```bash
cmake --preset release_shared
cmake --build --preset release_shared --target stringzilla_static
cp build_release_shared/libstringzilla_static.a golang/
```

Then, navigate to the GoLang module root directory and run the tests from there:

```bash
cd golang
go test
```

An archive elsewhere, like a Debug one, links through `CGO_LDFLAGS="-L<its directory>"`.
cGo links no sanitizer runtime, so build that one with `-D STRINGZILLA_BUILD_SANITIZERS=off`.

To benchmark:

```bash
cd golang
go run ../bench/stringzilla.go --input ../leipzig1M.txt
```

Alternatively:

```bash
export GO111MODULE="off"
go test
go run bench/stringzilla.go
```

## Java

The binding loads `libstringzilla_shared` through the Foreign Function & Memory API, and compiles for JDK 25, the oldest long-term release that has it.
`gradle build` stages the shared library from `build_release_shared/`, or from `-PstringzillaLibraryPath=<directory>`, where `NativeLoader` finds it, then runs the tests:

```sh
cmake --preset release_shared && cmake --build --preset release_shared
gradle spotlessCheck build
```

## Commits

`.githooks/commit-msg` checks every message:

- The title reads `<Verb>: <Summary>`, where the verb is one of `Add`, `Break`, `Chore`, `Docs`, `Fix`, `Improve` or `Make`.
- The summary opens with a capital letter, unless it opens with a code identifier, which keeps its own case.
- The title runs to at most 80 characters, and a body line to at most 100 columns, unless it holds a URL or is indented code.
- The message carries no AI-assistant attribution, since authorship is the human who committed.

Git's merge, `fixup!` and `squash!` titles are exempt from the title rules.
A `Release: vX.Y.Z` title is exempt too, but its commit must stamp every version site, and no other commit may change `VERSION`.
`.githooks/pre-commit` formats and judges the staged lines, and `.githooks/selftest` runs both hooks against their fixtures.
On a pull request, `.github/scripts/check-commits.sh` replays `commit-msg` over every commit and runs `pre-commit` over the whole range.

## Releases

`.github/scripts/mint-release.sh X.Y.Z` cuts a release from a clean checkout of the branch it ships.
It builds the Swift binaries for that version, stamps every version site and the binaries' checksums, and commits `Release: vX.Y.Z`.
`.github/scripts/check-release.sh` then checks that commit, even in a clone where `core.hooksPath` is unset, before the script tags it and pushes both.
Then it publishes the GitHub release with the binaries attached, marking `X.Y.Z-rc.N` candidates as prereleases.
Publishing starts every `release-<binding>.yml` at once, and each runs `check-release.sh` before it builds anything, so each registry trusts only its own binding's file.

## General Recommendations

### Operations Not Worth Optimizing

One of the hardest things to learn in HPC is when to stop optimizing, and where not to start.

It doesn't make sense to optimize the kernels behind `sz_order_best`, because almost always, the relative order of two strings depends on the first bytes.
Fetching more bytes is not worth it.
Behind `sz_equal_best`, however, in rare cases, SIMD can help, if the user is comparing two mostly similar strings with identical hashes or checksums.

### Unaligned Loads

One common surface of attack for performance optimizations is minimizing unaligned loads.
Such solutions are beautiful from the algorithmic perspective, but often lead to worse performance.
It's often cheaper to issue two interleaving wide-register loads, than try minimizing those loads at the cost of juggling registers.
Unaligned stores are a different story, especially on x86, where multiple reads can be issued in parallel, but only one write can be issued at a time.

### Register Pressure

Byte-level comparisons are simpler and often faster, than n-gram comparisons with subsequent interleaving.
In the following example we search for 4-byte needles in a haystack, loading at different offsets, and comparing then as arrays of 32-bit integers.

```c
h0_vec.zmm = _mm512_loadu_epi8(h);
h1_vec.zmm = _mm512_loadu_epi8(h + 1);
h2_vec.zmm = _mm512_loadu_epi8(h + 2);
h3_vec.zmm = _mm512_loadu_epi8(h + 3);
matches0 = _mm512_cmpeq_epi32_mask(h0_vec.zmm, n_vec.zmm);
matches1 = _mm512_cmpeq_epi32_mask(h1_vec.zmm, n_vec.zmm);
matches2 = _mm512_cmpeq_epi32_mask(h2_vec.zmm, n_vec.zmm);
matches3 = _mm512_cmpeq_epi32_mask(h3_vec.zmm, n_vec.zmm);
if (matches0 | matches1 | matches2 | matches3)
    return h + sz_u64_ctz(_pdep_u64(matches0, 0x1111111111111111) | //
                          _pdep_u64(matches1, 0x2222222222222222) | //
                          _pdep_u64(matches2, 0x4444444444444444) | //
                          _pdep_u64(matches3, 0x8888888888888888));
```

A simpler solution would be to compare byte-by-byte, but in that case we would need to populate multiple registers, broadcasting different letters of the needle into them.
That may not be noticeable on a micro-benchmark, but it would be noticeable on real-world workloads, where the CPU will speculatively interleave those search operations with something else happening in that context.

### Working on Alternative Hardware Backends

It's important to keep compiler support in mind when extending to new instruction sets.
Check the most recent CI pipeline configuration in `.github/workflows/test-c.yml` to see which compilers are used.
When extending capability detection, avoid compiler intrinsics and OS-specific APIs, as they may not be available on all platforms.
Instead, use inline assembly to read the feature flags, and report them through `sz_capabilities_detected_cpu`.
A new capability's kernels then go into a unit of their own, `src/capabilities/<capability>.c`, and into the lists in `src/dispatch/<family>.c` of every family that has them.

### Working on Faster Edit Distances

When dealing with non-trivial algorithms, like edit distances, it's advisory to provide pseudo-code or a reference implementation in addition to the optimized one.
Ideally, include it in `bench/` as a Python Jupyter Notebook with explanations and visualizations.

### Working on Sequence Processing and Sorting

Sorting algorithms for strings are a deeply studied area.
In general, string sorting algorithms discourage the use of comparisons, as they are expensive for variable-length data and also require pointer-chasing for most array layouts.
They are also harder to accelerate with SIMD, as most layouts imply 16-byte entries, which are often too big to benefit from simple SIMD techniques.
