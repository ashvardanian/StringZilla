# Contributing to StringZilla

Thank you for coming here!
It's always nice to have third-party contributors 🤗
Depending on the type of contribution, you may need to follow different steps.

---

## Directory Tree

```
include/stringzilla/      C and C++ headers — one .h per kernel family declaring its dispatch points, kernels and finder, stringzilla.hpp for C++
include/stringzilla/*/    Kernels, one file per CPU capability — serial, haswell, neon, rvv, etc. — plus the engines' GPU simt.cuh and metal.* sources
c/stringzilla.c           Library exports shared by every family — versions, statuses, capability queries, the kernel finder
c/target/*.c              Library units, one per CPU capability, each defining that capability's kernels once
c/dispatch/               Library units, one per kernel family, with its capability lists, dispatch points and finder
c/dispatch.h              The capability lists' shape and the kernel pick the dispatch units share, internal to the library
c/target/cuda.cu          The engines' cuda kernels and the CUDA device exports, under STRINGZILLA_BUILD_CUDA
c/target/rocm.hip         The engines' rocm kernels and the ROCm device exports, under STRINGZILLA_BUILD_ROCM
c/target/metal.c          The engines' metal kernels and the Metal device exports, under STRINGZILLA_BUILD_METAL
c/parallel.h, .c          Tile-parallel runs on each platform's thread pool, compiled into the Python and Node extensions only
probes/                   ISA probe sources, one per kit mirroring c/target/, each calling one of its kernels as the library compiles it
test/                     C++ and Python tests — see test/README.md
bench/                    C++ benchmarks — see bench/README.md
python/                   CPython extension, one unit per kernel family, with python/stringzilla/stringzilla.h as its private header
rust/                     Rust crate, one module per kernel family under rust/stringzilla/, re-exported by rust/stringzilla.rs
javascript/               Node.js native addon
swift/                    Swift package sources and tests
golang/                   Go cgo bindings
java/                     Java bindings
csharp/                   C# bindings
cmake/                    ISA probe modules, package config templates, and cross-compilation toolchain files
```

For minimal test coverage, check the following scripts:

- `test/main_cpu.cpp` - runs the family suites over the dispatch points and the C++ wrappers, and `test/cross_<arch>.cpp` hold every capability's kernels to the serial ones, built as `stringzilla_cpu_test`, `stringzilla_cpu_shared_test` and `stringzilla_cpu_header_test`.
- `test/main_cuda.cu`, `test/main_rocm.hip` and `test/main_metal.cpp` - each GPU vendor's kernels against the serial answers, through `test/cross_simt.cuh` and `test/cross_metal.cpp`, built as `stringzilla_cuda_test`, `stringzilla_rocm_test` and `stringzilla_metal_test`.
- `test/*.py` - tests the Python API against native strings, split per kernel family - `string_types.py`, `find.py`, `sort.py`, `hash.py`, `cipher.py`, `uncased.py`, `utf8_*.py` - with shared helpers in `base.py` and `utf8_helpers.py`.
- `test/main.js`.

At the C++ level all benchmarks also validate the results against the serial kernels and the STL baselines, serving as tests on real-world data.
They have the broadest coverage of the library, and are the most important to keep up-to-date:

- `bench/main.cpp` - runs every family file, then every `bench/cross_<arch>.cpp`, built as `stringzilla_cpu_bench`, and header-only as `stringzilla_cpu_header_bench`.
- `bench/token.cpp`, `bench/find.cpp`, `bench/sequence.cpp`, `bench/container.cpp`, and the other family files - the dispatch points against the STL and LibC baselines.
- `bench/cross_<arch>.cpp` - every capability's kernels against the serial ones, through the adapters in `bench/cross.hpp`.
- `bench/cross_cuda.cu`, `bench/cross_rocm.hip`, and `bench/cross_metal.cpp` - each GPU vendor's engines against the widest CPU tier the build carries.


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

Every all-caps name starts with the full project name, `STRINGZILLA_`.
A trailing `_` marks a name as internal: it may change in any release, and nothing outside this repository may define or test it.
A name without it is a public contract, either a switch you may set or a value you may read.

| Family                            | Form                          | Example                              |
| :-------------------------------- | :---------------------------- | :----------------------------------- |
| ISA tier, backend, GPU generation | `STRINGZILLA_TARGET_<TIER>`   | `STRINGZILLA_TARGET_HASWELL`         |
| Build mode                        | `STRINGZILLA_HEADER_ONLY`     |                                      |
| Optional feature                  | `STRINGZILLA_WITH_<FEATURE>`  | `STRINGZILLA_WITH_LIBC`              |
| GPU runtime the build links       | `STRINGZILLA_WITH_<RUNTIME>`  | `STRINGZILLA_WITH_METAL`             |
| Permission for a liberty          | `STRINGZILLA_ALLOW_<LIBERTY>` | `STRINGZILLA_ALLOW_MISALIGNED_LOADS` |
| Architecture fact                 | `STRINGZILLA_ARCH_<ARCH>_`    | `STRINGZILLA_ARCH_X8664_`            |
| Operating-system fact             | `STRINGZILLA_OS_<OS>_`        | `STRINGZILLA_OS_LINUX_`              |
| Toolchain fact                    | `STRINGZILLA_HAS_<FEATURE>_`  | `STRINGZILLA_HAS_CLANG_EVEX512_`     |

Architectures are spelled as one token each, `X8664`, `X8632`, `ARM64`, `RISCV64`, `PPC64`, `LOONGARCH64`, `S390X` and `WASM`, and GPU platforms `CUDA` and `ROCM`.
A GPU platform holds beside the host's architecture in both compiler passes, so it never follows a CPU architecture in an `#elif` chain.
Metal has no compiler macro on the host side, so its host API is a switch the build sets where it links Metal and Foundation.
Every name in these families is always defined, as 0 or 1, and tested with `#if`, never with `defined(...)`.

C names put the capability last, `sz_<family>_<what>_<capability>`, like `sz_find_haswell`.
Only a role suffix may follow it: `_t` for a type, `_k` for a constant, `_kernel_` for a GPU entry point, or the trailing `_` of an internal name, like `sz_levenshtein_u64x1_sweep_serial_` and `sz_overlap_scores_simt_kernel_`.
A dispatch point puts `best` in the capability's place, like `sz_copy_best`, and the kernels it picks from are its twins, like `sz_copy_haswell`.

Code that several capabilities share belongs to one of three layers, named in the capability's place:

- `serial`, portable and capability-neutral, in each family's `serial.h`.
- `simt`, the single C source that CUDA and HIP both compile, in each family's `simt.cuh`.
- `metal`, what every Metal tier shares, in each family's `metal.h` and the `metal.metal` shaders it embeds.

A constant identical across the GPU layers is defined once in `serial.h` with the adjective `gpu`, like `sz_levenshtein_gpu_warp_lanes_k`.
One whose value differs per layer takes its layer instead, like `sz_substrings_tally_slot_bits_simt_k` and `sz_substrings_tally_slot_bits_metal_k`.

The GPU capability groups are `cuda`, `rocm` and `metal`, beside the CPU's `cpu`.
Each word names its group's baseline bit, like `sz_cap_cuda_k`, its functions and its library unit, like `c/target/cuda.cu`, and no symbol or source path names a vendor, like `nvidia`, `amd` or `apple`.
A function that touches a device is either a producer or a consumer:

- A __producer__ reports a device's capabilities or opens a stream on it.
  Device queries end with their group. Producers are the only functions that take a device's `ordinal`: `sz_device_count_cuda(&count)`, `sz_capabilities_enabled_cuda(ordinal, &capabilities)` and `sz_stream_init_cuda(ordinal, &stream)`.
- A __consumer__ takes the `capabilities` it picks from and, where it queues work, a trailing `sz_stream_t stream`, but never an ordinal, since the stream names its device: `sz_stream_synchronize_best(capabilities, stream)`.
  It ends in `best` like any dispatch point, and its twins end in their capability, like `sz_stream_synchronize_cuda(stream)`.

A null stream is the default stream of the default device: the calling thread's current device on CUDA and ROCm, and the system default device on Metal.
On the CPU the stream must be null.
Each of these words has one meaning across the library:

| Word                                   | Meaning                                                                          | Appears as                                                                      |
| :------------------------------------- | :------------------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| `cpu`, `cuda`, `rocm`, `metal`         | A capability group and its baseline bit                                          | Device-query, twin and kernel suffix, `c/target/<group>.*`                           |
| `simt`                                 | The single C source CUDA and HIP both compile                                    | `<family>/simt.cuh`, the suffixes `_simt_`, `_simt_t`, `_simt_k`                |
| `metal`, as a layer                    | What every Metal tier shares                                                     | `<family>/metal.h`, `<family>/metal.metal`, the suffix `_metal_`                |
| `gpu`                                  | Adjective for every GPU group                                                    | `sz_cap_gpus_k`, `sz_missing_gpu_k`, `sz_levenshtein_gpu_warp_lanes_k`          |
| `device`                               | A processor kernels run on, which a stream belongs to                            | `sz_device_count_cuda`, `sz_device_memory_mismatch_k`, `sz_device_enter_simt_` |
| `ordinal`                              | A device's index within its group, as its runtime numbers it                     | Producer parameters only                                                        |
| `stream`                               | A `cudaStream_t`, `hipStream_t` or `id<MTLCommandQueue>`, which names its device | The trailing `void *stream` of every consumer                                   |
| `queue`                                | A tile work queue, never a stream                                                | `sz_tile_queue_t`                                                               |
| `unified`                              | Memory both the host and the stream's device address                            | `sz_allocator_init_unified_best`                                                |
| `tape`                                 | One block holding a sequence's offsets and bytes                                | What `sz_sequence_realloc_best` writes                                          |
| A capability, like `haswell` or `cuda` | One bit of a mask                                                                | The last token before the role suffix                                           |
| `kernel`                               | A GPU entry point                                                                | `_kernel_`, right after the capability                                          |

For C++ code:

- Explicitly use `std::` or `sz::` namespaces over global `memcpy`, `uint64_t`, etc.
- Explicitly mark `noexcept` or `noexcept(false)` for all library interfaces, except for `__global__` CUDA functions.
- Document all possible exceptions of an interface using `@throw` in Doxygen.
- Avoid C-style variadic arguments in favor of templates.
- Avoid C-style casts in favor of `static_cast`, `reinterpret_cast`, and `const_cast`, except for places where a C function is called.
- Use lower-case names for everything, except settings/conditions macros.
  Function-like macros, that take arguments, should be lowercase as well.
- In templates prefer `typename` over `class`.

For Python code:

- Use lower-case names for functions and variables.

## C++ and C

The primary C implementation and the C++ wrapper are built with CMake.
Assuming the extensive use of new SIMD intrinsics and recent C++ language features, using a recent compiler is recommended.
We prefer GCC 12 or newer, which is available from default Ubuntu repositories with Ubuntu 22.04 LTS onwards.
If this is your first experience with CMake, use the following commands to get started on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install cmake              # Consider pulling a newer version from PyPI
sudo apt-get install g++-12 gcc-12      # You may already have a newer version on Ubuntu 24
sudo apt install libstdc++6-12-dbg      # STL debugging symbols for GCC 12
```

On Linux, after that, if you want to compile the minimal set of tests, use the presets CI uses:

```bash
cmake --preset release
cmake --build --preset release --target stringzilla_cpu_test
ctest --preset release
```

`cmake --list-presets` shows the rest - `debug`, `cuda`, `rocm`, `metal`, `swift`, the `linux_<arch>` cross builds, and WASI.
Machine-specific settings, like a CUDA host compiler, belong in an untracked `CMakeUserPresets.json`.
Without a preset, a configure builds the libraries alone, as every binding's build does: `-D STRINGZILLA_BUILD_TEST=1` and `-D STRINGZILLA_BUILD_BENCH=1` add the suites, and `-D STRINGZILLA_BUILD_CUDA=1`, `_ROCM` or `_METAL` the GPUs.

Every binding builds the libraries through `CMakeLists.txt`, so CMake is the one place that probes which kits the toolchain builds.
Each kit's probe, `probes/<kit>.c`, calls one of its kernels, compiled header-only at the baseline flags as the library compiles it.
Every tier header scopes its kernels to their kit with `#pragma clang attribute` or `#pragma GCC target`, on every platform.
LASX and POWER9 also need `-mlasx` and `-mcpu=power9` file-wide, as `lasxintrin.h` and `altivec.h` hide their contents without them, so that flag reaches the kit's probe, its `c/target/` unit and, in the header-only suites, the `cross_<arch>.cpp` checks of its architecture, and nothing else.
The libraries compile every kit the toolchain builds and dispatch by runtime detection, so one artifact runs on any CPU of its architecture.
`-D STRINGZILLA_TARGET_<KIT>=0` drops a kit, and `=1` keeps one only where its probe compiles.

The baseline is each architecture's floor: `-march=x86-64`, `-march=armv8-a`, `-march=rv64gc`, `-mcpu=power8`, and `-march=loongarch64`.
WebAssembly is the exception, as an engine validates a module whole, so its one kit, `STRINGZILLA_TARGET_ARCH`, reaches every unit.
The RVV kernels need Clang 21 or newer; an older compiler probes those kits as 0.
The `linux_arm64`, `linux_riscv64`, `linux_ppc64le` and `linux_loongarch64` presets cross-compile with Clang, through `cmake/toolchain-<arch>-llvm.cmake`, and run the tests under QEMU.
Each toolchain emulates the richest CPU by default, so the tests reach every kit, and its `<ARCH>_QEMU_CPU` variable, like `-D PPC_QEMU_CPU=power8`, runs them on the floor instead.

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
cmake -B build_release -G "Visual Studio 17 2022" -A x64 -D STRINGZILLA_BUILD_TEST=1 -D CMAKE_BUILD_TYPE=Release
cmake --build build_release --config Release --parallel
build_release\\Release\\stringzilla_cpu_test.exe
```

For MinGW (MSYS2):

```bash
pacman -S --needed --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake make
cmake -G "MinGW Makefiles" -B build_release -D STRINGZILLA_BUILD_TEST=1 -D CMAKE_BUILD_TYPE=Release
cmake --build build_release --config Release --parallel
./build_release/stringzilla_cpu_test.exe
```

### Testing

Using modern syntax, this is how you build and run the test suite:

```bash
cmake -D STRINGZILLA_BUILD_TEST=1 -D STRINGZILLA_USE_SANITIZERS=0 -D CMAKE_BUILD_TYPE=Debug -B build_debug
cmake --build build_debug --config Debug --parallel   # Which will produce the following targets:
build_debug/stringzilla_cpu_test            # Dispatch points, C++ wrappers, and every kernel of the static library
build_debug/stringzilla_cpu_shared_test     # Dispatch points alone, through the shared library
build_debug/stringzilla_cpu_header_test     # Kernels compiled header-only, and the dispatch-point stubs
```

Kernel cross-checks are named `test_<family>_<tier>_<capability>`, so `STRINGZILLA_FILTER='_neon$'` runs one capability.

There is no separate SIMD-disabled target.
To get a build with SIMD dispatch narrowed, pass `-D STRINGZILLA_TARGET_<KIT>=0` at configure time for each kit to leave out, for example `-D STRINGZILLA_TARGET_SKYLAKE=0 -D STRINGZILLA_TARGET_ICELAKE=0` for x86 without AVX-512, then rebuild `stringzilla_cpu_test` against that configuration.
`-D STRINGZILLA_TARGET_ARCH=<name>`, like `native`, tunes the libraries for one CPU in place of their portable floor, without narrowing the kits.

Note, that Address Sanitizers have a hard time with masked load and store instructions in AVX-512 and SVE.

#### Test Tiers

Every kernel family registers its tests with a fixed set of suffixes, and there are no other tiers.
Do not invent a fifth one - fold the new case into whichever tier already owns that kind of assertion.

| Tier      | Asserts against                                                                          | Uses the multiplier |
| :-------- | :--------------------------------------------------------------------------------------- | :------------------ |
| `_unit`   | hand-written literal expectations only                                                   | no                  |
| `_rules`  | one motif per spec rule id, plus a coverage gate asserting every required rule was hit   | no                  |
| `_safety` | survival only - in bounds, `bytes_consumed` ≤ `length`, no crash on malformed input      | yes                 |
| `_all`    | the serial-vs-ISA differential, plus invariants that hold independently of any reference | yes                 |

The load-bearing rule is that `_unit` may never derive its expectations by calling another backend.
If the reference is wrong, a derived expectation is wrong with it, and the test passes anyway.
Likewise `_safety` never asserts content correctness - only that the kernel stayed inside its buffer and returned.

The `_rules` tier exists only for families with a Unicode rule table - wordbreaks, graphemes, sentences, and linebreaks.
Families without one - runes, tokens, delimiters, and norm - register three tiers rather than four.
Registration order is `_unit`, then `_rules` where it applies, then `_safety`, then `_all`.

The C++ and Python test suites support environment variables for reproducible stress testing and CI fuzzing:

| Variable             | Default | Meaning                                                                      |
| :------------------- | :------ | :--------------------------------------------------------------------------- |
| `STRINGZILLA_SEED`   | `42`    | Seed for every random generator, or `random` to draw one, which a run prints |
| `STRINGZILLA_SCALE`  | `1`     | Multiplier for every baseline iteration count, like `0.1` or `10`            |
| `STRINGZILLA_FILTER` | unset   | ECMAScript regex over C++ test names or pytest node ids                      |

A value that does not parse stops the run with a message naming the variable, in both suites.
A `STRINGZILLA_FILTER` that does not compile as a regex matches as a plain substring.
Python matches `STRINGZILLA_FILTER` against pytest node ids, alongside `pytest -k`.

Each test has its own baseline iteration count tuned for its operation complexity.
`STRINGZILLA_SCALE` is the suite's only tuning knob, and two members of the `test_context_t` a test takes put work under it.
`context.iterations(baseline)` scales a loop count and floors at 1, so a baseline small enough for the floor to swallow the multiplier is effectively a constant - don't use tiny baselines on expensive loops.
`context.sweep_stride(complete)` gives the step for walking an exhaustive space, such as every byte value, every byte pair, or every window phase.
A sweep becomes complete at the `10x` stress point rather than at the default, so a default run samples every space evenly and a nightly or farm run covers them exhaustively.

The intended operating points are a short default run for local iteration, `10x` for nightly CI, and `100x` for a stress farm.
Sizing is relative rather than absolute: each family gets a comparable share of the budget, so no single test dominates a run on any machine.
Each top-level test is also wall-clock timed and reported as `- name ... ok (N.NN s)`, so slow tests are obvious.

```bash
# Draw a fresh seed instead of the default 42; the run prints it
STRINGZILLA_SEED=random build_debug/stringzilla_cpu_test

# Quick smoke test (10% of normal iterations)
STRINGZILLA_SCALE=0.1 build_debug/stringzilla_cpu_test

# Fast inner loop: only the UTF-8 tests, at 10% iterations, reproducibly
STRINGZILLA_FILTER=utf8 STRINGZILLA_SCALE=0.1 STRINGZILLA_SEED=42 build_debug/stringzilla_cpu_test

# Thorough CI stress test (10x normal iterations)
STRINGZILLA_SCALE=10 build_debug/stringzilla_cpu_test

# Combine both for CI fuzzing
STRINGZILLA_SEED=12345 STRINGZILLA_SCALE=5 build_debug/stringzilla_cpu_test

# Python tests also respect STRINGZILLA_SEED, STRINGZILLA_SCALE and STRINGZILLA_FILTER
STRINGZILLA_SEED=random pytest test/
STRINGZILLA_FILTER=utf8 pytest test/
```

When a C++ test fails a `verify` or throws, the harness prints a `rerun:` line to stderr under the failure, naming the seed, a filter that selects only that test, and the binary, then moves on to the next test:

```text
- test_utf8_runes_all ... FAILED: verification failed
  rerun: STRINGZILLA_SEED=42 STRINGZILLA_FILTER='^test_utf8_runes_all$' build_debug/stringzilla_cpu_test
```

The run still exits with 1 at the end.
A library assertion or a crash stops the run instead, printing a backtrace where the C library can, and the last `- <name> ...` line names the test to put into the template printed at startup:

```text
- Seed: 42
- Rerun one test: STRINGZILLA_SEED=42 STRINGZILLA_FILTER='^<name>$' build_debug/stringzilla_cpu_test
```

This is particularly useful for debugging SIMD edge cases that only manifest with specific input patterns.

`STRINGZILLA_SEED` selects the seed, and `run_test` hands each test its own generator, seeded from that seed and the test name.
A test's inputs therefore never depend on which tests ran before it, so `STRINGZILLA_SEED` together with `STRINGZILLA_FILTER` is a faithful reproduction recipe.
Assertions use `verify`, `let_verify`, `scope_verify`, and `throws_verify`, deliberately independent of `NDEBUG`, because a plain `assert` compiles out in Release builds and silently disables the checks.

The scalable-vector backends must stay correct at every hardware vector length, and the CI `test_cross_qemu` matrix sweeps them all.
The same sweep runs locally with user-mode QEMU - on an Arm host that covers NEON-only dispatch plus SVE at 128/256/512 bits, and cross-compilers unlock the RISC-V, x86, LoongArch, and POWER backends too:

```bash
sudo apt install qemu-user gcc-x86-64-linux-gnu gcc-riscv64-linux-gnu gcc-loongarch64-linux-gnu gcc-powerpc64le-linux-gnu

# Sweep SVE vector lengths on the native Arm binary (sve-max-vq is VL/128)
for vq in 1 2 4; do
  STRINGZILLA_FILTER=utf8 STRINGZILLA_SCALE=0.1 qemu-aarch64 -cpu max,sve-max-vq=$vq build_release/stringzilla_cpu_test
done

# NEON-only dispatch (otherwise SVE2 always wins and NEON is never exercised)
qemu-aarch64 -cpu max,sve=off build_release/stringzilla_cpu_test

# Cross-compile a single-TU probe against another backend and run it emulated
x86_64-linux-gnu-gcc -O2 -mavx2 -mbmi -mbmi2 -mpopcnt -DSTRINGZILLA_TARGET_HASWELL=1 -Iinclude probe.c -o probe -static
qemu-x86_64 -cpu max ./probe
```

Note that QEMU's TCG cannot execute AVX-512, so the Ice Lake and Skylake backends are compile-checked only under emulation and need real x86 hardware to run.

To use CppCheck for static analysis make sure to export the compilation commands.
Overall, CppCheck and Clang-Tidy are extremely noisy and not suitable for CI, but may be useful for local development.

```bash
sudo apt install cppcheck clang-tidy-11

cmake -B build_artifacts \
  -D CMAKE_BUILD_TYPE=RelWithDebInfo \
  -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -D STRINGZILLA_BUILD_BENCH=1 \
  -D STRINGZILLA_BUILD_TEST=1

cppcheck --project=build_artifacts/compile_commands.json --enable=all

clang-tidy-11 -p build_artifacts
```

I'd recommend putting the following breakpoints:

- `__asan::ReportGenericError` - to detect illegal memory accesses.
- `__GI_exit` - to stop at exit points - the end of running any executable.
- `__builtin_unreachable` - to catch unexpected code paths.
- `sz_assert_failure_` - to catch StringZilla logic assertions.

### Benchmarking

For benchmarks, you can use the following commands:

```bash
cmake -D STRINGZILLA_BUILD_BENCH=1 -B build_release
cmake --build build_release --config Release --parallel # Produces the following targets:
build_release/stringzilla_cpu_bench                     # - every family's dispatch points, then every capability's kernels
build_release/stringzilla_cpu_header_bench              # - the kernels header-only, with the helpers the library keeps private
```

Each family times its dispatch points, like `sz_find_best`, against the standard baselines, like `find<std::strstr>`.
Each capability's kernels, like `sz_find_haswell`, are timed against the serial ones, like `sz_find_serial`, and skipped on a CPU that lacks the capability.
The header-only twin has no dispatch points to time, but it reaches the private helpers, like the SIMD sorts of pgrams and the Levenshtein and overlap steps.

Each GPU vendor has one launcher, built only when its `STRINGZILLA_BUILD_CUDA`, `STRINGZILLA_BUILD_ROCM`, or `STRINGZILLA_BUILD_METAL` option is on:

```sh
build_release/stringzilla_cuda_bench     # - for edit distances, window overlap, and multi-pattern search on Nvidia GPUs
build_release/stringzilla_rocm_bench     # - for the same on AMD GPUs
build_release/stringzilla_metal_bench    # - for window overlap on Apple GPUs
```

Their rows are named after the kernel they time, like `sz_levenshtein_distances_cuda:q1024:w16:sorted` or `sz_overlap_scores_metal:w5`, which is what `STRINGWARS_FILTER` matches.

All of them support customization via environment variables.
Let's say you want to benchmark large-batch DNA edit distances:

```sh
cmake -D STRINGZILLA_BUILD_BENCH=1 -B build_release
cmake --build build_release --config Release --target stringzilla_cpu_bench --parallel  # CPU
cmake --build build_release --config Release --target stringzilla_cuda_bench --parallel # GPU
STRINGWARS_FILTER="sz_levenshtein_distances" STRINGWARS_DATASET="acgt_1k.txt" build_release/stringzilla_cpu_bench
STRINGWARS_FILTER="sz_levenshtein_distances" STRINGWARS_DATASET="acgt_100k.txt" build_release/stringzilla_cuda_bench

STRINGWARS_FILTER="sz_levenshtein_distances_cuda:q[0-9]+:w" STRINGWARS_DATASET="acgt_1k.txt" build_release/stringzilla_cuda_bench
STRINGZILLA_STRESS=0 STRINGWARS_FILTER="sz_levenshtein_distances_cuda:q1024:w16" STRINGWARS_DATASET="acgt_100k.txt" build_release/stringzilla_cuda_bench
```

The benchmark harness reads these environment variables:

| Variable                        | Default     | Meaning                                                                              |
| :------------------------------ | :---------- | :----------------------------------------------------------------------------------- |
| `STRINGWARS_DATASET`            | per corpus  | Corpus file, instead of `leipzig1M.txt` or `xlsum.csv`                               |
| `STRINGWARS_TOKENS`             | per corpus  | `file`, `lines`, `words`, or an N-gram length like `64`                              |
| `STRINGWARS_BYTES`              | whole file  | Bytes read from the corpus, like `4096`, `64KB` or `1GB`; UTF-8 families read `64MB` |
| `STRINGWARS_UNIQUE`             | `0`         | Sorts the tokens and drops duplicates first                                          |
| `STRINGWARS_FILTER`             | unset       | ECMAScript regex over benchmark names, or a substring when it does not compile       |
| `STRINGWARS_SEED`               | `42`        | Seed for every random generator, or `random` to draw one, which a run prints         |
| `STRINGWARS_WARMUP`             | `1s`        | Untimed run before each benchmark, like `200ms` or `1s`                              |
| `STRINGWARS_TIME_LIMIT`         | `10s`       | Timed run of each benchmark, like `200ms` or `10s`; `1s` in debug builds             |
| `STRINGWARS_BATCH_PER_CORE`     | per backend | Candidates per call per CPU core, CUDA or ROCm multiprocessor, or Metal GPU core     |
| `STRINGZILLA_STRESS`            | `1`         | Checks each backend against its baseline before timing                               |
| `STRINGZILLA_STRESS_TIME_LIMIT` | time limit  | Time of each stress check, like `200ms` or `10s`                                     |
| `STRINGZILLA_STRESS_DIR`        | `.tmp`      | Directory for stress-check failure logs                                              |
| `STRINGZILLA_STRESS_LIMIT`      | `2`         | The stress-check failure that stops the run, so `1` stops at the first               |

For a fast inner loop, scope to one backend on a small dataset, cap the dataset size, skip the stress phase, and use short runs:

```bash
STRINGWARS_FILTER='sz_find' STRINGWARS_DATASET=leipzig1M.txt \
    STRINGWARS_BYTES=64KB STRINGWARS_BATCH_PER_CORE=1024 \
    STRINGZILLA_STRESS=0 STRINGWARS_TIME_LIMIT=1s \
    build_release/stringzilla_cpu_bench
```

Throughput is a time-bounded measurement: absolute GB/s drifts ±10-15% on a loaded machine, while the ratio between two backends in the _same_ run stays stable.
Compare A/B within one run; raise `STRINGWARS_TIME_LIMIT` and use a quiet machine when you need stable absolute numbers.
The work itself is deterministic: the tokens keep their file order, and a given `STRINGWARS_SEED` always draws the same samples from them.

Each family's benchmarks live in an identically named file in the `bench/` directory, and their kernel adapters in `bench/cross.hpp`.
All of them feature file-level documentation, and are designed to be self-explanatory.
You can easily log their descriptions until the first `*/` with the following `sed` and `awk` commands:

```sh
sed '/\*\//q' bench/memory.cpp
awk '/\*\// { exit } { print }' bench/memory.cpp
```

### Benchmarking Hardware-Specific Optimizations

Running on modern hardware, you may want to compile the code for older generations to compare the relative performance.
The assumption would be that newer ISA extensions would provide better performance.
On x86_64, you can use the following commands to compile for Sandy Bridge, Haswell, and Sapphire Rapids:

```bash
cmake -D CMAKE_BUILD_TYPE=Release -D STRINGZILLA_BUILD_BENCH=1 \
    -D STRINGZILLA_TARGET_ARCH="ivybridge" -B build_release/ivybridge && \
    cmake --build build_release/ivybridge --config Release --parallel
cmake -D CMAKE_BUILD_TYPE=Release -D STRINGZILLA_BUILD_BENCH=1 \
    -D STRINGZILLA_TARGET_ARCH="haswell" -B build_release/haswell && \
    cmake --build build_release/haswell --config Release --parallel
cmake -D CMAKE_BUILD_TYPE=Release -D STRINGZILLA_BUILD_BENCH=1 \
    -D STRINGZILLA_TARGET_ARCH="sapphirerapids" -B build_release/sapphirerapids && \
    cmake --build build_release/sapphirerapids --config Release --parallel
```

### Benchmarking Compiler-Specific Optimizations

Alternatively, you may want to compare the performance of the code compiled with different compilers.
On x86_64, you may want to compare GCC, Clang, and ICX.

```bash
cmake -D CMAKE_BUILD_TYPE=Release -D STRINGZILLA_BUILD_BENCH=1 -D STRINGZILLA_BUILD_SHARED=1 \
    -D CMAKE_CXX_COMPILER=g++-12 -D CMAKE_C_COMPILER=gcc-12 \
    -B build_release/gcc && cmake --build build_release/gcc --config Release --parallel
cmake -D CMAKE_BUILD_TYPE=Release -D STRINGZILLA_BUILD_BENCH=1 -D STRINGZILLA_BUILD_SHARED=1 \
    -D CMAKE_CXX_COMPILER=clang++-14 -D CMAKE_C_COMPILER=clang-14 \
    -B build_release/clang && cmake --build build_release/clang --config Release --parallel
```

### Profiling

To simplify tracing and profiling, build with symbols using the `RelWithDebInfo` configuration.
Here is an example for profiling the hashing benchmarks of `stringzilla_cpu_bench`.

```bash
cmake -D STRINGZILLA_BUILD_BENCH=1 \
    -D STRINGZILLA_BUILD_TEST=1 \
    -D STRINGZILLA_BUILD_SHARED=1 \
    -D CMAKE_BUILD_TYPE=RelWithDebInfo \
    -B build_profile
cmake --build build_profile --config Release --target stringzilla_cpu_bench --parallel

# Check that the debugging symbols are there with your favorite tool
readelf --sections build_profile/stringzilla_cpu_bench | grep debug
objdump -h build_profile/stringzilla_cpu_bench | grep debug

# Profile
sudo env STRINGWARS_DATASET=./leipzig1M.txt STRINGWARS_FILTER='sz_hash' perf record -g build_profile/stringzilla_cpu_bench
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
build_debug/stringzilla_cpu_test
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
build_debug/stringzilla_cpu_test
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
    -D CMAKE_CXX_COMPILER=g++ -D CMAKE_C_COMPILER=gcc -D STRINGZILLA_TARGET_ARCH="ivybridge" \
    -B build_debug
cmake3 --build build_debug --config Debug --target stringzilla_cpu_test
build_debug/stringzilla_cpu_test
```

The CentOS-based __Amazon Linux 2__ is still used in older AWS Lambda functions.
Sadly, the newest GCC version it supports is 10, and it can't handle AVX-512 instructions.

```bash
sudo docker run -it --rm -v "$(pwd)":/workspace/StringZilla amazonlinux:2 bash
cd /workspace/StringZilla
yum install -y make cmake3 gcc10 gcc10-c++
cmake3 -D STRINGZILLA_BUILD_TEST=1 -D CMAKE_BUILD_TYPE=Debug \
    -D CMAKE_CXX_COMPILER=g++ -D CMAKE_C_COMPILER=gcc -D STRINGZILLA_TARGET_ARCH="ivybridge" \
    -B build_debug
cmake3 --build build_debug --config Debug --target stringzilla_cpu_test
build_debug/stringzilla_cpu_test
```

> [!CAUTION]
> 
> Even with GCC 10 the tests compilation will fail, as the STL implementation of the `insert` function doesn't conform to standard.
> The `s.insert(s.begin() + 1, {'a', 'b', 'c'}) == (s.begin() + 1)` expression is illformed, as the `std::string::insert` return `void`.

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
cmake --build build_wasm --target stringzilla_cpu_test
ctest --test-dir build_wasm # runs each .wasm under Wasmtime
```

A module carries one SIMD kit, `STRINGZILLA_TARGET_ARCH`, so the relaxed-SIMD `v128relaxed` build above is a separate artifact from the strict `v128` one and the SIMD-free `serial` one.
Pass `-DSTRINGZILLA_TARGET_ARCH=v128` or `=serial`, or set the environment variable, for the others; CMake turns the kit into `-msimd128` and `-mrelaxed-simd` for every unit, so no other opcode reaches the binary.
Shared libraries stay off in both configurations, since WASI has no dynamic loader.

## CUDA

`STRINGZILLA_BUILD_CUDA` adds `c/target/cuda.cu` to `stringzilla_static` and `stringzilla_shared`: the engines' `cuda` kernels, compiled from each family's `simt.cuh`, and the CUDA device exports: the producers `sz_device_count_cuda`, `sz_capabilities_detected_cuda` and `sz_stream_init_cuda`, and the twins behind the `_best` dispatch points in `memory.h`, like `sz_allocator_init_unified_cuda` and `sz_sequence_realloc_cuda`.
`stringzilla_cuda_test`, built from `test/main_cuda.cu` and `test/cross_cuda.cu` over the static library, checks the CUDA kernels and the dispatch points over them against the serial answers:

```sh
cmake -D CMAKE_BUILD_TYPE=Debug -D STRINGZILLA_BUILD_TEST=1 -D STRINGZILLA_BUILD_CUDA=1 -B build_debug
cmake --build build_debug --config Debug --target stringzilla_cuda_test --parallel
```

```sh
cmake -D CMAKE_BUILD_TYPE=Release -D STRINGZILLA_BUILD_TEST=1 -D STRINGZILLA_BUILD_CUDA=1 -B build_release
cmake --build build_release --config Release --target stringzilla_cuda_test --parallel
```

```sh
cuda-gdb ./build_debug/stringzilla_cuda_test
cuda-memcheck ./build_debug/stringzilla_cuda_test
```

## Metal and ROCm

`STRINGZILLA_BUILD_METAL` adds `c/target/metal.c` to the same libraries on Apple platforms: the engines' `metal` kernels, whose `metal.metal` shaders travel as embedded source and compile on the device at first use, and the Metal device exports, the producers and the `_metal` twins behind the `memory.h` dispatch points, like `sz_stream_synchronize_metal`.
`STRINGZILLA_BUILD_ROCM` adds `c/target/rocm.hip` the same way, with the `rocm` kernels and the `sz_rocm_*` device exports, and every vendor asked for sits side by side in one library.

The Metal backends of all three engine families build into `stringzilla_metal_test` on Apple platforms, which needs `STRINGZILLA_BUILD_METAL` on and checks them against the serial answers:

```sh
cmake --preset metal
cmake --build --preset metal
ctest --preset metal
```

`stringzilla_rocm_test`, built under `STRINGZILLA_BUILD_ROCM` on Linux from `test/main_rocm.hip` and `test/cross_rocm.hip`, runs the checks `test/cross_simt.cuh` shares with the CUDA test, and exits zero without a device.
It builds beside the CUDA test when both options are on:

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
The ISA probes pick the SIMD kits exactly as they do for the C library.
CMake options pass through `-C`:

```bash
uv pip install . -C cmake.build-type=Debug -C cmake.define.STRINGZILLA_USE_SANITIZERS=OFF # debug asserts on
uv pip install . -C cmake.define.STRINGZILLA_BUILD_CUDA=ON                               # CUDA backends too
STRINGZILLA_TARGET_ARCH=native uv pip install .                                           # tuned for this machine
```

To check the installed version and capabilities, try:

```bash
uv run --no-project python -c "import stringzilla as sz; print(sz.__version__, repr(sz.cpu_capabilities_enabled()))"
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

For testing we use PyTest, which may not be installed on your system.

```bash
uv pip install --group test --group test-oracles                # the suite, NumPy and the oracles
uv run --no-project python -X faulthandler -m pytest -x          # how the CI runs it
uv run --no-project python -m pytest test/doctests.py            # to run the docstring examples
uv run --no-project python -c 'from stringzilla import hash as sz_hash; print(sz_hash("abc", 100))'
```

StringZilla for Python seems to cover more OS and hardware combinations, than NumPy.
That's why NumPy isn't a required dependency.
Still, many tests may use NumPy, so consider installing it on mainstream platforms.
Several suites also cross-check against reference implementations, and skip themselves when those are absent:

```bash
uv pip install pycryptodome uniseg grapheme pysbd pyicu # oracles for the cipher and UTF-8 suites
```

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

Each build adds its slice or variant to the artifact in `STRINGZILLA_SWIFT_DIRECTORY`, so the release fills one XCFramework and one bundle from a build per target, as `.github/workflows/_swift.yml` does.

Running Swift on Linux requires a couple of extra steps - [`swift.org/install` page](https://www.swift.org/install).
Alternatively, on Linux, the official Swift Docker image can be used for builds and tests:

```bash
sudo docker run --rm -v "$PWD:/workspace" -w /workspace swift:6.0 /bin/bash -cl "swift build -c release --static-swift-stdlib && swift test -c release"
```

To format the code on Linux:

```bash
sudo docker run --rm -v "$PWD:/workspace" -w /workspace swift:6.0 /bin/bash -c "swift format . -i -r --configuration .swift-format"
```

## Rust

StringZilla's Rust crate supports both `std` and `no_std` builds.
Other options include:

- `std`, on by default: enables standard library support.
- `cuda`: the CUDA backend for NVIDIA GPUs, which implies `std`.
- `rocm`: the AMD counterpart, which implies `std`.
- `metal`: the Metal backend for Apple GPUs, which implies `std`.

Each GPU feature lets `Stream::new` make a stream on that vendor's devices, which every engine constructor takes.
`build.rs` builds `stringzilla_static` through CMake, with every capability the toolchain can emit, picked per call by the capability mask, and each GPU feature switches on its `STRINGZILLA_BUILD_*` option.
It forwards `STRINGZILLA_TARGET_ARCH`, the `STRINGZILLA_TARGET_<KIT>` overrides and the GPU architecture lists from the environment, rebuilding when one changes.
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

## GoLang

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
cGo links no sanitizer runtime, so build that one with `-D STRINGZILLA_USE_SANITIZERS=OFF`.

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
Check the most recent CI pipeline configurations in `prerelease.yml` and `release.yml` to see which compilers are used.
When extending capability detection, avoid compiler intrinsics and OS-specific APIs, as they may not be available on all platforms.
Instead, use inline assembly to read the feature flags, and report them through `sz_capabilities_detected_cpu`.
A new capability's kernels then go into a unit of their own, `c/target/<capability>.c`, and into the lists in `c/dispatch/<family>.c` of every family that has them.

### Working on Faster Edit Distances

When dealing with non-trivial algorithms, like edit distances, it's advisory to provide pseudo-code or a reference implementation in addition to the optimized one.
Ideally, include it in `bench/` as a Python Jupyter Notebook with explanations and visualizations.

### Working on Sequence Processing and Sorting

Sorting algorithms for strings are a deeply studied area.
In general, string sorting algorithms discourage the use of comparisons, as they are expensive for variable-length data and also require pointer-chasing for most array layouts.
They are also harder to accelerate with SIMD, as most layouts imply 16-byte entries, which are often too big to benefit from simple SIMD techniques.
