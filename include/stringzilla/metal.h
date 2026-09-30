/**
 *  @file include/stringzilla/metal.h
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Core types on an Apple GPU: the device, the arena both sides address, and the queue every
 *      Metal family encodes into - the twin of `types.cuh`.
 *
 *  A shared Metal buffer's host address is never its GPU address, and StringZilla hands kernels
 *  pointers inside pointers - an engine's forest, a sequence's views, every view's text - which no
 *  per-argument binding can reach. So an @ref sz_metal_device_t owns one arena: a @c MAP_NORESERVE
 *  reservation wrapped by one shared buffer, from which its allocator carves every block. A kernel
 *  gets the arena and its host base, and turns any host pointer inside it into its own with one
 *  subtraction; @ref sz_memory_reaches_metal is a range check against the same bounds.
 *
 *  Written in C, as every GPU header in this library is. Metal's API is Objective-C, reached
 *  through @c objc_msgSend cast to each call's exact prototype, so no translation unit here needs
 *  an Objective-C compiler.
 *
 *  @sa include/stringzilla/types.cuh
 */
#ifndef STRINGZILLA_METAL_H_
#define STRINGZILLA_METAL_H_

#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h" // `sz_fill_serial_`, `sz_move_serial_`

#if STRINGZILLA_WITH_METAL
#include <TargetConditionals.h> // `TARGET_OS_OSX`
#include <objc/message.h>       // `objc_msgSend`
#include <objc/runtime.h>       // `sel_registerName`, `objc_getClass`
#include <stdlib.h>             // `malloc`, `realloc`, `free`
#include <sys/mman.h>           // `mmap`, `munmap`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Runtime

/** A launch extent, laid out as @c MTLSize so it passes to Metal by value. */
typedef struct {
    sz_size_t width, height, depth;
} sz_metal_size_t;

void *MTLCreateSystemDefaultDevice(void);
#if TARGET_OS_OSX
void *MTLCopyAllDevices(void);
#endif
void *objc_autoreleasePoolPush(void);
void objc_autoreleasePoolPop(void *pool);

STRINGZILLA_INLINE void *sz_metal_get_(void *object, char const *selector) {
    return ((void *(*)(void *, SEL))objc_msgSend)(object, sel_registerName(selector));
}
STRINGZILLA_INLINE void sz_metal_do_(void *object, char const *selector) {
    ((void (*)(void *, SEL))objc_msgSend)(object, sel_registerName(selector));
}
STRINGZILLA_INLINE sz_size_t sz_metal_count_(void *object, char const *selector) {
    return ((sz_size_t (*)(void *, SEL))objc_msgSend)(object, sel_registerName(selector));
}
STRINGZILLA_INLINE void *sz_metal_string_(char const *text) {
    return ((void *(*)(void *, SEL, char const *))objc_msgSend)((void *)objc_getClass("NSString"),
                                                                sel_registerName("stringWithUTF8String:"), text);
}

/** How many Metal devices the system lists: every GPU on macOS, the default one elsewhere. */
STRINGZILLA_INLINE sz_size_t sz_metal_list_devices_(void) {
#if TARGET_OS_OSX
    void *const devices = MTLCopyAllDevices();
    if (!devices) return 0;
    sz_size_t const count = sz_metal_count_(devices, "count");
    sz_metal_do_(devices, "release");
    return count;
#else
    void *const device = MTLCreateSystemDefaultDevice();
    if (!device) return 0;
    sz_metal_do_(device, "release");
    return 1;
#endif
}

/** The system's Metal device at @p index, retained for the caller to release, or null. */
STRINGZILLA_INLINE void *sz_metal_device_(sz_size_t index) {
#if TARGET_OS_OSX
    void *const devices = MTLCopyAllDevices();
    if (!devices) return STRINGZILLA_NULL;
    void *device = STRINGZILLA_NULL;
    if (index < sz_metal_count_(devices, "count")) {
        device = ((void *(*)(void *, SEL, sz_size_t))objc_msgSend)(devices, sel_registerName("objectAtIndex:"), index);
        sz_metal_do_(device, "retain");
    }
    sz_metal_do_(devices, "release");
    return device;
#else
    return index == 0 ? MTLCreateSystemDefaultDevice() : STRINGZILLA_NULL;
#endif
}

#pragma endregion Runtime

#pragma region Device

/** One free run of the arena, as a byte offset and a length. */
typedef struct {
    sz_size_t offset, bytes;
} sz_metal_run_t;

/** One compiled kernel, keyed by its function name. */
typedef struct {
    char const *name;
    void *pipeline;
} sz_metal_pipeline_t;

/** Pipelines one device keeps: one per kernel of every family, and one per
 *  rung Levenshtein reaches. */
enum { sz_metal_pipelines_max_k = 128 };

/** Alignment of every block the arena hands out, a cache line. */
enum { sz_metal_arena_alignment_k = 64 };

/**
 *  @brief One GPU, its arena, and the command stream every Metal family encodes into.
 *
 *  Owned by whoever declared it, and passed as the @c stream of every Metal engine call. Not
 *  thread-safe: one device serves one host thread at a time.
 */
typedef struct sz_metal_device_t {

    /** The @c id<MTLDevice>. */
    void *device;

    /** Its position in the system's device list, which an engine built on it records. */
    sz_size_t ordinal;

    /** The @c id<MTLCommandQueue> command buffers come from. */
    void *queue;

    /** The command buffer and compute encoder of the call being encoded. */
    void *commands, *encoder;

    /** Command buffers committed since the last @ref sz_metal_device_synchronize, in order. */
    void **pending;
    sz_size_t pending_count, pending_capacity;

    /** The first failure since the last @ref sz_metal_device_synchronize. */
    sz_status_t status;

    /** The @c id<MTLBuffer> wrapping the whole reservation, and the reservation's host bytes. */
    void *arena;
    char *arena_host;
    sz_size_t arena_bytes;

    /** The arena's free runs, ascending by offset and never adjacent. */
    sz_metal_run_t *runs;
    sz_size_t runs_count, runs_capacity;

    /** The @c id<MTLLibrary> per family source, keyed by the source text. */
    void *libraries[8];
    char const *library_sources[8];

    /** Every pipeline built so far. */
    sz_metal_pipeline_t pipelines[sz_metal_pipelines_max_k];
    sz_size_t pipelines_count;
} sz_metal_device_t;

/**
 *  @brief Opens GPU @p ordinal with an arena of @p arena_bytes reserved, and none of it committed.
 *
 *  Pages commit as the allocator's blocks are first written, so a generous reservation costs
 *  address space alone - but the first command buffer to touch the arena pays tens of milliseconds
 *  per reserved gigabyte to map it, once.
 *
 *  @param[in] ordinal The device's position in the system's list, below @c sz_metal_count_devices.
 *  @return @c sz_success_k, @c sz_bad_alloc_k when the reservation fails, or
 *      @c sz_missing_gpu_k when there is no such Metal device.
 */
STRINGZILLA_API sz_status_t sz_metal_device_init(sz_size_t ordinal, sz_size_t arena_bytes, sz_metal_device_t *device);

/**
 *  @brief Waits for every call committed since the last synchronization.
 *  @return The first failure since the last call: an encoding refusal, or
 *      @c sz_device_code_mismatch_k when a command buffer finished in error.
 */
STRINGZILLA_API sz_status_t sz_metal_device_synchronize(sz_metal_device_t *device);

/** Drains the device, then releases every pipeline, library, the arena, and the device itself. */
STRINGZILLA_API void sz_metal_device_free(sz_metal_device_t *device);

STRINGZILLA_INLINE sz_status_t sz_metal_device_init_(sz_size_t ordinal, sz_size_t arena_bytes,
                                                     sz_metal_device_t *device) {
    sz_fill_serial_((char *)device, sizeof(*device), 0);
    device->device = sz_metal_device_(ordinal);
    if (!device->device) return sz_missing_gpu_k;
    device->ordinal = ordinal;
    device->queue = sz_metal_get_(device->device, "newCommandQueue");
    sz_size_t const page = 16384;
    arena_bytes = sz_size_divide_round_up(arena_bytes, page) * page;
    void *const host = mmap(STRINGZILLA_NULL, arena_bytes, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    device->runs = (sz_metal_run_t *)malloc(16 * sizeof(sz_metal_run_t));
    if (host == MAP_FAILED || !device->runs || !device->queue) {
        if (host != MAP_FAILED) munmap(host, arena_bytes);
        free(device->runs);
        if (device->queue) sz_metal_do_(device->queue, "release");
        sz_metal_do_(device->device, "release");
        sz_fill_serial_((char *)device, sizeof(*device), 0);
        return sz_bad_alloc_k;
    }
    sz_size_t const shared_storage = 0; // `MTLResourceStorageModeShared`
    device->arena = ((void *(*)(void *, SEL, void *, sz_size_t, sz_size_t, void *))objc_msgSend)(
        device->device, sel_registerName("newBufferWithBytesNoCopy:length:options:deallocator:"), host, arena_bytes,
        shared_storage, STRINGZILLA_NULL);
    device->arena_host = (char *)host, device->arena_bytes = arena_bytes;
    device->runs[0].offset = 0, device->runs[0].bytes = arena_bytes;
    device->runs_count = 1, device->runs_capacity = 16;
    return device->arena ? sz_success_k : sz_bad_alloc_k;
}

/** Records @p status as the device's failure, unless an earlier one still waits to be reported. */
STRINGZILLA_INLINE void sz_metal_fail_(sz_metal_device_t *device, sz_status_t status) {
    if (device->status == sz_success_k) device->status = status;
}

STRINGZILLA_INLINE sz_status_t sz_metal_device_synchronize_(sz_metal_device_t *device) {
    sz_size_t const completed = 4; // `MTLCommandBufferStatusCompleted`
    for (sz_size_t index = 0; index != device->pending_count; ++index) {
        sz_metal_do_(device->pending[index], "waitUntilCompleted");
        if (sz_metal_count_(device->pending[index], "status") != completed)
            sz_metal_fail_(device, sz_device_code_mismatch_k);
        sz_metal_do_(device->pending[index], "release");
    }
    device->pending_count = 0;
    sz_status_t const status = device->status;
    device->status = sz_success_k;
    return status;
}

STRINGZILLA_INLINE void sz_metal_device_free_(sz_metal_device_t *device) {
    if (!device->device) return;
    sz_metal_device_synchronize_(device);
    for (sz_size_t index = 0; index != device->pipelines_count; ++index)
        sz_metal_do_(device->pipelines[index].pipeline, "release");
    for (sz_size_t index = 0; index != sizeof(device->libraries) / sizeof(device->libraries[0]); ++index)
        if (device->libraries[index]) sz_metal_do_(device->libraries[index], "release");
    if (device->arena) sz_metal_do_(device->arena, "release");
    if (device->arena_host) munmap(device->arena_host, device->arena_bytes);
    free(device->runs);
    free(device->pending);
    sz_metal_do_(device->queue, "release");
    sz_metal_do_(device->device, "release");
    sz_fill_serial_((char *)device, sizeof(*device), 0);
}

/** Whether @p pointer lies inside @p device 's arena, the only memory its kernels reach. */
STRINGZILLA_INLINE sz_bool_t sz_memory_reaches_metal(sz_metal_device_t const *device, void const *pointer) {
    char const *const address = (char const *)pointer;
    return address >= device->arena_host && address < device->arena_host + device->arena_bytes ? sz_true_k : sz_false_k;
}

/** First fit over the free runs; @return the block, or null when no run is wide enough. */
STRINGZILLA_INLINE void *sz_metal_arena_allocate_(sz_size_t bytes, void *handle) {
    sz_metal_device_t *const device = (sz_metal_device_t *)handle;
    bytes = bytes ? sz_size_divide_round_up(bytes, sz_metal_arena_alignment_k) * sz_metal_arena_alignment_k
                  : sz_metal_arena_alignment_k;
    for (sz_size_t index = 0; index != device->runs_count; ++index) {
        sz_metal_run_t *const run = device->runs + index;
        if (run->bytes < bytes) continue;
        char *const block = device->arena_host + run->offset;
        run->offset += bytes, run->bytes -= bytes;
        if (!run->bytes) {
            sz_move_serial_((char *)(run), (char const *)(run + 1),
                            (device->runs_count - index - 1) * sizeof(sz_metal_run_t));
            --device->runs_count;
        }
        return block;
    }
    return STRINGZILLA_NULL;
}

/** Returns a block to the free runs, merging it with the neighbours it touches. */
STRINGZILLA_INLINE void sz_metal_arena_free_(void *pointer, sz_size_t bytes, void *handle) {
    sz_metal_device_t *const device = (sz_metal_device_t *)handle;
    if (!pointer) return;
    bytes = bytes ? sz_size_divide_round_up(bytes, sz_metal_arena_alignment_k) * sz_metal_arena_alignment_k
                  : sz_metal_arena_alignment_k;
    sz_size_t const offset = (sz_size_t)((char *)pointer - device->arena_host);
    sz_size_t position = 0;
    while (position != device->runs_count && device->runs[position].offset < offset) ++position;
    sz_bool_t const joins_before =
        position != 0 && device->runs[position - 1].offset + device->runs[position - 1].bytes == offset ? sz_true_k
                                                                                                        : sz_false_k;
    sz_bool_t const joins_after = position != device->runs_count && offset + bytes == device->runs[position].offset
                                      ? sz_true_k
                                      : sz_false_k;
    if (joins_before && joins_after) {
        device->runs[position - 1].bytes += bytes + device->runs[position].bytes;
        sz_move_serial_((char *)(device->runs + position), (char const *)(device->runs + position + 1),
                        (device->runs_count - position - 1) * sizeof(sz_metal_run_t));
        --device->runs_count;
        return;
    }
    if (joins_before) {
        device->runs[position - 1].bytes += bytes;
        return;
    }
    if (joins_after) {
        device->runs[position].offset = offset, device->runs[position].bytes += bytes;
        return;
    }
    if (device->runs_count == device->runs_capacity) {
        sz_metal_run_t *const grown = (sz_metal_run_t *)realloc(device->runs,
                                                                device->runs_capacity * 2 * sizeof(sz_metal_run_t));
        if (!grown) return; // The run is lost to the arena, never to the host.
        device->runs = grown, device->runs_capacity *= 2;
    }
    sz_move_serial_((char *)(device->runs + position + 1), (char const *)(device->runs + position),
                    (device->runs_count - position) * sizeof(sz_metal_run_t));
    device->runs[position].offset = offset, device->runs[position].bytes = bytes;
    ++device->runs_count;
}

/** Initializes an allocator handing back arena memory both the host and @p device 's kernels
 *  address, the twin of @c sz_cuda_memory_allocator_init_unified. A block freed returns to the
 *  arena at once, so a round still reading it, or an engine built in it, is synchronized first. */
STRINGZILLA_INLINE void sz_memory_allocator_init_metal(sz_memory_allocator_t *allocator, sz_metal_device_t *device) {
    allocator->allocate = sz_metal_arena_allocate_;
    allocator->free = sz_metal_arena_free_;
    allocator->handle = device;
}

/** Whether two NUL-terminated names match; each translation unit embeds its own source copy. */
STRINGZILLA_INLINE sz_bool_t sz_metal_same_text_(char const *first, char const *second) {
    while (*first && *first == *second) ++first, ++second;
    return *first == *second ? sz_true_k : sz_false_k;
}

/**
 *  @brief Kernel @p name's pipeline from the library built out of @p source, built on first use.
 *  @return The pipeline, or null after recording @c sz_device_code_mismatch_k.
 */
STRINGZILLA_INLINE void *sz_metal_pipeline_(sz_metal_device_t *device, char const *source, char const *name) {
    for (sz_size_t index = 0; index != device->pipelines_count; ++index)
        if (sz_metal_same_text_(device->pipelines[index].name, name)) return device->pipelines[index].pipeline;
    sz_size_t const slots = sizeof(device->libraries) / sizeof(device->libraries[0]);
    sz_size_t slot = 0;
    while (slot != slots && device->library_sources[slot] &&
           !sz_metal_same_text_(device->library_sources[slot], source))
        ++slot;
    if (slot == slots || device->pipelines_count == sz_metal_pipelines_max_k) {
        sz_metal_fail_(device, sz_device_code_mismatch_k);
        return STRINGZILLA_NULL;
    }

    void *const pool = objc_autoreleasePoolPush();
    void *pipeline = STRINGZILLA_NULL;
    if (!device->libraries[slot]) {
        void *const options = sz_metal_get_(sz_metal_get_((void *)objc_getClass("MTLCompileOptions"), "alloc"), "init");
        sz_size_t const metal_3_2 = (3u << 16) | 2u; // `MTLLanguageVersion3_2`
        ((void (*)(void *, SEL, sz_size_t))objc_msgSend)(options, sel_registerName("setLanguageVersion:"), metal_3_2);
        void *error = STRINGZILLA_NULL;
        device->libraries[slot] = ((void *(*)(void *, SEL, void *, void *, void **))objc_msgSend)(
            device->device, sel_registerName("newLibraryWithSource:options:error:"), sz_metal_string_(source), options,
            &error);
        sz_metal_do_(options, "release");
        if (device->libraries[slot]) device->library_sources[slot] = source;
    }
    if (device->libraries[slot]) {
        void *const function = ((void *(*)(void *, SEL, void *))objc_msgSend)(
            device->libraries[slot], sel_registerName("newFunctionWithName:"), sz_metal_string_(name));
        void *error = STRINGZILLA_NULL;
        if (function) {
            pipeline = ((void *(*)(void *, SEL, void *, void **))objc_msgSend)(
                device->device, sel_registerName("newComputePipelineStateWithFunction:error:"), function, &error);
            sz_metal_do_(function, "release");
        }
    }
    objc_autoreleasePoolPop(pool);
    if (!pipeline) {
        sz_metal_fail_(device, sz_device_code_mismatch_k);
        return STRINGZILLA_NULL;
    }
    device->pipelines[device->pipelines_count].name = name;
    device->pipelines[device->pipelines_count].pipeline = pipeline;
    ++device->pipelines_count;
    return pipeline;
}

/** Opens one call's command buffer and compute encoder; @ref sz_metal_dispatch_ commits them. */
STRINGZILLA_INLINE void *sz_metal_encoder_(sz_metal_device_t *device) {
    if (device->encoder) return device->encoder;
    if (device->pending_count == device->pending_capacity) {
        sz_size_t const capacity = device->pending_capacity ? device->pending_capacity * 2 : 16;
        void **grown = (void **)realloc(device->pending, capacity * sizeof(void *));
        if (!grown) {
            sz_metal_fail_(device, sz_bad_alloc_k);
            return STRINGZILLA_NULL;
        }
        device->pending = grown, device->pending_capacity = capacity;
    }
    void *const pool = objc_autoreleasePoolPush();
    device->commands = sz_metal_get_(sz_metal_get_(device->queue, "commandBuffer"), "retain");
    if (device->commands)
        device->encoder = sz_metal_get_(sz_metal_get_(device->commands, "computeCommandEncoder"), "retain");
    objc_autoreleasePoolPop(pool);
    if (!device->encoder) sz_metal_fail_(device, sz_device_code_mismatch_k);
    return device->encoder;
}

/**
 *  @brief Encodes @p pipeline over @p groups threadgroups of @p threads, leaving the call open.
 *
 *  A compute encoder runs its dispatches serially, each seeing the writes of those before it, so a
 *  multi-pass call encodes every pass here and pays for one command buffer.
 */
STRINGZILLA_INLINE void sz_metal_enqueue_(sz_metal_device_t *device, void *pipeline, sz_metal_size_t groups,
                                          sz_metal_size_t threads) {
    void *const encoder = device->encoder;
    ((void (*)(void *, SEL, void *))objc_msgSend)(encoder, sel_registerName("setComputePipelineState:"), pipeline);
    ((void (*)(void *, SEL, sz_metal_size_t, sz_metal_size_t))objc_msgSend)(
        encoder, sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"), groups, threads);
}

/** Commits the call @ref sz_metal_encoder_ opened, so it starts running while the host moves on. */
STRINGZILLA_INLINE void sz_metal_commit_(sz_metal_device_t *device) {
    void *const encoder = device->encoder;
    sz_metal_do_(encoder, "endEncoding");
    sz_metal_do_(encoder, "release");
    sz_metal_do_(device->commands, "commit");
    device->pending[device->pending_count++] = device->commands;
    device->commands = STRINGZILLA_NULL, device->encoder = STRINGZILLA_NULL;
}

/** Encodes one dispatch and commits it, as a CUDA launch starts. */
STRINGZILLA_INLINE void sz_metal_dispatch_(sz_metal_device_t *device, void *pipeline, sz_metal_size_t groups,
                                           sz_metal_size_t threads) {
    sz_metal_enqueue_(device, pipeline, groups, threads);
    sz_metal_commit_(device);
}

/**
 *  @brief Binds the arena at buffer zero and @p arguments at buffer one, then encodes kernel
 *      @p name of @p source over @p groups of @p threads, leaving the call open.
 *  @return @c sz_success_k, or the failure the device recorded for a missing kernel or encoder.
 */
STRINGZILLA_INLINE sz_status_t sz_metal_encode_(sz_metal_device_t *device, char const *source, char const *name,
                                                void const *arguments, sz_size_t arguments_bytes,
                                                sz_metal_size_t groups, sz_metal_size_t threads) {
    void *const pipeline = sz_metal_pipeline_(device, source, name);
    void *const encoder = pipeline ? sz_metal_encoder_(device) : STRINGZILLA_NULL;
    if (!encoder) return device->status;
    ((void (*)(void *, SEL, void *, sz_size_t, sz_size_t))objc_msgSend)(
        encoder, sel_registerName("setBuffer:offset:atIndex:"), device->arena, 0, 0);
    ((void (*)(void *, SEL, void const *, sz_size_t, sz_size_t))objc_msgSend)(
        encoder, sel_registerName("setBytes:length:atIndex:"), arguments, arguments_bytes, 1);
    sz_metal_enqueue_(device, pipeline, groups, threads);
    return sz_success_k;
}

/**
 *  @brief Whether @p sequence is a view array inside @p device 's arena: the one layout a Metal
 *      kernel reads, as Metal has no device function pointers to call cheaply.
 *  @return @c sz_success_k, @c sz_device_memory_mismatch_k for views outside the arena, or
 *      @c sz_device_code_mismatch_k for accessors answering anything but the views they sit on.
 */
STRINGZILLA_INLINE sz_status_t sz_metal_views_(sz_metal_device_t const *device, sz_sequence_t const *sequence) {
    sz_string_view_t const *const views = (sz_string_view_t const *)sequence->handle;
    sz_size_t const last = sequence->count - 1;
    if (!sequence->count) return sz_success_k;
    if (!sz_memory_reaches_metal(device, views) || !sz_memory_reaches_metal(device, views + last))
        return sz_device_memory_mismatch_k;
    // Known by what the accessors answer rather than by their address, since every translation unit
    // carries its own copy of the view accessors.
    if (sequence->get_start(views, 0) != views[0].start || sequence->get_length(views, 0) != views[0].length ||
        sequence->get_start(views, last) != views[last].start ||
        sequence->get_length(views, last) != views[last].length)
        return sz_device_code_mismatch_k;
    return sz_success_k;
}

#pragma endregion Device

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_metal_device_init(sz_size_t ordinal, sz_size_t arena_bytes, sz_metal_device_t *device) {
    return sz_metal_device_init_(ordinal, arena_bytes, device);
}

STRINGZILLA_API sz_status_t sz_metal_device_synchronize(sz_metal_device_t *device) {
    return sz_metal_device_synchronize_(device);
}

STRINGZILLA_API void sz_metal_device_free(sz_metal_device_t *device) { sz_metal_device_free_(device); }

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_WITH_METAL
#endif // STRINGZILLA_METAL_H_
