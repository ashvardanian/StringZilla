/**
 *  @file include/stringzilla/metal.h
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief Core types on an Apple GPU: the library's state per device, the unified memory both sides
 *      address, and the calls every Metal family encodes, the twin of `types.cuh`.
 *
 *  A stream is an @c id<MTLCommandQueue>, which names its device; a null stream means the system
 *  default device's own queue. The library keeps one @ref sz_metal_context_t per device for the
 *  process, as a CUDA primary context lives: that default queue, the compiled pipelines, the
 *  command buffers committed to each stream, and a registry of every shared buffer the unified
 *  allocator handed out, sorted by host address.
 *
 *  A shared buffer's host address is never its GPU address, so a kernel gets every block it reads
 *  as a buffer of its own, at the offset @ref sz_metal_resolve_ finds in the registry. A block
 *  points inside itself by host address, which the kernel turns into its own against the host
 *  address the block was bound at. No block points into another, which is why sequences are tapes.
 *
 *  Encoders live on each call's stack, so any number of threads may encode into any queues; the
 *  registry, the pending command buffers and the pipelines sit under the context's lock.
 *
 *  Written in C, as every GPU header in this library is. Metal's API is Objective-C, reached
 *  through @c objc_msgSend cast to each call's exact prototype, so no translation unit here needs
 *  an Objective-C compiler.
 *
 *  @sa include/stringzilla/types.cuh
 *  @sa include/stringzilla/types.metal
 */
#ifndef STRINGZILLA_METAL_H_
#define STRINGZILLA_METAL_H_

#include "stringzilla/types.h"
#include "stringzilla/memory/serial.h"  // `sz_move_serial_`
#include "stringzilla/compare/serial.h" // `sz_equal_serial_`

#if STRINGZILLA_WITH_METAL
#include <TargetConditionals.h> // `TARGET_OS_OSX`
#include <objc/message.h>       // `objc_msgSend`
#include <objc/runtime.h>       // `sel_registerName`, `objc_getClass`
#include <os/lock.h>            // `os_unfair_lock`
#include <stdlib.h>             // `realloc`, `free`

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

/** The system's Metal device at @p ordinal, retained for the caller to release, or null. */
STRINGZILLA_INLINE void *sz_metal_device_(sz_size_t ordinal) {
#if TARGET_OS_OSX
    void *const devices = MTLCopyAllDevices();
    if (!devices) return STRINGZILLA_NULL;
    void *device = STRINGZILLA_NULL;
    if (ordinal < sz_metal_count_(devices, "count")) {
        device = ((void *(*)(void *, SEL, sz_size_t))objc_msgSend)(devices, sel_registerName("objectAtIndex:"),
                                                                   ordinal);
        sz_metal_do_(device, "retain");
    }
    sz_metal_do_(devices, "release");
    return device;
#else
    return ordinal == 0 ? MTLCreateSystemDefaultDevice() : STRINGZILLA_NULL;
#endif
}

#pragma endregion Runtime

#pragma region Context

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"

/** The MSL prelude every family library is compiled behind, ahead of the family's own source. */
static char const sz_metal_prelude_source_[] = {
#embed "stringzilla/types.metal"
    , 0};
#pragma clang diagnostic pop

typedef struct {
    char const *name;
    void *library, *pipeline;
} sz_metal_pipeline_t;

typedef struct {
    char const *source;
    void *library;
} sz_metal_library_t;

/** Devices the library keeps state for, more than any Mac carries. */
enum { sz_metal_contexts_max_k = 8 };

/** One block of the unified allocator. */
typedef struct sz_metal_block_t {

    /** The @c id<MTLBuffer> in shared storage. */
    void *buffer;

    /** Its contents, the address every host pointer into it falls inside. */
    char *host;

    /** Bytes the allocation asked for. */
    sz_size_t bytes;

} sz_metal_block_t;

/** One stream with committed work. */
typedef struct sz_metal_pending_t {

    /** The @c id<MTLCommandQueue>, retained while pending. */
    sz_stream_t queue;

    /** Committed command buffers not yet removed after completion, in order. */
    void **commands;
    sz_size_t commands_count, commands_capacity;

    /** Synchronizations still waiting on their retained command snapshot. */
    sz_size_t waiters;
} sz_metal_pending_t;

/** What the library keeps for one device, for the life of the process. */
typedef struct sz_metal_context_t {

    /** The @c id<MTLDevice>. */
    void *device;

    /** Its default @c id<MTLCommandQueue>, which a null stream means. */
    sz_stream_t queue;

    /** Guards every member below. */
    os_unfair_lock lock;

    /** Every live block, sorted by host address. */
    sz_metal_block_t *blocks;
    sz_size_t blocks_count, blocks_capacity;

    /** Every stream with committed work or a synchronization in progress. */
    sz_metal_pending_t *pending;
    sz_size_t pending_count, pending_capacity;

    /** The @c id<MTLLibrary> per family source, keyed by the source text. */
    sz_metal_library_t *libraries;
    sz_size_t libraries_count, libraries_capacity;

    /** Every pipeline built so far. */
    sz_metal_pipeline_t *pipelines;
    sz_size_t pipelines_count, pipelines_capacity;
} sz_metal_context_t;

/** Where a host pointer lies for a kernel: the buffer holding it and its offset there. */
typedef struct sz_metal_bound_t {
    void *buffer;
    sz_size_t offset;
} sz_metal_bound_t;

/** One call's encoding state, on its caller's stack. */
typedef struct sz_metal_call_t {

    /** The device's state. */
    sz_metal_context_t *context;

    /** The @c id<MTLCommandQueue> the call encodes into. */
    sz_stream_t queue;

    /** The retained command buffer and compute encoder. */
    void *commands, *encoder;
} sz_metal_call_t;

/** Every device's context, the live ones first, each kept for the process once built, and the lock
 *  in @p contexts_lock that guards which of them are live. */
STRINGZILLA_API sz_metal_context_t *sz_metal_contexts_(os_unfair_lock_t *contexts_lock);

#if STRINGZILLA_HEADER_ONLY
STRINGZILLA_API sz_metal_context_t *sz_metal_contexts_(os_unfair_lock_t *contexts_lock) {
    // Zero is `OS_UNFAIR_LOCK_INIT`, so the static storage starts unlocked.
    static sz_metal_context_t contexts[sz_metal_contexts_max_k];
    static os_unfair_lock lock;
    *contexts_lock = &lock;
    return contexts;
}
#endif

/**
 *  @brief The library's state for the device of @p stream, or of the system default device for a
 *      null one, built on first use and kept for the process.
 *  @return The context, or null without a device or past @ref sz_metal_contexts_max_k devices.
 */
STRINGZILLA_INLINE sz_metal_context_t *sz_metal_context_(sz_stream_t stream) {
    os_unfair_lock_t contexts_lock;
    sz_metal_context_t *const contexts = sz_metal_contexts_(&contexts_lock);
    void *const device = stream ? sz_metal_get_(stream, "device") : MTLCreateSystemDefaultDevice();
    if (!device) return STRINGZILLA_NULL;
    if (stream) sz_metal_do_(device, "retain");

    sz_metal_context_t *context = STRINGZILLA_NULL;
    sz_bool_t kept = sz_false_k;
    os_unfair_lock_lock(contexts_lock);
    for (sz_size_t index = 0; index != sz_metal_contexts_max_k; ++index) {
        if (contexts[index].device == device) {
            context = contexts + index;
            break;
        }
        if (contexts[index].device) continue;
        sz_stream_t const queue = sz_metal_get_(device, "newCommandQueue");
        if (queue) contexts[index].device = device, contexts[index].queue = queue, context = contexts + index;
        kept = queue ? sz_true_k : sz_false_k;
        break;
    }
    os_unfair_lock_unlock(contexts_lock);
    if (!kept) sz_metal_do_(device, "release");
    return context;
}

/** Resolves the stream a call names: @p stream itself, or @p context 's default queue for null. */
STRINGZILLA_INLINE sz_stream_t sz_metal_queue_(sz_metal_context_t const *context, sz_stream_t stream) {
    return stream ? stream : context->queue;
}

/**
 *  @brief Opens one call and compute encoder on @p stream 's device.
 *  @return @c sz_success_k, or @c sz_missing_gpu_k without a device.
 */
STRINGZILLA_INLINE sz_status_t sz_device_enter_metal_(sz_stream_t stream, sz_metal_call_t *call) {
    call->queue = STRINGZILLA_NULL, call->commands = STRINGZILLA_NULL, call->encoder = STRINGZILLA_NULL;
    call->context = sz_metal_context_(stream);
    if (!call->context) return sz_missing_gpu_k;
    call->queue = sz_metal_get_(stream ? stream : call->context->queue, "retain");
    void *const pool = objc_autoreleasePoolPush();
    call->commands = sz_metal_get_(sz_metal_get_(call->queue, "commandBuffer"), "retain");
    if (call->commands) call->encoder = sz_metal_get_(sz_metal_get_(call->commands, "computeCommandEncoder"), "retain");
    objc_autoreleasePoolPop(pool);
    if (!call->encoder) {
        if (call->commands) sz_metal_do_(call->commands, "release");
        sz_metal_do_(call->queue, "release");
        call->queue = STRINGZILLA_NULL, call->commands = STRINGZILLA_NULL;
        return sz_device_code_mismatch_k;
    }
    return sz_success_k;
}

/** How many registered blocks start at or before @p address, which is also where a block starting
 *  there belongs. Called under the lock. */
STRINGZILLA_INLINE sz_size_t sz_metal_blocks_upto_(sz_metal_context_t const *context, char const *address) {
    sz_size_t low = 0, high = context->blocks_count;
    while (low < high) {
        sz_size_t const middle = low + (high - low) / 2;
        if ((sz_size_t)context->blocks[middle].host <= (sz_size_t)address) low = middle + 1;
        else high = middle;
    }
    return low;
}

/**
 *  @brief Finds the block holding the @p bytes at @p pointer, as a kernel reaches no other memory.
 *  @return Whether one registered block holds them all, with @p bound set when it does.
 */
STRINGZILLA_INLINE sz_bool_t sz_metal_resolve_(sz_metal_context_t *context, void const *pointer, sz_size_t bytes,
                                               sz_metal_bound_t *bound) {
    char const *const address = (char const *)pointer;
    sz_bool_t found = sz_false_k;
    os_unfair_lock_lock(&context->lock);
    sz_size_t const upto = sz_metal_blocks_upto_(context, address);
    if (upto) {
        sz_metal_block_t const *const block = context->blocks + upto - 1;
        sz_size_t const offset = (sz_size_t)address - (sz_size_t)block->host;
        if (offset <= block->bytes && bytes <= block->bytes - offset)
            bound->buffer = block->buffer, bound->offset = offset, found = sz_true_k;
    }
    os_unfair_lock_unlock(&context->lock);
    return found;
}

STRINGZILLA_INLINE void *sz_metal_reserve_(void *items, sz_size_t count, sz_size_t *capacity, sz_size_t size) {
    if (count < *capacity) return items;
    if (*capacity > ((sz_size_t)-1) / 2) return STRINGZILLA_NULL;
    sz_size_t const grown_capacity = *capacity ? *capacity * 2 : 16;
    if (!size || grown_capacity > ((sz_size_t)-1) / size) return STRINGZILLA_NULL;
    void *const grown = realloc(items, grown_capacity * size);
    if (grown) *capacity = grown_capacity;
    return grown;
}

STRINGZILLA_INLINE sz_bool_t sz_metal_resolve_call_(sz_metal_call_t *call, void const *pointer, sz_size_t bytes,
                                                    sz_metal_bound_t *bound) {
    sz_metal_context_t *const context = call->context;
    sz_bool_t found = sz_false_k;
    os_unfair_lock_lock(&context->lock);
    sz_size_t const upto = sz_metal_blocks_upto_(context, (char const *)pointer);
    sz_metal_block_t const *const block = upto ? context->blocks + upto - 1 : STRINGZILLA_NULL;
    sz_size_t const offset = block ? (sz_size_t)pointer - (sz_size_t)block->host : 0;
    if (block && offset <= block->bytes && bytes <= block->bytes - offset) {
        // The command buffer retains the resource before the registry lock is released.
        ((void (*)(void *, SEL, void *, sz_size_t))objc_msgSend)(call->encoder, sel_registerName("useResource:usage:"),
                                                                 block->buffer, 3);
        bound->buffer = block->buffer, bound->offset = offset, found = sz_true_k;
    }
    os_unfair_lock_unlock(&context->lock);
    return found;
}

/** The pending entry of @p queue, or null. Called under the lock. */
STRINGZILLA_INLINE sz_metal_pending_t *sz_metal_pending_find_(sz_metal_context_t *context, void const *queue) {
    for (sz_size_t index = 0; index != context->pending_count; ++index)
        if (context->pending[index].queue == queue) return context->pending + index;
    return STRINGZILLA_NULL;
}

/** The pending entry of @p queue with room for one more command buffer, added when missing.
 *  Called under the lock. @return The entry, or null when it cannot grow. */
STRINGZILLA_INLINE sz_metal_pending_t *sz_metal_pending_reserve_(sz_metal_context_t *context, sz_stream_t queue) {
    sz_metal_pending_t *pending = sz_metal_pending_find_(context, queue);
    if (!pending) {
        sz_metal_pending_t *const grown = (sz_metal_pending_t *)sz_metal_reserve_(
            context->pending, context->pending_count, &context->pending_capacity, sizeof(sz_metal_pending_t));
        if (!grown) return STRINGZILLA_NULL;
        context->pending = grown;
        pending = context->pending + context->pending_count++;
        pending->queue = sz_metal_get_(queue, "retain");
        pending->commands = STRINGZILLA_NULL, pending->commands_count = 0, pending->commands_capacity = 0;
        pending->waiters = 0;
    }
    void **const grown = (void **)sz_metal_reserve_(pending->commands, pending->commands_count,
                                                    &pending->commands_capacity, sizeof(void *));
    if (!grown) return STRINGZILLA_NULL;
    pending->commands = grown;
    return pending;
}

STRINGZILLA_INLINE void *sz_allocate_unified_metal_(sz_size_t bytes, void *handle, sz_stream_t stream) {
    sz_metal_context_t *const context = sz_metal_context_(stream);
    sz_unused_(handle);
    if (!context) return STRINGZILLA_NULL;
    // A buffer holds at least one byte, so an empty request still gets a block to free.
    bytes = bytes ? bytes : 1;
    sz_size_t const shared_storage = 0; // `MTLResourceStorageModeShared`
    void *const buffer = ((void *(*)(void *, SEL, sz_size_t, sz_size_t))objc_msgSend)(
        context->device, sel_registerName("newBufferWithLength:options:"), bytes, shared_storage);
    if (!buffer) return STRINGZILLA_NULL;
    char *const host = (char *)sz_metal_get_(buffer, "contents");

    sz_bool_t registered = sz_true_k;
    os_unfair_lock_lock(&context->lock);
    sz_metal_block_t *const grown = (sz_metal_block_t *)sz_metal_reserve_(
        context->blocks, context->blocks_count, &context->blocks_capacity, sizeof(sz_metal_block_t));
    if (grown) context->blocks = grown;
    else registered = sz_false_k;
    if (registered) {
        sz_size_t const position = sz_metal_blocks_upto_(context, host);
        sz_move_serial_((char *)(context->blocks + position + 1), (char const *)(context->blocks + position),
                        (context->blocks_count - position) * sizeof(sz_metal_block_t));
        sz_metal_block_t *const block = context->blocks + position;
        block->buffer = buffer, block->host = host, block->bytes = bytes;
        ++context->blocks_count;
    }
    os_unfair_lock_unlock(&context->lock);
    if (!registered) sz_metal_do_(buffer, "release");
    return registered ? host : STRINGZILLA_NULL;
}

/**
 *  @brief Unregisters and releases the block starting at @p pointer if @p context holds it.
 *  @return Whether @p context holds it.
 */
STRINGZILLA_INLINE sz_bool_t sz_metal_free_in_(sz_metal_context_t *context, void *pointer) {
    void *released = STRINGZILLA_NULL;
    os_unfair_lock_lock(&context->lock);
    sz_size_t const upto = sz_metal_blocks_upto_(context, (char const *)pointer);
    sz_metal_block_t *const block = upto ? context->blocks + upto - 1 : STRINGZILLA_NULL;
    sz_bool_t const held = block && block->host == pointer ? sz_true_k : sz_false_k;
    if (held) {
        released = block->buffer;
        sz_move_serial_((char *)block, (char const *)(block + 1),
                        (context->blocks_count - upto) * sizeof(sz_metal_block_t));
        --context->blocks_count;
    }
    os_unfair_lock_unlock(&context->lock);
    if (released) sz_metal_do_(released, "release");
    return held;
}

STRINGZILLA_INLINE void sz_free_unified_metal_(void *pointer, sz_size_t bytes, void *handle, sz_stream_t stream) {
    sz_unused_(bytes && handle);
    if (!pointer) return;
    sz_metal_context_t *const context = sz_metal_context_(stream);
    if (context && sz_metal_free_in_(context, pointer)) return;

    // A null stream may name a different device from the allocation.
    os_unfair_lock_t contexts_lock;
    sz_metal_context_t *const contexts = sz_metal_contexts_(&contexts_lock);
    sz_size_t live = 0;
    os_unfair_lock_lock(contexts_lock);
    while (live != sz_metal_contexts_max_k && contexts[live].device) ++live;
    os_unfair_lock_unlock(contexts_lock);
    for (sz_size_t index = 0; index != live; ++index)
        if (contexts + index != context && sz_metal_free_in_(contexts + index, pointer)) return;
}

STRINGZILLA_INLINE sz_status_t sz_allocator_init_unified_metal_(sz_allocator_t *allocator) {
    allocator->allocate = sz_allocate_unified_metal_;
    allocator->free = sz_free_unified_metal_;
    allocator->handle = STRINGZILLA_NULL;
    return sz_success_k;
}

STRINGZILLA_INLINE sz_status_t sz_sequence_realloc_metal_(sz_sequence_t *target, sz_sequence_t const *source,
                                                          sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                          sz_stream_t stream) {
    sz_metal_context_t *const context = sz_metal_context_(stream);
    if (!context) return sz_missing_gpu_k;
    sz_metal_bound_t bound;
    sz_sequence_t next = *source;
    sz_u64_t const *const source_offsets = (sz_u64_t const *)source->handle;
    sz_size_t bytes = 0;
    sz_status_t status = sz_success_k;
    if (source->get_start != sz_sequence_tape_start || source->get_length != sz_sequence_tape_length ||
        !sz_metal_resolve_(context, source->handle, (sz_size_t)source_offsets[source->count], &bound))
        status = sz_sequence_realloc_serial_(&next, source, allocator, &bytes, stream);
    sz_u64_t const *const next_offsets = (sz_u64_t const *)next.handle;
    if (status == sz_success_k &&
        !sz_metal_resolve_(context, next.handle, (sz_size_t)next_offsets[next.count], &bound)) {
        if (bytes) allocator->free((void *)next.handle, bytes, allocator->handle, stream);
        status = sz_device_memory_mismatch_k;
    }
    if (status == sz_success_k) *target = next, *allocated_bytes = bytes;
    return status;
}

STRINGZILLA_INLINE sz_status_t sz_stream_synchronize_metal_(sz_stream_t stream) {
    sz_metal_context_t *const context = sz_metal_context_(stream);
    if (!context) return sz_missing_gpu_k;
    sz_stream_t const queue = sz_metal_queue_(context, stream);
    void **commands = STRINGZILLA_NULL;
    sz_size_t commands_count = 0;
    os_unfair_lock_lock(&context->lock);
    sz_metal_pending_t *pending = sz_metal_pending_find_(context, queue);
    if (pending) {
        commands_count = pending->commands_count;
        if (commands_count) commands = (void **)malloc(commands_count * sizeof(void *));
        if (commands_count && !commands) {
            os_unfair_lock_unlock(&context->lock);
            return sz_bad_alloc_k;
        }
        for (sz_size_t index = 0; index != commands_count; ++index)
            commands[index] = sz_metal_get_(pending->commands[index], "retain");
        ++pending->waiters;
    }
    os_unfair_lock_unlock(&context->lock);
    if (!pending) return sz_success_k;

    sz_status_t status = sz_success_k;
    for (sz_size_t index = 0; index != commands_count; ++index) {
        sz_metal_do_(commands[index], "waitUntilCompleted");
        if (sz_metal_count_(commands[index], "status") != 4) status = sz_device_code_mismatch_k;
        sz_metal_do_(commands[index], "release");
    }
    free(commands);

    void *drained = STRINGZILLA_NULL;
    os_unfair_lock_lock(&context->lock);
    pending = sz_metal_pending_find_(context, queue);
    sz_size_t completed = 0;
    while (completed != pending->commands_count && sz_metal_count_(pending->commands[completed], "status") >= 4)
        sz_metal_do_(pending->commands[completed++], "release");
    if (completed) {
        pending->commands_count -= completed;
        sz_move_serial_((char *)pending->commands, (char const *)(pending->commands + completed),
                        pending->commands_count * sizeof(void *));
    }
    if (--pending->waiters == 0 && pending->commands_count == 0) {
        drained = pending->queue;
        free(pending->commands);
        *pending = context->pending[--context->pending_count];
    }
    os_unfair_lock_unlock(&context->lock);

    if (drained) sz_metal_do_(drained, "release");
    return status;
}

/** Kernel @p name's pipeline from the library built out of @p source, built on first use under the
 *  lock. @return The pipeline, or null when it fails to build. */
STRINGZILLA_INLINE void *sz_metal_pipeline_(sz_metal_context_t *context, char const *source, char const *name) {
    sz_size_t const language_version = (3u << 16) | 2u;
    void *pipeline = STRINGZILLA_NULL;
    os_unfair_lock_lock(&context->lock);
    sz_size_t source_length = 0, name_length = 0;
    while (source[source_length]) ++source_length;
    while (name[name_length]) ++name_length;
    sz_size_t slot = 0;
    for (; slot != context->libraries_count; ++slot) {
        char const *const cached_source = context->libraries[slot].source;
        if (cached_source == source) break;
        sz_size_t cached_length = 0;
        while (cached_source[cached_length]) ++cached_length;
        if (cached_length == source_length && sz_equal_serial_(cached_source, source, source_length)) break;
    }
    void *const cached_library = slot < context->libraries_count ? context->libraries[slot].library : STRINGZILLA_NULL;
    for (sz_size_t index = 0; index != context->pipelines_count; ++index) {
        sz_metal_pipeline_t const *const entry = context->pipelines + index;
        if (entry->library != cached_library) continue;
        sz_size_t cached_length = 0;
        while (entry->name[cached_length]) ++cached_length;
        if (cached_length == name_length && sz_equal_serial_(entry->name, name, name_length)) {
            pipeline = entry->pipeline;
            break;
        }
    }
    if (pipeline) {
        os_unfair_lock_unlock(&context->lock);
        return pipeline;
    }
    sz_metal_pipeline_t *const pipelines = (sz_metal_pipeline_t *)sz_metal_reserve_(
        context->pipelines, context->pipelines_count, &context->pipelines_capacity, sizeof(sz_metal_pipeline_t));
    if (!pipelines) {
        os_unfair_lock_unlock(&context->lock);
        return STRINGZILLA_NULL;
    }
    context->pipelines = pipelines;
    void *const pool = objc_autoreleasePoolPush();
    if (slot == context->libraries_count) {
        sz_metal_library_t *const libraries = (sz_metal_library_t *)sz_metal_reserve_(
            context->libraries, context->libraries_count, &context->libraries_capacity, sizeof(sz_metal_library_t));
        if (libraries) {
            context->libraries = libraries;
            void *const options = sz_metal_get_(sz_metal_get_((void *)objc_getClass("MTLCompileOptions"), "alloc"),
                                                "init");
            ((void (*)(void *, SEL, sz_size_t))objc_msgSend)(options, sel_registerName("setLanguageVersion:"),
                                                             language_version);
            void *error = STRINGZILLA_NULL;
            void *const library = ((void *(*)(void *, SEL, void *, void *, void **))objc_msgSend)(
                context->device, sel_registerName("newLibraryWithSource:options:error:"),
                ((void *(*)(void *, SEL, void *))objc_msgSend)(sz_metal_string_(sz_metal_prelude_source_),
                                                               sel_registerName("stringByAppendingString:"),
                                                               sz_metal_string_(source)),
                options, &error);
            sz_metal_do_(options, "release");
            if (library) {
                libraries[slot].source = source;
                libraries[slot].library = library;
                ++context->libraries_count;
            }
        }
    }
    if (slot < context->libraries_count) {
        void *const function = ((void *(*)(void *, SEL, void *))objc_msgSend)(
            context->libraries[slot].library, sel_registerName("newFunctionWithName:"), sz_metal_string_(name));
        void *error = STRINGZILLA_NULL;
        if (function) {
            pipeline = ((void *(*)(void *, SEL, void *, void **))objc_msgSend)(
                context->device, sel_registerName("newComputePipelineStateWithFunction:error:"), function, &error);
            sz_metal_do_(function, "release");
        }
    }
    objc_autoreleasePoolPop(pool);
    if (pipeline) {
        sz_metal_pipeline_t *const entry = context->pipelines + context->pipelines_count++;
        entry->name = name, entry->library = context->libraries[slot].library, entry->pipeline = pipeline;
    }
    os_unfair_lock_unlock(&context->lock);
    return pipeline;
}

/** Gives each threadgroup @p encoder dispatches next @p bytes of threadgroup memory at @p index. */
STRINGZILLA_INLINE void sz_metal_threadgroup_memory_(void *encoder, sz_size_t bytes, sz_size_t index) {
    ((void (*)(void *, SEL, sz_size_t, sz_size_t))objc_msgSend)(
        encoder, sel_registerName("setThreadgroupMemoryLength:atIndex:"), bytes, index);
}

/**
 *  @brief Binds @p buffers from index zero and @p arguments right after them, then encodes
 *      @p pipeline over @p groups threadgroups of @p threads, leaving the call open.
 *
 *  A compute encoder runs its dispatches serially, each seeing the writes of those before it, so a
 *  multi-pass call encodes every pass here and pays for one command buffer. An unbound entry of
 *  @p buffers binds nothing, for a kernel that never reads it.
 */
STRINGZILLA_INLINE void sz_metal_enqueue_(void *encoder, void *pipeline, sz_metal_bound_t const *buffers,
                                          sz_size_t buffers_count, void const *arguments, sz_size_t arguments_bytes,
                                          sz_metal_size_t groups, sz_metal_size_t threads) {
    for (sz_size_t index = 0; index != buffers_count; ++index)
        ((void (*)(void *, SEL, void *, sz_size_t, sz_size_t))objc_msgSend)(
            encoder, sel_registerName("setBuffer:offset:atIndex:"), buffers[index].buffer, buffers[index].offset,
            index);
    ((void (*)(void *, SEL, void const *, sz_size_t, sz_size_t))objc_msgSend)(
        encoder, sel_registerName("setBytes:length:atIndex:"), arguments, arguments_bytes, buffers_count);
    ((void (*)(void *, SEL, void *))objc_msgSend)(encoder, sel_registerName("setComputePipelineState:"), pipeline);
    ((void (*)(void *, SEL, sz_metal_size_t, sz_metal_size_t))objc_msgSend)(
        encoder, sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"), groups, threads);
}

/**
 *  @brief Encodes kernel @p name of @p source into @p call, as @ref sz_metal_enqueue_ does.
 *  @return @c sz_success_k, or @c sz_device_code_mismatch_k for a kernel or encoder that fails.
 */
STRINGZILLA_INLINE sz_status_t sz_metal_encode_(sz_metal_call_t *call, char const *source, char const *name,
                                                sz_metal_bound_t const *buffers, sz_size_t buffers_count,
                                                void const *arguments, sz_size_t arguments_bytes,
                                                sz_metal_size_t groups, sz_metal_size_t threads) {
    void *const pipeline = sz_metal_pipeline_(call->context, source, name);
    void *const encoder = pipeline ? call->encoder : STRINGZILLA_NULL;
    if (!encoder) return sz_device_code_mismatch_k;
    sz_metal_enqueue_(encoder, pipeline, buffers, buffers_count, arguments, arguments_bytes, groups, threads);
    return sz_success_k;
}

/**
 *  @brief Closes @p call: commits what it encoded when @p status is a success, so it starts running
 *      while the host moves on, and drops it otherwise.
 *  @return @p status, or @c sz_bad_alloc_k when the commit cannot be tracked for a synchronization.
 */
STRINGZILLA_INLINE sz_status_t sz_metal_commit_(sz_metal_call_t *call, sz_status_t status) {
    if (call->encoder) sz_metal_do_(call->encoder, "endEncoding"), sz_metal_do_(call->encoder, "release");
    if (call->commands) {
        sz_metal_context_t *const context = call->context;
        sz_metal_pending_t *pending = STRINGZILLA_NULL;
        os_unfair_lock_lock(&context->lock);
        if (status == sz_success_k) pending = sz_metal_pending_reserve_(context, call->queue);
        if (pending) {
            sz_metal_do_(call->commands, "commit");
            pending->commands[pending->commands_count++] = call->commands;
        }
        os_unfair_lock_unlock(&context->lock);
        if (!pending) sz_metal_do_(call->commands, "release");
        if (!pending && status == sz_success_k) status = sz_bad_alloc_k;
    }
    if (call->queue) sz_metal_do_(call->queue, "release");
    call->queue = STRINGZILLA_NULL, call->commands = STRINGZILLA_NULL, call->encoder = STRINGZILLA_NULL;
    return status;
}

/**
 *  @brief Binds @p sequence for a kernel, which reads it as a tape, as Metal has no device function
 *      pointers to call cheaply.
 *  @return @c sz_success_k, @c sz_device_memory_mismatch_k for a tape outside the registry, or
 *      @c sz_device_code_mismatch_k for accessors answering anything but the tape they sit on.
 *  @pre @p sequence is not empty.
 */
STRINGZILLA_INLINE sz_status_t sz_metal_tape_call_(sz_metal_call_t *call, sz_sequence_t const *sequence,
                                                   sz_metal_bound_t *bound) {
    if (!sequence->count || sequence->count > ((sz_size_t)-1) / sizeof(sz_u64_t) - 1)
        return sz_device_memory_mismatch_k;
    sz_u64_t const *const offsets = (sz_u64_t const *)sequence->handle;
    sz_cptr_t const tape = (sz_cptr_t)sequence->handle;
    sz_size_t const last = sequence->count - 1;
    // Retain the allocation before reading its offsets.
    if (!sz_metal_resolve_call_(call, offsets, (sequence->count + 1) * sizeof(sz_u64_t), bound) ||
        !sz_metal_resolve_call_(call, offsets, offsets[sequence->count], bound))
        return sz_device_memory_mismatch_k;
    if (!sequence->get_start || !sequence->get_length || offsets[0] < (sequence->count + 1) * sizeof(sz_u64_t) ||
        offsets[0] > offsets[1] || offsets[1] > offsets[sequence->count] || offsets[last] > offsets[sequence->count])
        return sz_device_memory_mismatch_k;
    // Accessor addresses differ between translation units in header-only builds.
    if (sequence->get_start(tape, 0) != tape + offsets[0] || sequence->get_length(tape, 0) != offsets[1] - offsets[0] ||
        sequence->get_start(tape, last) != tape + offsets[last] ||
        sequence->get_length(tape, last) != offsets[last + 1] - offsets[last])
        return sz_device_code_mismatch_k;
    return sz_success_k;
}

#pragma endregion Context

#pragma region Public API

/** Initializes @p allocator to hand back shared buffers on the device of each call's stream, which
 *  both the host and device address. Encoded work retains buffers until completion. The allocator
 *  is stateless, so its handle is null. @return @c sz_success_k. */
STRINGZILLA_API sz_status_t sz_allocator_init_unified_metal(sz_allocator_t *allocator);

/** @copydoc sz_sequence_realloc_best */
STRINGZILLA_API sz_status_t sz_sequence_realloc_metal(sz_sequence_t *target, sz_sequence_t const *source,
                                                      sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                      sz_stream_t stream);

/**
 *  @brief Waits for every call committed to @p stream before the synchronization begins.
 *  @return The first failure: @c sz_device_code_mismatch_k when a command buffer finished in error,
 *      or @c sz_missing_gpu_k without a device.
 */
STRINGZILLA_API sz_status_t sz_stream_synchronize_metal(sz_stream_t stream);

#pragma endregion Public API

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_allocator_init_unified_metal(sz_allocator_t *allocator) {
    return sz_allocator_init_unified_metal_(allocator);
}

STRINGZILLA_API sz_status_t sz_sequence_realloc_metal(sz_sequence_t *target, sz_sequence_t const *source,
                                                      sz_allocator_t *allocator, sz_size_t *allocated_bytes,
                                                      sz_stream_t stream) {
    return sz_sequence_realloc_metal_(target, source, allocator, allocated_bytes, stream);
}

STRINGZILLA_API sz_status_t sz_stream_synchronize_metal(sz_stream_t stream) {
    return sz_stream_synchronize_metal_(stream);
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_WITH_METAL
#endif // STRINGZILLA_METAL_H_
