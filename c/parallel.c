/**
 *  @file c/parallel.c
 *  @author Ash Vardanian
 *  @date September 30, 2026
 *  @brief Tile-parallel execution for StringZilla language bindings.
 *
 *  One shared tile counter drives every backend, so dynamic scheduling and the thread-count cap
 *  behave identically no matter which pool runs the tiles.
 */
#include "parallel.h"

#if defined(__APPLE__)
#include <dispatch/dispatch.h> // `dispatch_apply_f`, part of libSystem
#include <unistd.h>            // `sysconf`
#elif defined(_WIN32)
#include <windows.h> // `CreateThreadpoolWork`, part of kernel32
#else
#include <unistd.h> // `sysconf`
#if defined(_OPENMP)
#include <omp.h>
#endif
#endif

/** OpenMP schedules tiles itself, so the counter below is for the other pools. */
#if !defined(__APPLE__) && !defined(_WIN32) && defined(_OPENMP)
#define STRINGZILLA_PARALLEL_VIA_OPENMP 1
#else
#define STRINGZILLA_PARALLEL_VIA_OPENMP 0
#endif

#if !STRINGZILLA_PARALLEL_VIA_OPENMP

#pragma region Tile Queue

/** Tiles remaining, the body that consumes them, and one status slot per worker. */
typedef struct sz_tile_queue_t {
    sz_tile_body_t body;
    void *context;
    sz_size_t tile_count;
    sz_status_t *statuses;
#if defined(_MSC_VER)
    volatile long long next_tile;
#else
    sz_size_t next_tile;
#endif
#if defined(_WIN32)
    volatile long long next_worker;
#endif
} sz_tile_queue_t;

/** Claim and run tiles until the queue is empty or one fails, whose status only this worker's slot
 *  receives. Safe to call from every worker at once, each with its own @p worker. */
static void sz_tile_queue_drain_(sz_tile_queue_t *queue, sz_size_t worker) {
    for (;;) {
#if defined(_MSC_VER)
        sz_size_t const tile_index = (sz_size_t)_InterlockedExchangeAdd64(&queue->next_tile, 1);
#else
        sz_size_t const tile_index = __atomic_fetch_add(&queue->next_tile, 1, __ATOMIC_RELAXED);
#endif
        if (tile_index >= queue->tile_count) return;
        sz_status_t const status = queue->body(tile_index, queue->context);
        if (status != sz_success_k) {
            queue->statuses[worker] = status;
            return;
        }
    }
}

#pragma endregion Tile Queue

#endif // !STRINGZILLA_PARALLEL_VIA_OPENMP

#pragma region Platform Pools

#if defined(__APPLE__)

/** libdispatch hands each worker its index, also its status slot; the queue picks its tiles. */
static void sz_tile_worker_dispatch_(void *context, size_t worker_index) {
    sz_tile_queue_drain_((sz_tile_queue_t *)context, (sz_size_t)worker_index);
}

#elif defined(_WIN32)

/** The pool passes no index, so each submitted worker claims its status slot from a counter. */
static void CALLBACK sz_tile_worker_threadpool_(PTP_CALLBACK_INSTANCE instance, PVOID context, PTP_WORK work) {
    sz_unused_(instance), sz_unused_(work);
    sz_tile_queue_t *queue = (sz_tile_queue_t *)context;
#if defined(_MSC_VER)
    sz_size_t const worker = (sz_size_t)_InterlockedIncrement64(&queue->next_worker);
#else
    sz_size_t const worker = (sz_size_t)__atomic_add_fetch(&queue->next_worker, 1, __ATOMIC_RELAXED);
#endif
    sz_tile_queue_drain_(queue, worker);
}

#endif

sz_size_t sz_parallel_concurrency(void) {
#if defined(_WIN32)
    DWORD const count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return count ? (sz_size_t)count : 1;
#elif defined(_SC_NPROCESSORS_ONLN)
    long const count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? (sz_size_t)count : 1;
#else
    return 1;
#endif
}

sz_status_t sz_parallel_for_tiles(sz_size_t tile_count, sz_size_t threads, sz_tile_body_t body, void *context) {
    if (!tile_count) return sz_success_k;
    if (threads == 0) threads = sz_parallel_concurrency();
    if (threads > tile_count) threads = tile_count;
    if (threads > STRINGZILLA_PARALLEL_MAX_THREADS) threads = STRINGZILLA_PARALLEL_MAX_THREADS;

    // A single worker needs no pool, no atomics, and no launch latency.
    if (threads <= 1) {
        for (sz_size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
            sz_status_t const status = body(tile_index, context);
            if (status != sz_success_k) return status;
        }
        return sz_success_k;
    }

    sz_status_t statuses[STRINGZILLA_PARALLEL_MAX_THREADS];
    for (sz_size_t worker = 0; worker < threads; ++worker) statuses[worker] = sz_success_k;

#if STRINGZILLA_PARALLEL_VIA_OPENMP
    // OpenMP's own dynamic queue beats draining a shared counter.
    long long const tile_limit = (long long)tile_count;
#pragma omp parallel for schedule(dynamic, 1) num_threads((int)threads)
    for (long long tile_index = 0; tile_index < tile_limit; ++tile_index) {
        sz_status_t *slot = &statuses[omp_get_thread_num()];
        if (*slot != sz_success_k) continue;
        sz_status_t const status = body((sz_size_t)tile_index, context);
        if (status != sz_success_k) *slot = status;
    }
#else
    sz_tile_queue_t queue;
    queue.body = body;
    queue.context = context;
    queue.tile_count = tile_count;
    queue.statuses = statuses;
    queue.next_tile = 0;

#if defined(__APPLE__)
    dispatch_apply_f((size_t)threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &queue,
                     sz_tile_worker_dispatch_);
#elif defined(_WIN32)
    queue.next_worker = 0;
    PTP_WORK work = CreateThreadpoolWork(sz_tile_worker_threadpool_, &queue, NULL);
    if (!work) {
        sz_tile_queue_drain_(&queue, 0); // Out of pool resources — the caller still needs the result
        return statuses[0];
    }
    // The calling thread takes a share too, as worker 0, so only the extra workers are submitted.
    for (sz_size_t worker = 1; worker < threads; ++worker) SubmitThreadpoolWork(work);
    sz_tile_queue_drain_(&queue, 0);
    WaitForThreadpoolWorkCallbacks(work, FALSE);
    CloseThreadpoolWork(work);
#else
    sz_tile_queue_drain_(&queue, 0);
#endif
#endif // STRINGZILLA_PARALLEL_VIA_OPENMP

    for (sz_size_t worker = 0; worker < threads; ++worker)
        if (statuses[worker] != sz_success_k) return statuses[worker];
    return sz_success_k;
}

#pragma endregion Platform Pools
