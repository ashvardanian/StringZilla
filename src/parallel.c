/**
 *  @file src/parallel.c
 *  @author Ash Vardanian
 *  @date September 30, 2026
 *  @brief Tile-parallel execution for StringZilla language bindings.
 *
 *  One shared tile counter drives every backend, so dynamic scheduling and the thread-count cap
 *  behave identically no matter which pool runs the tiles.
 */
#include "parallel.h"

#if STRINGZILLA_OS_APPLE_
#include <dispatch/dispatch.h> // `dispatch_apply_f`, part of libSystem
#include <unistd.h>            // `sysconf`
#elif STRINGZILLA_OS_WINDOWS_
#include <winsock2.h>
#include <windows.h> // `CreateThreadpoolWork`, part of kernel32
#else
#include <unistd.h> // `sysconf`
#if defined(_OPENMP)
#include <omp.h>
#else
#include <pthread.h> // `pthread_create`, `pthread_join`
#include <stdlib.h>  // `malloc`, `free`
#endif
#endif

/** OpenMP schedules tiles itself, so the counter below is for the other pools. */
#if !STRINGZILLA_OS_APPLE_ && !STRINGZILLA_OS_WINDOWS_ && defined(_OPENMP)
#define STRINGZILLA_PARALLEL_VIA_OPENMP 1
#else
#define STRINGZILLA_PARALLEL_VIA_OPENMP 0
#endif

#if !STRINGZILLA_PARALLEL_VIA_OPENMP

#pragma region Tile Queue

/** Tiles remaining, the body that consumes them, and the first failure. */
typedef struct sz_tile_queue_t {
    sz_tile_body_t body;
    void *context;
    sz_size_t tile_count;
#if defined(_MSC_VER)
    volatile long long next_tile;
    volatile long failure;
#else
    sz_size_t next_tile;
    sz_status_t failure;
#endif
} sz_tile_queue_t;

/** Claim and run tiles until the queue is empty or some tile failed, keeping the first failure.
 *  Safe to call from every worker at once. */
static void sz_tile_queue_drain_(sz_tile_queue_t *queue) {
    for (;;) {
#if defined(_MSC_VER)
        if (queue->failure != sz_success_k) return;
        sz_size_t const tile_index = (sz_size_t)_InterlockedExchangeAdd64(&queue->next_tile, 1);
#else
        if (__atomic_load_n(&queue->failure, __ATOMIC_RELAXED) != sz_success_k) return;
        sz_size_t const tile_index = __atomic_fetch_add(&queue->next_tile, 1, __ATOMIC_RELAXED);
#endif
        if (tile_index >= queue->tile_count) return;
        sz_status_t const status = queue->body(tile_index, queue->context);
        if (status == sz_success_k) continue;
#if defined(_MSC_VER)
        _InterlockedCompareExchange(&queue->failure, (long)status, (long)sz_success_k);
#else
        sz_status_t expected = sz_success_k;
        __atomic_compare_exchange_n(&queue->failure, &expected, status, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
#endif
        return;
    }
}

#pragma endregion Tile Queue

#endif // !STRINGZILLA_PARALLEL_VIA_OPENMP

#pragma region Platform Pools

#if STRINGZILLA_OS_APPLE_

/** libdispatch hands each worker an index it does not need; the queue picks its tiles. */
static void sz_tile_worker_dispatch_(void *context, size_t worker_index) {
    sz_unused_(worker_index);
    sz_tile_queue_drain_((sz_tile_queue_t *)context);
}

#elif STRINGZILLA_OS_WINDOWS_

static void CALLBACK sz_tile_worker_threadpool_(PTP_CALLBACK_INSTANCE instance, PVOID context, PTP_WORK work) {
    sz_unused_(instance), sz_unused_(work);
    sz_tile_queue_drain_((sz_tile_queue_t *)context);
}

#elif !STRINGZILLA_PARALLEL_VIA_OPENMP

static void *sz_tile_worker_posix_(void *queue) {
    sz_tile_queue_drain_((sz_tile_queue_t *)queue);
    return NULL;
}

#endif

sz_size_t sz_parallel_concurrency(void) {
#if STRINGZILLA_OS_WINDOWS_
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

    // A single worker needs no pool, no atomics, and no launch latency.
    if (threads <= 1) {
        for (sz_size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
            sz_status_t const status = body(tile_index, context);
            if (status != sz_success_k) return status;
        }
        return sz_success_k;
    }

#if STRINGZILLA_PARALLEL_VIA_OPENMP
    // OpenMP's own dynamic queue beats draining a shared counter.
    sz_status_t failure = sz_success_k;
    long long const tile_limit = (long long)tile_count;
#pragma omp parallel for schedule(dynamic, 1) num_threads((int)threads)
    for (long long tile_index = 0; tile_index < tile_limit; ++tile_index) {
        if (__atomic_load_n(&failure, __ATOMIC_RELAXED) != sz_success_k) continue;
        sz_status_t const status = body((sz_size_t)tile_index, context);
        sz_status_t expected = sz_success_k;
        if (status != sz_success_k)
            __atomic_compare_exchange_n(&failure, &expected, status, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    }
    return failure;
#else
    sz_tile_queue_t queue;
    queue.body = body;
    queue.context = context;
    queue.tile_count = tile_count;
    queue.next_tile = 0;
    queue.failure = sz_success_k;

#if STRINGZILLA_OS_APPLE_
    dispatch_apply_f((size_t)threads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &queue,
                     sz_tile_worker_dispatch_);
#elif STRINGZILLA_OS_WINDOWS_
    PTP_WORK work = CreateThreadpoolWork(sz_tile_worker_threadpool_, &queue, NULL);
    if (work) {
        // The calling thread takes a share too, so only the extra workers are submitted.
        for (sz_size_t worker = 1; worker < threads; ++worker) SubmitThreadpoolWork(work);
        sz_tile_queue_drain_(&queue);
        WaitForThreadpoolWorkCallbacks(work, FALSE);
        CloseThreadpoolWork(work);
    }
    else sz_tile_queue_drain_(&queue); // Out of pool resources — the caller still needs the result
#else
    // The calling thread takes a share too, so only the extra workers get threads of their own.
    // Any that fail to start, or all when the handles get no memory, leave their tiles to the rest.
    pthread_t *workers = (pthread_t *)malloc((threads - 1) * sizeof(pthread_t));
    sz_size_t started = 0;
    if (workers)
        while (started + 1 < threads && pthread_create(&workers[started], NULL, sz_tile_worker_posix_, &queue) == 0)
            ++started;
    sz_tile_queue_drain_(&queue);
    for (sz_size_t worker = 0; worker != started; ++worker) pthread_join(workers[worker], NULL);
    free(workers);
#endif
    return (sz_status_t)queue.failure;
#endif // STRINGZILLA_PARALLEL_VIA_OPENMP
}

#pragma endregion Platform Pools
