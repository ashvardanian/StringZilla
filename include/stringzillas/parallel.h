/**
 *  @file include/stringzillas/parallel.h
 *  @brief Tiny fork_union bridge so the StringZillas entries parallelize their per-pair loop.
 *  @author Ash Vardanian
 *
 *  The megakernel entries iterate over the input collection one pair (or one text) at a time.
 *  This header lifts that loop behind `szs_parallel_for_`: given a device-scope POD it either runs the
 *  per-index body on the calling thread (serial scope / NULL pool) or fans it across the fork_union
 *  pool via `fu_pool_for_n`. Results are identical either way — each index is independent and writes a
 *  distinct, stride-addressed output slot. The body is supplied as a C callback + context, mirroring
 *  the `executor.for_n` shape of the original C++ engines.
 */
#ifndef STRINGZILLAS_PARALLEL_H_
#define STRINGZILLAS_PARALLEL_H_

#include "stringzillas/types.h" // `szs_device_scope_impl_t`
#include "fork_union.h"         // `fu_pool_for_n`, `fu_pool_t`

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Per-index work body. @p context carries the entry's loop-invariant state. */
typedef sz_status_t (*szs_parallel_body_t)(void *context, sz_size_t index);

/** @brief Shared context the fork_union callback threads to the per-index body. */
typedef struct szs_parallel_ctx_t {
    szs_parallel_body_t body; ///< Per-index work body.
    void *context;            ///< Entry-supplied state.
    sz_status_t status;       ///< First non-success status observed (sticky), default success.
} szs_parallel_ctx_t;

/** @brief fork_union prong trampoline: runs one index, latching the first failure. */
static void szs_parallel_prong_(void *ctx_punned, size_t task, size_t thread, size_t colocation) {
    sz_unused_(thread);
    sz_unused_(colocation);
    szs_parallel_ctx_t *ctx = (szs_parallel_ctx_t *)ctx_punned;
    sz_status_t status = ctx->body(ctx->context, (sz_size_t)task);
    // Latch the first failure without a lock: benign last-writer-wins race, all failures are terminal.
    if (status != sz_success_k) ctx->status = status;
}

/**
 *  @brief Run @p body for every index in `[0, count)`, in parallel if @p scope carries a CPU pool.
 *  @param scope The device-scope POD behind the public scope handle (may be NULL => calling thread).
 *  @return The first non-success status from any index, or `sz_success_k`.
 */
static inline sz_status_t szs_parallel_for_(szs_device_scope_impl_t *scope, sz_size_t count, szs_parallel_body_t body,
                                            void *context) {
    fu_pool_t *pool = (scope && scope->kind != szs_scope_gpu_k) ? (fu_pool_t *)scope->cpu.pool : (fu_pool_t *)0;
    if (!pool) {
        for (sz_size_t i = 0; i < count; ++i) {
            sz_status_t status = body(context, i);
            if (status != sz_success_k) return status;
        }
        return sz_success_k;
    }
    szs_parallel_ctx_t ctx;
    ctx.body = body;
    ctx.context = context;
    ctx.status = sz_success_k;
    fu_pool_for_n(pool, (size_t)count, &szs_parallel_prong_, &ctx);
    return ctx.status;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STRINGZILLAS_PARALLEL_H_
