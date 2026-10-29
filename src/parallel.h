/**
 *  @file c/parallel.h
 *  @author Ash Vardanian
 *  @date September 30, 2026
 *  @brief Tile-parallel execution for StringZilla language bindings.
 *
 *  Compiled into the Python and Node extensions, never into the libraries.
 *  Each platform's own pool runs the tiles, so no binding ships a @c libomp.
 */
#ifndef STRINGZILLA_PARALLEL_H_
#define STRINGZILLA_PARALLEL_H_

#include <stringzilla/types.h> // `sz_size_t`, `sz_status_t`

#ifdef __cplusplus
extern "C" {
#endif

/** Most workers one call runs, as each keeps its first failure in a slot on the caller's stack. */
#define STRINGZILLA_PARALLEL_MAX_THREADS 1024

/**
 *  @brief Work for one tile. Must be reentrant and confine writes to its own tile.
 *  @param[in] tile_index Zero-based index below the @c tile_count passed to the scheduler.
 *  @param[in] context Caller-owned state, shared unsynchronized across all tiles.
 *  @return @c sz_success_k, or a failure that stops this worker from claiming more tiles.
 */
typedef sz_status_t (*sz_tile_body_t)(sz_size_t tile_index, void *context);

/** Logical processors available to this process, or 1 when undetectable. */
sz_size_t sz_parallel_concurrency(void);

/**
 *  @brief Run @p tile_count independent tiles, at most @p threads of them at a time.
 *
 *  Tiles are claimed from a shared counter, giving the dynamic scheduling uneven workloads need.
 *  @p threads of 0 means every logical processor, up to @c STRINGZILLA_PARALLEL_MAX_THREADS, and 1
 *  runs the tiles inline on the calling thread with no pool involved. Returns once every worker
 *  stopped, at its first failure or when no tile is left. The caller must release the GIL if the
 *  bodies do not need it.
 *
 *  @return @c sz_success_k, or the failure of the lowest-numbered worker that had one.
 */
sz_status_t sz_parallel_for_tiles(sz_size_t tile_count, sz_size_t threads, sz_tile_body_t body, void *context);

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_PARALLEL_H_
