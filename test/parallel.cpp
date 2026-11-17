/**
 *  @file test/parallel.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Checks that @c sz_parallel_for_tiles runs every tile once at any thread count,
 *      and that the status of a failing tile comes back.
 */
#include <cstdio>  // `std::fprintf`
#include <cstdlib> // `EXIT_SUCCESS`, `EXIT_FAILURE`

#include <atomic> // `std::atomic_ref`
#include <vector> // `std::vector`

#include "parallel.h" // `sz_parallel_for_tiles`

namespace ashvardanian::stringzilla::test {

struct tiles_t {
    std::vector<unsigned> runs;
    sz_size_t failing_tile = static_cast<sz_size_t>(-1);
};

sz_status_t count_tile(sz_size_t tile_index, void *context) {
    tiles_t &tiles = *static_cast<tiles_t *>(context);
    std::atomic_ref<unsigned>(tiles.runs[tile_index]).fetch_add(1, std::memory_order_relaxed);
    return tile_index == tiles.failing_tile ? sz_bad_alloc_k : sz_success_k;
}

} // namespace ashvardanian::stringzilla::test

int main() {
    using namespace ashvardanian::stringzilla::test;
    int failures = 0;
    // 4096 threads exceeds any core count here, so the scheduler multiplexes the threads.
    for (sz_size_t threads : {sz_size_t {1}, sz_size_t {4}, sz_size_t {64}, sz_size_t {4096}}) {
        tiles_t tiles;
        tiles.runs.assign(10000, 0);
        sz_status_t const status = sz_parallel_for_tiles(tiles.runs.size(), threads, count_tile, &tiles);
        sz_size_t once = 0;
        for (unsigned runs : tiles.runs) once += runs == 1;
        if (status != sz_success_k || once != tiles.runs.size()) {
            std::fprintf(stderr, "%zu threads: status %d, %zu of %zu tiles ran once\n", static_cast<size_t>(threads),
                         static_cast<int>(status), static_cast<size_t>(once), tiles.runs.size());
            ++failures;
        }

        tiles.runs.assign(10000, 0);
        tiles.failing_tile = 777;
        if (sz_parallel_for_tiles(tiles.runs.size(), threads, count_tile, &tiles) != sz_bad_alloc_k) {
            std::fprintf(stderr, "%zu threads: the failing tile's status was lost\n", static_cast<size_t>(threads));
            ++failures;
        }
    }
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
