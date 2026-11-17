/**
 *  @file test/parallel.cpp
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Checks that @c sz_parallel_for_tiles runs every tile once at any thread count,
 *      and that the status of a failing tile comes back.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <atomic> // `std::atomic_ref`
#include <vector> // `std::vector`

#include "harness.hpp"
#include "parallel.h" // `sz_parallel_for_tiles`

namespace ashvardanian::stringzilla::test {

struct tiles_t {
    std::vector<unsigned> runs;
    sz_size_t failing_tile = static_cast<sz_size_t>(-1);
};

sz_status_t count_tile_(sz_size_t tile_index, void *context) {
    tiles_t &tiles = *static_cast<tiles_t *>(context);
    std::atomic_ref<unsigned>(tiles.runs[tile_index]).fetch_add(1, std::memory_order_relaxed);
    return tile_index == tiles.failing_tile ? sz_bad_alloc_k : sz_success_k;
}

/** Thread counts to run, the largest past any core count, so the scheduler multiplexes them. */
constexpr sz_size_t thread_counts_k[] = {1, 4, 64, 4096};
constexpr std::size_t tiles_count_k = 10000;

/** Every tile runs exactly once, and the call reports success, at every thread count. */
void test_parallel_every_tile_once() {
    for (sz_size_t threads : thread_counts_k) {
        tiles_t tiles;
        tiles.runs.assign(tiles_count_k, 0);
        sz_status_t const status = sz_parallel_for_tiles(tiles.runs.size(), threads, count_tile_, &tiles);
        std::size_t once = 0;
        for (unsigned runs : tiles.runs) once += runs == 1;
        expect_equal(status, sz_success_k, "{} threads", threads);
        expect_equal(once, tiles.runs.size(), "{} threads", threads);
    }
}

/** The status of a failing tile comes back from the call, at every thread count. */
void test_parallel_failing_tile() {
    for (sz_size_t threads : thread_counts_k) {
        tiles_t tiles;
        tiles.runs.assign(tiles_count_k, 0);
        tiles.failing_tile = 777;
        expect_equal(sz_parallel_for_tiles(tiles.runs.size(), threads, count_tile_, &tiles), sz_bad_alloc_k,
                     "{} threads", threads);
    }
}

} // namespace ashvardanian::stringzilla::test

int main(int argc, char **argv) {
    using namespace ashvardanian::stringzilla::test;
    start_test_run(argc, argv);
    run_test_case("test_parallel_every_tile_once", test_parallel_every_tile_once);
    run_test_case("test_parallel_failing_tile", test_parallel_failing_tile);
    return finish_test_run();
}
