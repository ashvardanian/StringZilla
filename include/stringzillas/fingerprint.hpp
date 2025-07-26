/**
 *  @brief  Hardware-accelerated Min-Hash fingerprinting for string collections.
 *  @file   fingerprint.hpp
 *  @author Ash Vardanian
 *
 *  The `sklearn.feature_extraction` module for @b TF-IDF, `CountVectorizer`, and @b `HashingVectorizer`
 *  is one of the most commonly used in the industry due to its extreme flexibility. It can:
 *
 *  - Tokenize by words, N-grams, or in-word N-grams.
 *  - Use arbitrary Regular Expressions as word separators.
 *  - Return matrices of different types, normalized or not.
 *  - Exclude "stop words" and remove ASCII and Unicode accents.
 *  - Dynamically build a vocabulary or use a fixed list/dictionary.
 *
 *  @see https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
 *  @see https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
 *
 *  That level of flexibility is not feasible for a hardware-accelerated SIMD library, but we can provide a
 *  subset of that functionality for producing fixed-size "sketches" or "fingerprints" of documents for large-scale
 *  retrieval tasks. We must also keep in mind, that however costly, the "fingerprinting" is a one-time operation, and
 *  the quality of the resulting "sketch" is no less important than the speed of the algorithm.
 *
 *  @section Polynomial @b Rolling Hashes
 *
 *  At its core we compute many Karp-Rabin-like "rolling hashes" over multiple window widths and multipliers.
 *  We avoid 64-bit hashes, due to the lack of hardware support for efficient multiplication and modulo operations.
 *  That's especially noticeable on GPUs, where 64-bit ops are often emulated using 32-bit and can be 8-32x slower.
 *  Instead, we use 32-bit hashes, and windows of size 4, 8, 16, and 32 bytes, including up to 8x UTF-32 characters.
 *
 *  @see https://en.wikipedia.org/wiki/MinHash
 *  @see https://en.wikipedia.org/wiki/Universal_hashing
 *
 *  For every byte T(i) we see, the update rule for the hash H(i) is:
 *
 *  1. multiply the hashes by a constant,
 *  2. broadcast the new byte across the register,
 *  3. add broadcasted byte to the hashes,
 *  4. compute the modulo of the hashes with a large prime number.
 *
 *  That logic is packed into 3 functions: @b `push`, @b `roll`, and @b `digest`.
 *  The typical instructions for high-resolution integer multiplication are like are:
 *
 *  - `VPMULLQ (ZMM, ZMM, ZMM)` for `_mm512_mullo_epi64`:
 *    - on Intel Ice Lake: 15 cycles on port 0.
 *    - on AMD Zen4: 3 cycles on ports 0 or 1.
 *  - `VPMULLD (ZMM, ZMM, ZMM)` for `_mm512_mullo_epi32`:
 *    - on Intel Ice Lake: 10 cycles on port 0.
 *    - on AMD Zen4: 3 cycles on ports 0 or 1.
 *  - `VPMULLW (ZMM, ZMM, ZMM)` for `_mm512_mullo_epi16`:
 *    - on Intel Ice Lake: 5 cycles on port 0.
 *    - on AMD Zen4: 3 cycles on ports 0 or 1.
 *  - `VPMADD52LUQ (ZMM, ZMM, ZMM)` for `_mm512_madd52lo_epu64` for 52-bit multiplication:
 *    - on Intel Ice Lake: 4 cycles on port 0.
 *    - on AMD Zen4: 4 cycles on ports 0 or 1.
 *
 *  Such multiplication is typically much more expensive than smaller integer types, and one may expect
 *  more such SIMD instructions appearing due to the AI demand for quantized dot-products, but currently
 *  they don't seem much cheaper:
 *
 *  - `VPDPWSSDS (ZMM, ZMM, ZMM)` for `_mm512_dpwssds_epi32` for 16-bit signed FMA into 32-bit:
 *    - on Intel Ice Lake: 5 cycles on port 0.
 *    - on AMD Zen4: 4 cycles on ports 0 or 1.
 *
 *  An alternative may be to switch to floating-point arithmetic:
 *
 *  - `VFMADD132PS (ZMM, ZMM, ZMM)` for `_mm512_fmadd_ps` for 32-bit FMA:
 *    - on Intel Ice Lake: 4 cycles on port 0.
 *    - on AMD Zen4: 4 cycles on ports 0 or 1.
 *  - `VFMADD132PD (ZMM, ZMM, ZMM)` for `_mm512_fmadd_pd` for 64-bit FMA:
 *    - on Intel Ice Lake: 4 cycles on port 0.
 *    - on AMD Zen4: 4 cycles on ports 0 or 1.
 *
 *  The significand of a `double` can store at least 52 bits worth of unique values, and the latencies of
 *  the `VFMADD132PD` and `VPMADD52LUQ` seem identical, which suggests that under the hood, those instructions
 *  may be using the same machinery. Importantly, floating-point division is still expensive:
 *
 *  - `VDIVPS (ZMM, ZMM, ZMM)` for `_mm512_div_ps` for 32-bit division:
 *    - on Intel Ice Lake: 17 cycles on port 0.
 *    - on AMD Zen4: 11 cycles on ports 0 or 1.
 *  - `VDIVPD (ZMM, ZMM, ZMM)` for `_mm512_div_pd` for 64-bit division:
 *    - on Intel Ice Lake: 23 cycles on port 0.
 *    - on AMD Zen4: 13 cycles on ports 0 or 1.
 *
 *  So optimizations, like the Barrett reduction can still be useful.
 *
 *  Choosing the right "window width" is task- and domain-dependant. For example, most English words are
 *  between 3 and 7 characters long, so a window of 4 bytes would be a good choice. For DNA sequences,
 *  the "window width" might be a multiple of 3, as the codons are 3 (nucleotides) bytes long.
 *  With such minimalistic alphabets of just four characters (AGCT) longer windows might be needed.
 *  For protein sequences the alphabet is 20 characters long, so the window can be shorter, than for DNAs.
 *
 *  @section Fingerprinting, @b Min-Hashing, or @b Count-Min-Sketching?
 *
 *  Computing one such hash won't help us much in large-scale retrieval tasks, but there is a common technique
 *  called "Min-Hashing" that can. The idea is to apply many hash functions for different slices of the input,
 *  and then output the minimum of each hash function as an individual dimension of a resulting vector.
 *
 *  Picking the right number of dimensions is task-dependant. The longer and more diverse are the input strings,
 *  the more dimensions may be needed to capture their uniqueness. The shorter and more similar the strings,
 *  the fewer dimensions are needed. A good starting point is to use roughly the same amount of memory as the size
 *  of input documents. So if you are processing 4 KB memory pages, I'd recommend 1024 dimensions, each encoded
 *  as a 32-bit integer, which is 4 KB in total.
 *
 *  From the hardware perspective, however, on both CPUs and GPUs we vectorize the code. Hash functions that have
 *  the same window width can be processed simultaneously without complex memory access patterns. Assuming, the state
 *  of each rolling hash is 8 bytes:
 *
 *  - on AVX-512 capable CPUs, take at least 8 hash-functions of each width,
 *  - on AVX-512 capable CPUs with a physical 512-bit path, take 16 or more, to increase register utilization,
 *  - on Nvidia GPUs, take at least 32 hash-functions of each width, to activate all 32 threads in a warp.
 *  - on AMD GPUs, take at least 64 hash-functions of each width, to activate all 64 threads in a wave.
 */
#ifndef STRINGZILLAS_FINGERPRINT_HPP_
#define STRINGZILLAS_FINGERPRINT_HPP_

#include "stringzilla/types.hpp"  // `sz::error_cost_t`
#include "stringzilla/memory.h"   // `sz_move`
#include "stringzillas/types.hpp" // `sz::executor_like`

#include <cstddef>
#include <limits>   // `std::numeric_limits` for numeric types
#include <iterator> // `std::iterator_traits` for iterators
#include <cmath>    // `std::fabsf` for `f32_rolling_hasher`
#include <numeric>  // `std::gcd` for `choose_coprime_modulo`

namespace ashvardanian {
namespace stringzillas {

#pragma region - Baseline Rolling Hashers

/**
 *  @brief The simplest example of a rolling hash function, leveraging 2^N modulo arithmetic.
 *  @tparam hash_type_ Type of the hash value, e.g., `std::uint64_t`.
 */
template <typename hash_type_ = std::uint64_t>
struct multiplying_rolling_hasher {
    using state_t = hash_type_;
    using hash_t = hash_type_;

    explicit multiplying_rolling_hasher(std::size_t window_width, hash_t multiplier = static_cast<hash_t>(257)) noexcept
        : window_width_ {window_width}, multiplier_ {multiplier}, highest_power_ {1} {

        sz_assert_(window_width_ > 1 && "Window width must be > 1");
        sz_assert_(multiplier_ > 0 && "Multiplier must be positive");

        for (std::size_t i = 0; i + 1 < window_width_; ++i) highest_power_ = highest_power_ * multiplier_;
    }

    inline std::size_t window_width() const noexcept { return window_width_; }

    inline state_t push(state_t state, byte_t new_char) const noexcept { return state * multiplier_ + new_char; }

    inline state_t roll(state_t state, byte_t old_char, byte_t new_char) const noexcept {
        state_t const without_head = state - old_char * highest_power_;
        return without_head * multiplier_ + new_char;
    }

    inline hash_t digest(state_t const state) const noexcept { return state; }

  private:
    std::size_t window_width_;
    state_t multiplier_;
    state_t highest_power_;
};

/**
 *  @brief Rabin-Karp–style rolling polynomial hash function.
 *  @tparam hash_type_ Type of the hash value, can be `u16`, `u32`, or `u64`.
 *  @tparam accumulator_type_ Type used for modulo arithmetic, e.g., `std::uint64_t`.
 *
 *  Barrett's reduction can be used to avoid overflow in the multiplication and modulo operations.
 *  That, however, is quite tricky and computationally expensive, so this algorithm is provided merely
 *  as a baseline for retrieval benchmarks.
 *  @sa `multiplying_rolling_hasher`
 */
template <typename hash_type_ = std::uint32_t, typename accumulator_type_ = std::uint64_t>
struct rabin_karp_rolling_hasher {
    using hash_t = hash_type_;
    using state_t = accumulator_type_;

    static_assert(is_same_type<hash_t, std::uint16_t>::value || is_same_type<hash_t, std::uint32_t>::value ||
                      is_same_type<hash_t, std::uint64_t>::value,
                  "Unsupported hash type");

    static constexpr hash_t default_alphabet_size_k = 256u;
    static constexpr hash_t default_modulo_base_k = //
        is_same_type<hash_t, std::uint16_t>::value   ? SZ_U16_MAX_PRIME
        : is_same_type<hash_t, std::uint32_t>::value ? SZ_U32_MAX_PRIME
                                                     : SZ_U64_MAX_PRIME;

    explicit rabin_karp_rolling_hasher(              //
        std::size_t window_width,                    //
        hash_t multiplier = default_alphabet_size_k, //
        hash_t modulo = default_modulo_base_k) noexcept
        : window_width_ {window_width}, modulo_ {modulo}, multiplier_ {multiplier}, discarding_multiplier_ {1} {

        sz_assert_(window_width_ > 1 && "Window width must be > 1");
        sz_assert_(multiplier_ > 0 && "Multiplier must be positive");
        sz_assert_(modulo_ > 1 && "Modulo base must be > 1");

        for (std::size_t i = 0; i + 1 < window_width_; ++i)
            discarding_multiplier_ = mul_mod(discarding_multiplier_, multiplier_);
    }

    inline std::size_t window_width() const noexcept { return window_width_; }

    inline state_t push(state_t state, byte_t new_char) const noexcept {
        state_t new_term = static_cast<state_t>(new_char + 1u);
        return add_mod(mul_mod(state, multiplier_), new_term);
    }

    inline state_t roll(state_t state, byte_t old_char, byte_t new_char) const noexcept {
        state_t old_term = static_cast<state_t>(old_char + 1u);
        state_t new_term = static_cast<state_t>(new_char + 1u);

        state_t without_old = sub_mod(state, mul_mod(old_term, discarding_multiplier_));
        state_t with_new = add_mod(mul_mod(without_old, multiplier_), new_term);
        return with_new;
    }

    inline hash_t digest(state_t state) const noexcept { return static_cast<hash_t>(state); }

  private:
    inline state_t mul_mod(state_t a, state_t b) const noexcept { return (a * b) % modulo_; }
    inline state_t add_mod(state_t a, state_t b) const noexcept { return (a + b) % modulo_; }
    inline state_t sub_mod(state_t a, state_t b) const noexcept { return (a + modulo_ - b) % modulo_; }

    std::size_t window_width_;
    state_t modulo_;
    state_t multiplier_;
    state_t discarding_multiplier_;
};

/**
 *  @brief BuzHash rolling hash function leveraging a fixed-size lookup table and bitwise operations.
 *  @tparam hash_type_ Type of the hash value, e.g., `std::uint64_t`.
 *  @sa `multiplying_rolling_hasher`, `rabin_karp_rolling_hasher`
 */
template <typename hash_type_ = std::uint64_t>
struct buz_rolling_hasher {
    using state_t = hash_type_;
    using hash_t = hash_type_;

    explicit buz_rolling_hasher(std::size_t window_width, std::uint64_t seed = 0x9E3779B97F4A7C15ull) noexcept
        : window_width_ {window_width} {

        sz_assert_(window_width_ > 1 && "Window width must be > 1");
        for (std::size_t i = 0; i < 256; ++i) table_[i] = split_mix64(seed);
    }

    inline std::size_t window_width() const noexcept { return window_width_; }

    inline state_t push(state_t state, byte_t new_char) const noexcept {
        return rotl(state, 1) ^ table_[new_char & 0xFFu];
    }

    inline state_t roll(state_t state, byte_t old_char, byte_t new_char) const noexcept {
        constexpr unsigned bits_k = sizeof(state_t) * 8u;
        state_t const rolled = rotl(state, 1);
        state_t const remove_term = rotl(table_[old_char & 0xFFu], window_width_ & (bits_k - 1u));
        return rolled ^ remove_term ^ table_[new_char & 0xFFu];
    }

    inline hash_t digest(state_t state) const noexcept { return state; }

  private:
    static inline state_t rotl(state_t v, unsigned r) noexcept {
        constexpr unsigned bits_k = sizeof(state_t) * 8u;
        return (v << r) | (v >> (bits_k - r));
    }

    static inline std::uint64_t split_mix64(std::uint64_t &state) noexcept {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    std::size_t window_width_;
    state_t table_[256];
};

/**
 *  @brief Helper function to pick the second co-prime "modulo" base for the Karp-Rabin rolling hashes.
 *  @retval 0 on failure, or a valid prime number otherwise.
 */
inline std::uint64_t choose_coprime_modulo(std::uint64_t multiplier, std::uint64_t limit) noexcept {
    if (multiplier == 0 || multiplier >= limit || limit <= 1) return 0;

    // Upper bound guaranteeing no overflow in non-discarding `update` calls
    std::uint64_t max_input = std::numeric_limits<byte_t>::max() + 1u;
    std::uint64_t bound = (limit - (max_input + 1)) / multiplier + 1;

    if (!(bound & 1u)) --bound; // Make odd

    for (std::uint64_t p = bound; p >= 3; p -= 2)
        if (std::gcd(p, multiplier) == 1) return p;

    return 0;
}

template <typename state_type_ = float>
struct floating_rolling_hasher;

/**
 *  @brief Rabin-Karp-style Rolling hash function for single-precision floating-point numbers.
 *  @tparam state_type_ Type of the floating-point number, e.g., `float`.
 *
 *  The IEEE 754 single-precision `float` has a 24-bit significand (23 explicit bits + 1 implicit bit).
 *  For simplicity, we just focus on the 23-bit part, which is capable of exactly representing integers
 *  up to (2²³ - 1) = (8'388'607), available in @b `limit_k`.
 *
 *  Some of the large primes fitting right before that limit are:
 *      8'388'539, 8'388'547, 8'388'571, 8'388'581, 8'388'587, 8'388'593.
 *
 *  Assuming the multipliers are typically within @b [256;~1000) and the additive factor is always within @b [1;257],
 *  a safer choice of modulo is the largest prime under `limit_k/1000-257`:
 *
 *     8'089, 8'093, 8'101, 8'111, 8'117, 8'123
 *
 *  ! Notice how small those modulo values are, so there's going to be very little information encoded in hashes.
 *  ! So the `floating_rolling_hasher<float>` should only be used for exploratory purposes & testing.
 *
 *  @sa `floating_rolling_hasher<double>` for 52 bit variant.
 */
template <>
struct floating_rolling_hasher<float> {
    using state_t = float;
    using hash_t = std::uint32_t;

    /** @brief The largest integer exactly representable as a float. */
    static constexpr state_t limit_k = 8'388'607.0f;

    /** @brief The typical size of the alphabet - the 256 possible values of a single byte. */
    static constexpr hash_t default_alphabet_size_k = 256u;

    /** @brief The largest prime, that multiplied by `default_alphabet_size_k` and added a term - stays `limit_k`. */
    static constexpr hash_t default_modulo_base_k = 8123u;

    explicit floating_rolling_hasher(                      //
        std::size_t const window_width,                    //
        hash_t const multiplier = default_alphabet_size_k, //
        hash_t const modulo = default_modulo_base_k) noexcept
        : window_width_ {window_width}, multiplier_ {static_cast<state_t>(multiplier)},
          modulo_ {static_cast<state_t>(modulo)}, inverse_modulo_ {1.0f / modulo_},
          negative_discarding_multiplier_ {1.0f} {

        sz_assert_(window_width_ > 1 && "Window width must be > 1");
        sz_assert_(multiplier_ > 0 && "Multiplier must be positive");
        sz_assert_(modulo_ > 1 && "Modulo must be > 1");

        // If we want to avoid hitting +inf or NaN, we need to make sure that the product of our post-modulo
        // normalized number with the multiplier and added subsequent term stays within the exactly representable range.
        state_t const largest_input_term = std::numeric_limits<byte_t>::max() + 1.0f;
        state_t const largest_normalized_state = modulo_ - 1;
        state_t const largest_intermediary = largest_normalized_state * multiplier_ + largest_input_term;
        sz_assert_(largest_intermediary < limit_k && "Intermediate state overflows the limit");

        // ! The GCC header misses the `std::fmodf` overload, so we use the underlying C version
        for (std::size_t i = 0; i + 1 < window_width_; ++i)
            negative_discarding_multiplier_ = ::fmodf(negative_discarding_multiplier_ * multiplier_, modulo_);
        negative_discarding_multiplier_ = -negative_discarding_multiplier_;
    }

    inline std::size_t window_width() const noexcept { return window_width_; }

    inline state_t push(state_t state, byte_t new_char) const noexcept {
        state_t new_term = state_t(new_char) + 1.0f;
        return fma_mod(state, multiplier_, new_term);
    }

    inline state_t roll(state_t state, byte_t old_char, byte_t new_char) const noexcept {

        state_t old_term = state_t(old_char) + 1.0f;
        state_t new_term = state_t(new_char) + 1.0f;

        state_t without_old = fma_mod(negative_discarding_multiplier_, old_term, state);
        return fma_mod(without_old, multiplier_, new_term);
    }

    inline hash_t digest(state_t state) const noexcept { return static_cast<hash_t>(state); }

  private:
    inline state_t fma_mod(state_t a, state_t b, state_t c) const noexcept { return barrett_mod(a * b + c); }

    /**
     *  @brief Barrett-style `std::fmodf` alternative to avoid overflow.
     *  @see https://en.cppreference.com/w/cpp/numeric/math/fmod
     */
    inline state_t barrett_mod(state_t x) const noexcept {

        state_t q = std::floor(x * inverse_modulo_);
        state_t result = x - q * modulo_;

        // Clamp into the [0, modulo_) range.
        if (result >= modulo_) result -= modulo_;
        if (result < 0.0f) result += modulo_;

        sz_assert_(result >= 0 && "Intermediate x underflows the zero");
        sz_assert_(result < limit_k && "Intermediate x overflows the limit");
        sz_assert_(static_cast<std::uint64_t>(::fmodf(x, modulo_) + (::fmodf(x, modulo_) < 0.0f ? modulo_ : 0.0f)) ==
                       static_cast<std::uint64_t>(result) &&
                   "Floating point modulo was incorrect");

        return result;
    }

    std::size_t window_width_;
    state_t multiplier_;
    state_t modulo_;
    state_t inverse_modulo_;
    state_t negative_discarding_multiplier_;
};

inline f64_t absolute_fmod(f64_t x, f64_t y) noexcept {
    f64_t result = std::fmod(x, y);
    return result < 0.0 ? result + y : result;
}

inline u64_t absolute_umod(f64_t x, f64_t y) noexcept { return static_cast<u64_t>(absolute_fmod(x, y)); }

/**
 *  @brief Rabin-Karp-style Rolling hash function for double-precision floating-point numbers.
 *  @tparam state_type_ Type of the floating-point number, e.g., `float`.
 *
 *  The IEEE 754 double-precision `float` has a 53-bit significand (52 explicit bits + 1 implicit bit).
 *  For simplicity, we just focus on the 52-bit part, which is capable of exactly representing integers
 *  up to (2⁵² - 1) = (4'503'599'627'370'495), available in @b `limit_k`.
 *
 *  Some of the large primes fitting right before that limit are:
 *      4'503'599'627'370'287, 4'503'599'627'370'299, 4'503'599'627'370'313,
 *      4'503'599'627'370'323, 4'503'599'627'370'353, 4'503'599'627'370'449.
 *
 *  Assuming the multipliers are typically within @b [256;~1000) and the additive factor is always within @b [1;257],
 *  a safer choice of modulo is the largest prime under `limit_k/1000-257`:
 *      4'503'599'626'781, 4'503'599'626'783, 4'503'599'626'807,
 *      4'503'599'626'907, 4'503'599'626'957, 4'503'599'626'977.
 *
 *  @sa `rabin_karp_rolling_hasher<std::uint32_t, std::uint64_t>` integer implementation for small modulo variants.
 *  @sa `floating_rolling_hasher<float>` for a lower-resolution hash.
 */
template <>
struct floating_rolling_hasher<double> {
    using state_t = double;
    using hash_t = std::uint64_t;

    static constexpr state_t limit_k = 4503599627370495.0;
    static constexpr hash_t default_alphabet_size_k = 256u;
    static constexpr hash_t default_modulo_base_k = 4503599626977u;

    explicit floating_rolling_hasher(                       //
        std::size_t const window_width,                     //
        state_t const multiplier = default_alphabet_size_k, //
        state_t const modulo = default_modulo_base_k) noexcept
        : window_width_ {window_width}, multiplier_ {static_cast<state_t>(multiplier)},
          modulo_ {static_cast<state_t>(modulo)}, inverse_modulo_ {1.0 / modulo_},
          negative_discarding_multiplier_ {1.0} {

        sz_assert_(window_width_ > 1 && "Window width must be > 1");
        sz_assert_(multiplier_ > 0 && "Multiplier must be positive");
        sz_assert_(modulo_ > 1 && "Modulo must be > 1");

        // If we want to avoid hitting +inf or NaN, we need to make sure that the product of our post-modulo
        // normalized number with the multiplier and added subsequent term stays within the exactly representable range.
        state_t const largest_input_term = std::numeric_limits<byte_t>::max() + 1.0;
        state_t const largest_normalized_state = modulo_ - 1;
        state_t const largest_intermediary = largest_normalized_state * multiplier_ + largest_input_term;
        sz_assert_(largest_intermediary < limit_k && "Intermediate state overflows the limit");

        for (std::size_t i = 0; i + 1 < window_width_; ++i)
            negative_discarding_multiplier_ = std::fmod(negative_discarding_multiplier_ * multiplier_, modulo_);
        negative_discarding_multiplier_ = -negative_discarding_multiplier_;
    }

    inline std::size_t window_width() const noexcept { return window_width_; }

    inline state_t push(state_t state, byte_t new_char) const noexcept {
        state_t new_term = state_t(new_char) + 1.0;
        return fma_mod(state, multiplier_, new_term);
    }

    inline state_t roll(state_t state, byte_t old_char, byte_t new_char) const noexcept {
        state_t old_term = state_t(old_char) + 1.0;
        state_t new_term = state_t(new_char) + 1.0;
        state_t without_old = fma_mod(negative_discarding_multiplier_, old_term, state);
        return fma_mod(without_old, multiplier_, new_term);
    }

    inline hash_t digest(state_t state) const noexcept { return static_cast<hash_t>(state); }

    inline state_t multiplier() const noexcept { return multiplier_; }
    inline state_t modulo() const noexcept { return modulo_; }
    inline state_t inverse_modulo() const noexcept { return inverse_modulo_; }
    inline state_t negative_discarding_multiplier() const noexcept { return negative_discarding_multiplier_; }

  private:
    inline state_t fma_mod(state_t a, state_t b, state_t c) const noexcept { return barrett_mod(a * b + c); }

    /**
     *  @brief Barrett-style `std::fmod` alternative to avoid overflow.
     *  @see https://en.cppreference.com/w/cpp/numeric/math/fmod
     */
    inline state_t barrett_mod(state_t x) const noexcept {

        state_t q = std::floor(x * inverse_modulo_);
        state_t result = x - q * modulo_;

        // Clamp into the [0, modulo_) range.
        if (result >= modulo_) result -= modulo_;
        if (result < 0.0) result += modulo_;

        sz_assert_(result >= 0 && "Intermediate x underflows the zero");
        sz_assert_(result < limit_k && "Intermediate x overflows the limit");
        sz_assert_(static_cast<std::uint64_t>(absolute_fmod(x, modulo_)) == static_cast<std::uint64_t>(result) &&
                   "Floating point modulo was incorrect");
        return result;
    }

    std::size_t window_width_;
    state_t multiplier_;
    state_t modulo_;
    state_t inverse_modulo_;
    state_t negative_discarding_multiplier_;
};

#pragma endregion - Baseline Rolling Hashers

#pragma region - Optimized Rolling MinHashers

template <size_t dimensions_ = SZ_SIZE_MAX, typename hash_type_ = std::uint32_t, typename count_type_ = std::uint32_t>
void merge_count_min_sketches(                                                                           //
    span<hash_type_ const, dimensions_> a_min_hashes, span<count_type_ const, dimensions_> a_min_counts, //
    span<hash_type_ const, dimensions_> b_min_hashes, span<count_type_ const, dimensions_> b_min_counts, //
    span<hash_type_, dimensions_> c_min_hashes, span<count_type_, dimensions_> c_min_counts) noexcept {

    sz_assert_(a_min_hashes.size() == b_min_hashes.size() && "Input sketches must have the same size");
    sz_assert_(a_min_counts.size() == b_min_counts.size() && "Input counts must have the same size");
    sz_assert_(c_min_hashes.size() == a_min_hashes.size() && "Output hashes must have the same size");
    sz_assert_(c_min_counts.size() == a_min_counts.size() && "Output counts must have the same size");

    for (std::size_t dim = 0; dim < c_min_hashes.size(); ++dim) {
        if (a_min_hashes[dim] < b_min_hashes[dim]) {
            c_min_hashes[dim] = a_min_hashes[dim];
            c_min_counts[dim] = a_min_counts[dim];
        }
        else if (b_min_hashes[dim] < a_min_hashes[dim]) {
            c_min_hashes[dim] = b_min_hashes[dim];
            c_min_counts[dim] = b_min_counts[dim];
        }
        else {
            c_min_hashes[dim] = a_min_hashes[dim];
            c_min_counts[dim] = a_min_counts[dim] + b_min_counts[dim];
        }
    }
}

/**
 *  @brief Boring Min-Hash / Count-Min-Sketch implementation over any rolling hashing algorithm just for benchmarking.
 *  @tparam hasher_type_ Can be the Rabin-Karp, BuzHash, or anything else compatible.
 */
template <                                                                           //
    typename hasher_type_ = rabin_karp_rolling_hasher<std::uint32_t, std::uint64_t>, //
    typename min_hash_type_ = std::uint32_t,                                         //
    typename min_count_type_ = std::uint32_t,                                        //
    typename allocator_type_ = std::allocator<hasher_type_>                          //
    >
struct basic_rolling_hashers {

    using hasher_t = hasher_type_;
    using rolling_state_t = typename hasher_t::state_t;
    using rolling_hash_t = typename hasher_t::hash_t;

    using min_hash_t = min_hash_type_;
    using min_count_t = min_count_type_;
    using allocator_t = allocator_type_;

    static constexpr rolling_state_t skipped_rolling_state_k = std::numeric_limits<rolling_state_t>::max();
    static constexpr rolling_hash_t skipped_rolling_hash_k = std::numeric_limits<rolling_hash_t>::max();
    static constexpr min_hash_t max_hash_k = std::numeric_limits<min_hash_t>::max();

  private:
    using allocator_traits_t = std::allocator_traits<allocator_type_>;
    using hasher_allocator_t = typename allocator_traits_t::template rebind_alloc<hasher_t>;
    using rolling_states_allocator_t = typename allocator_traits_t::template rebind_alloc<rolling_state_t>;
    using rolling_hashes_allocator_t = typename allocator_traits_t::template rebind_alloc<rolling_hash_t>;
    using min_counts_allocator_t = typename allocator_traits_t::template rebind_alloc<min_count_t>;

    allocator_t allocator_;
    safe_vector<hasher_t, hasher_allocator_t> hashers_;
    std::size_t max_window_width_ = 0;

  public:
    basic_rolling_hashers(allocator_t allocator = {}) noexcept
        : allocator_(std::move(allocator)),
          hashers_(allocator_traits_t::select_on_container_copy_construction(allocator)) {}

    std::size_t max_window_width() const noexcept { return max_window_width_; }
    std::size_t dimensions() const noexcept { return hashers_.size(); }

    /**
     *  @brief Appends multiple new rolling hashers for a given @p window_width.
     *
     *  @param[in] window_width Width of the rolling window, typically 3, 4, 5, 6, or 7.
     *  @param[in] dims Number of hash functions to use, typically 768, 1024, or 1536.
     *  @param[in] alphabet_size Size of the alphabet, typically 256 for UTF-8, 4 for DNA, or 20 for proteins.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     *
     *  Typical usage of this interface (error handling aside) would be like:
     *
     *  @code{.cpp}
     *  basic_rolling_hashers<rabin_karp_rolling_hasher<std::uint32_t>> hashers;
     *  hashers.try_extend(3, 32); // 32 dims for 3-grams
     *  hashers.try_extend(5, 32); // 32 dims for 5-grams
     *  hashers.try_extend(7, 64); // 64 dims for 7-grams
     *  std::array<std::uint32_t, 128> fingerprint; // 128 total dims
     *  hashers("some text", fingerprint);
     *  @endcode
     */
    status_t try_extend(std::size_t window_width, std::size_t dims, std::size_t alphabet_size = 256) noexcept {
        if (hashers_.try_reserve(dims) != status_t::success_k) return status_t::bad_alloc_k;
        for (std::size_t dim = 0; dim < dims; ++dim) {
            status_t status = try_append(hasher_t(window_width, alphabet_size + dim));
            sz_assert_(status == status_t::success_k && "Couldn't fail after the reserve");
        }
        return status_t::success_k;
    }

    /**
     *  @brief Appends a new rolling @p hasher to the collection via `try_append`.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     */
    status_t try_append(hasher_t hasher) noexcept {
        auto const new_window_width = hasher.window_width();
        if (hashers_.try_push_back(std::move(hasher)) != status_t::success_k) return status_t::bad_alloc_k;

        max_window_width_ = (std::max)(new_window_width, max_window_width_);
        return status_t::success_k;
    }

    /**
     *  @brief Computes the fingerprint of a single @p `text` on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     */
    template <size_t dimensions_ = SZ_SIZE_MAX>
    status_t try_fingerprint(                     //
        span<byte_t const> text,                  //
        span<min_hash_t, dimensions_> min_hashes, //
        span<min_count_t, dimensions_> min_counts) const noexcept {

        sz_assert_(dimensions() == min_hashes.size() && "Dimensions number & hashers number mismatch");
        sz_assert_(dimensions() == min_counts.size() && "Dimensions number & hash-counts number mismatch");

        // Allocate temporary states
        safe_vector<rolling_state_t, rolling_states_allocator_t> rolling_states_buffer(
            allocator_traits_t::select_on_container_copy_construction(allocator_));
        safe_vector<rolling_hash_t, rolling_hashes_allocator_t> rolling_minimums_buffer(
            allocator_traits_t::select_on_container_copy_construction(allocator_));
        if (rolling_states_buffer.try_resize(dimensions()) != status_t::success_k ||
            rolling_minimums_buffer.try_resize(dimensions()) != status_t::success_k)
            return status_t::bad_alloc_k;

        // Initialize the starting states
        for (auto &state : rolling_states_buffer) state = rolling_state_t(0);
        for (auto &minimum : rolling_minimums_buffer) minimum = skipped_rolling_hash_k;

        // Roll through the entire `text`
        auto rolling_states =
            span<rolling_state_t, dimensions_>(rolling_states_buffer.data(), rolling_states_buffer.size());
        auto rolling_minimums =
            span<rolling_hash_t, dimensions_>(rolling_minimums_buffer.data(), rolling_minimums_buffer.size());
        fingerprint_chunk<dimensions_>(text, rolling_states, rolling_minimums, min_hashes, min_counts);
        return status_t::success_k;
    }

    /**
     *  @brief Underlying machinery of `fingerprint` that fills the states of the hashers.
     *  @param[in] text_chunk A chunk of text to update the @p `last_states` with.
     *  @param[inout] last_states The last computed floats for each hasher; start with @b zeroes.
     *  @param[inout] rolling_minimums The minimum floats for each hasher; start with @b `skipped_rolling_hash_k`.
     *  @param[out] min_hashes The @b optional output for minimum hashes, which are the final fingerprints.
     *  @param[out] min_counts The frequencies of @p `rolling_minimums` and optional @p `min_hashes` hashes.
     *  @param[in] passed_progress The offset of the received @p `text_chunk` in the whole text; defaults to 0.
     *
     *  Unlike the `fingerprint` method, this function can be used in a @b rolling fashion, i.e., it can be called
     *  multiple times with different chunks of text, and it will update the states accordingly. In the end, it
     *  will anyways export the composing Count-Min-Sketch fingerprint into the @p `min_hashes` and @p `min_counts`,
     *  as its a relatively cheap operation.
     */
    template <size_t dimensions_ = SZ_SIZE_MAX>
    void fingerprint_chunk(                                 //
        span<byte_t const> text_chunk,                      //
        span<rolling_state_t, dimensions_> last_states,     //
        span<rolling_hash_t, dimensions_> rolling_minimums, //
        span<min_hash_t, dimensions_> min_hashes,           //
        span<min_count_t, dimensions_> min_counts,          //
        std::size_t const passed_progress = 0) const noexcept {

        sz_assert_(dimensions() == last_states.size() && "Dimensions number & states number mismatch");
        sz_assert_(dimensions() == rolling_minimums.size() && "Dimensions number & minimums number mismatch");
        sz_assert_(dimensions() == min_hashes.size() && "Dimensions number & min-hashes number mismatch");
        sz_assert_(dimensions() == min_counts.size() && "Dimensions number & hash-counts number mismatch");

        // Until we reach the maximum window length, use a branching code version
        std::size_t const prefix_length = (std::min)(text_chunk.size(), max_window_width_);
        std::size_t new_char_offset = passed_progress;
        for (; new_char_offset < prefix_length; ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            for (std::size_t dim = 0; dim < last_states.size(); ++dim) {
                auto &hasher = hashers_[dim];
                rolling_state_t &last_state = last_states[dim];
                rolling_hash_t &rolling_minimum = rolling_minimums[dim];
                min_count_t &min_count = min_counts[dim];
                if (hasher.window_width() > new_char_offset) {
                    last_state = hasher.push(last_state, new_char);
                    if (hasher.window_width() == (new_char_offset + 1)) {
                        rolling_minimum = (std::min)(rolling_minimum, hasher.digest(last_state));
                        min_count = 1; // First occurrence of this hash
                    }
                }
                else {
                    auto const old_char = text_chunk[new_char_offset - hasher.window_width()];
                    last_state = hasher.roll(last_state, old_char, new_char);
                    rolling_hash_t new_hash = hasher.digest(last_state);
                    min_count *= new_hash >= rolling_minimum; // Discard `min_count` to 0, if a new minimum is found
                    min_count += new_hash == rolling_minimum; // Increments `min_count` by 1 for new & existing minimums
                    rolling_minimum = (std::min)(rolling_minimum, new_hash);
                }
            }
        }

        // Now we can avoid a branch in the nested loop, as we are passed the longest window width
        for (; new_char_offset < text_chunk.size(); ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            for (std::size_t dim = 0; dim < last_states.size(); ++dim) {
                auto &hasher = hashers_[dim];
                rolling_state_t &last_state = last_states[dim];
                rolling_hash_t &rolling_minimum = rolling_minimums[dim];
                min_count_t &min_count = min_counts[dim];
                auto const old_char = text_chunk[new_char_offset - hasher.window_width()];
                last_state = hasher.roll(last_state, old_char, new_char);
                rolling_hash_t new_hash = hasher.digest(last_state);
                min_count *= new_hash >= rolling_minimum; // Discard `min_count` to 0, if a new minimum is found
                min_count += new_hash == rolling_minimum; // Increments `min_count` by 1 for new & existing minimums
                rolling_minimum = (std::min)(rolling_minimum, new_hash);
            }
        }

        // Finally, export the minimum hashes into the smaller representations
        if (min_hashes)
            for (std::size_t dim = 0; dim < min_hashes.size(); ++dim) {
                rolling_hash_t const &rolling_minimum = rolling_minimums[dim];
                min_hash_t &min_hash = min_hashes[dim];
                min_hash = rolling_minimum == skipped_rolling_hash_k
                               ? max_hash_k // If the rolling minimum is not set, use the maximum hash value
                               : static_cast<min_hash_t>(rolling_minimum & max_hash_k);
            }
    }

    /**
     *  @brief Computes many fingerprints in parallel for input @p texts via an @p executor.
     *  @param[in] texts The input texts to hash, typically a sequential container of UTF-8 encoded strings.
     *  @param[out] min_hashes_per_text The output fingerprints, an array of vectors of minimum hashes.
     *  @param[out] min_counts_per_text The output frequencies of @p `min_hashes_per_text` hashes.
     *  @param[in] executor The executor to use for parallel processing, defaults to a dummy executor.
     *  @param[in] specs The CPU specifications to use, defaults to an empty `cpu_specs_t`.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     */
    template <typename texts_type_, typename min_hashes_per_text_type_, typename min_counts_per_text_type_,
              typename executor_type_ = dummy_executor_t>
#if SZ_IS_CPP20_
        requires executor_like<executor_type_>
#endif
    status_t operator()(                                                                                  //
        texts_type_ const &texts,                                                                         //
        min_hashes_per_text_type_ &&min_hashes_per_text, min_counts_per_text_type_ &&min_counts_per_text, //
        executor_type_ &&executor = {}, cpu_specs_t specs = {}) const noexcept {

        // Depending on document sizes, choose the appropriate parallelization strategy
        // - Either split each text into chunks across threads
        // - Or split the texts themselves across threads
        std::size_t const text_size_threshold = executor.threads_count() * specs.l2_bytes;
        std::size_t const dims = dimensions();

        // Allocate enough temporary states for all cores to have individual states
        safe_vector<rolling_state_t, rolling_states_allocator_t> rolling_states(
            allocator_traits_t::select_on_container_copy_construction(allocator_));
        safe_vector<rolling_hash_t, rolling_hashes_allocator_t> rolling_minimums(
            allocator_traits_t::select_on_container_copy_construction(allocator_));
        safe_vector<min_count_t, min_counts_allocator_t> rolling_counts(
            allocator_traits_t::select_on_container_copy_construction(allocator_));
        if (rolling_states.try_resize(executor.threads_count() * dims) != status_t::success_k ||
            rolling_minimums.try_resize(executor.threads_count() * dims) != status_t::success_k ||
            rolling_counts.try_resize(executor.threads_count() * dims) != status_t::success_k)
            return status_t::bad_alloc_k;

        // Process small texts by individual threads
        using executor_t = typename std::decay<executor_type_>::type;
        using prong_t = typename executor_t::prong_t;
        executor.for_n_dynamic(texts.size(), [&](prong_t prong) noexcept {
            auto const text_index = prong.task;
            auto const thread_index = prong.thread;

            auto const &text = texts[text_index];
            if (text.size() >= text_size_threshold) return;

            auto min_hashes = to_span(min_hashes_per_text[text_index]);
            auto min_counts = to_span(min_counts_per_text[text_index]);

            span<byte_t const> text_view = to_bytes_view(text);
            span<rolling_state_t> thread_local_states {rolling_states.data() + thread_index * dims, dims};
            span<rolling_hash_t> thread_local_minimums {rolling_minimums.data() + thread_index * dims, dims};

            // Clear the thread-local buffers & run the rolling fingerprinting API
            for (auto &state : thread_local_states) state = rolling_state_t(0);
            for (auto &minimum : thread_local_minimums) minimum = skipped_rolling_hash_k;
            fingerprint_chunk<SZ_SIZE_MAX>(text_view, thread_local_states, thread_local_minimums, min_hashes,
                                           min_counts);
        });

        // Process large texts by splitting them into chunks
        for (std::size_t text_index = 0; text_index < texts.size(); ++text_index) {

            auto const &text = texts[text_index];
            if (text.size() < text_size_threshold) continue;

            // Split the text into chunks of the maximum window width
            auto const text_view = to_bytes_view(text);
            std::size_t const chunk_size = round_up_to_multiple(             //
                divide_round_up(text_view.size(), executor.threads_count()), //
                specs.cache_line_width);

            // Distribute overlapping chunks across threads
            executor.for_threads([&](std::size_t thread_index) noexcept {
                auto text_start = text_view.data() + (std::min)(text_view.size(), thread_index * chunk_size);
                // ? This overlap will be different for different window widths, but assuming we are
                // ? computing the non-weighted Min-Hash, recomputing & comparing a few hashes for the
                // ? same slices isn't a big deal.
                auto overlapping_text_end =
                    (std::min)(text_start + chunk_size + max_window_width_ - 1, text_view.end());
                auto thread_local_text = span<byte_t const>(text_start, overlapping_text_end);
                auto thread_local_states = span<rolling_state_t> {rolling_states.data() + thread_index * dims, dims};
                auto thread_local_minimums = span<rolling_hash_t> {rolling_minimums.data() + thread_index * dims, dims};
                auto thread_local_counts = span<min_count_t> {rolling_counts.data() + thread_index * dims, dims};

                // Clear the thread-local buffers & run the rolling fingerprinting API
                for (auto &state : thread_local_states) state = rolling_state_t(0);
                for (auto &minimum : thread_local_minimums) minimum = skipped_rolling_hash_k;
                fingerprint_chunk<SZ_SIZE_MAX>(thread_local_text, thread_local_states, thread_local_minimums, {},
                                               thread_local_counts);
            });

            // Compute the minimums of each thread's local states
            auto min_hashes = to_span(min_hashes_per_text[text_index]);
            auto min_counts = to_span(min_counts_per_text[text_index]);
            for (std::size_t dim = 0; dim < min_hashes.size(); ++dim) {
                rolling_hash_t min_hash = skipped_rolling_hash_k;
                min_count_t min_count = 0;
                for (std::size_t thread_index = 0; thread_index < executor.threads_count(); ++thread_index) {
                    rolling_hash_t thread_local_min_hash = rolling_minimums[thread_index * dims + dim];
                    min_count_t thread_local_min_count = rolling_counts[thread_index * dims + dim];
                    if (thread_local_min_hash == min_hash) { min_count += thread_local_min_count; }
                    else if (thread_local_min_hash > min_hash) { continue; }
                    else { min_hash = thread_local_min_hash, min_count = thread_local_min_count; }
                }
                min_hashes[dim] = static_cast<min_hash_t>(min_hash & max_hash_k);
                min_counts[dim] = min_count;
            }
        }

        return status_t::success_k;
    }
};

/**
 *  @brief Computes many fingerprints in parallel for input @p texts, calling @p engine on each thread of @p executor.
 *  @param[in] executor The executor to use for parallel processing, defaults to a dummy executor.
 *  @param[in] texts The input texts to hash, typically a sequential container of UTF-8 encoded strings.
 *  @param[out] min_hashes_per_text The output fingerprints, an array of vectors of minimum hashes.
 *  @param[out] min_counts_per_text The output frequencies of @p `min_hashes_per_text` hashes.
 *  @param[in] executor The executor to use for parallel processing, defaults to a dummy executor.
 *  @param[in] specs The CPU specifications to use, defaults to an empty `cpu_specs_t`.
 *  @retval status_t::success_k on success, or an error code otherwise.
 *  @retval status_t::bad_alloc_k if the memory allocation fails.
 */
template <typename engine_type_, typename texts_type_, typename min_hashes_per_text_type_,
          typename min_counts_per_text_type_,
          typename executor_type_ = dummy_executor_t>
#if SZ_IS_CPP20_
    requires executor_like<executor_type_>
#endif
status_t floating_rolling_hashers_in_parallel_(                                                       //
    engine_type_ const &engine, texts_type_ const &texts,                                             //
    min_hashes_per_text_type_ &&min_hashes_per_text, min_counts_per_text_type_ &&min_counts_per_text, //
    executor_type_ &&executor = {}, cpu_specs_t specs = {}) noexcept {

    using engine_t = engine_type_;
    using rolling_state_t = typename engine_t::rolling_state_t;
    using min_count_t = typename engine_t::min_count_t;
    using min_hash_t = typename engine_t::min_hash_t;
    static constexpr auto window_width_k = engine_t::window_width_k;
    static constexpr auto dimensions_k = engine_t::dimensions_k;
    static constexpr auto skipped_rolling_hash_k = engine_t::skipped_rolling_hash_k;
    static constexpr auto max_hash_k = engine_t::max_hash_k;

    // Depending on document sizes, choose the appropriate parallelization strategy
    // - Either split each text into chunks across threads
    // - Or split the texts themselves across threads
    std::size_t const text_size_threshold = specs.l2_bytes * executor.threads_count();

    // Process small texts by individual threads
    using executor_t = typename std::decay<executor_type_>::type;
    using prong_t = typename executor_t::prong_t;
    executor.for_n_dynamic(texts.size(), [&](prong_t prong) noexcept {
        auto const text_index = prong.task;

        auto const &text = texts[text_index];
        if (text.size() >= text_size_threshold) return;

        auto text_view = to_bytes_view(text);
        auto min_hashes = to_span<dimensions_k>(min_hashes_per_text[text_index]);
        auto min_counts = to_span<dimensions_k>(min_counts_per_text[text_index]);
        engine.fingerprint(text_view, min_hashes, min_counts);
    });

    // Process large texts by splitting them into chunks
    for (std::size_t text_index = 0; text_index < texts.size(); ++text_index) {

        auto const &text = texts[text_index];
        if (text.size() < text_size_threshold) continue;

        // Split the text into chunks of the maximum window width
        auto text_view = to_bytes_view(text);
        std::size_t const chunk_size = round_up_to_multiple(             //
            divide_round_up(text_view.size(), executor.threads_count()), //
            specs.cache_line_width);

        rolling_state_t rolling_minimums[dimensions_k];
        for (std::size_t dim = 0; dim < dimensions_k; ++dim) rolling_minimums[dim] = skipped_rolling_hash_k;

        // Distribute overlapping chunks across threads
        auto min_hashes = to_span(min_hashes_per_text[text_index]);
        auto min_counts = to_span(min_counts_per_text[text_index]);
        auto gather_mutex = executor.make_mutex();
        executor.for_threads([&](std::size_t thread_index) noexcept {
            auto text_start = text_view.data() + (std::min)(text_view.size(), thread_index * chunk_size);
            // ? This overlap will be different for different window widths, but assuming we are
            // ? computing the non-weighted Min-Hash, recomputing & comparing a few hashes for the
            // ? same slices isn't a big deal.
            auto overlapping_text_end = (std::min)(text_start + chunk_size + window_width_k - 1, text_view.end());
            auto thread_local_text = span<byte_t const>(text_start, overlapping_text_end);

            rolling_state_t thread_local_states[dimensions_k];
            rolling_state_t thread_local_minimums[dimensions_k];
            min_count_t thread_local_counts[dimensions_k];
            for (std::size_t dim = 0; dim < dimensions_k; ++dim)
                thread_local_states[dim] = 0, thread_local_minimums[dim] = skipped_rolling_hash_k;
            engine.fingerprint_chunk(thread_local_text, thread_local_states, thread_local_minimums, {},
                                     thread_local_counts);

            lock_guard lock(gather_mutex);
            for (std::size_t dim = 0; dim < dimensions_k; ++dim) {
                rolling_state_t &min_hash = rolling_minimums[dim];
                min_count_t &min_count = min_counts[dim];
                rolling_state_t thread_local_min_hash = thread_local_minimums[dim];
                min_count_t thread_local_min_count = thread_local_counts[dim];
                if (thread_local_min_hash == min_hash) { min_count += thread_local_min_count; }
                else if (thread_local_min_hash > min_hash) { continue; }
                else { min_hash = thread_local_min_hash, min_count = thread_local_min_count; }
            }
        });

        // Digest the smallest hash states, luckily for us, for this hash function,
        // the smallest state corresponds to the smallest digested hash :)
        // This is also never a bottleneck, so let's keep it sequential for simplicity.
        for (std::size_t dim = 0; dim < min_hashes.size(); ++dim) {
            rolling_state_t const &rolling_minimum = rolling_minimums[dim];
            min_hash_t &min_hash = min_hashes[dim];
            auto const rolling_minimum_as_uint = static_cast<std::uint64_t>(rolling_minimum);
            min_hash = rolling_minimum == skipped_rolling_hash_k
                           ? max_hash_k // If the rolling minimum is not set, use the maximum hash value
                           : static_cast<min_hash_t>(rolling_minimum_as_uint & max_hash_k);
        }
    }

    return status_t::success_k;
}

/**
 *  @brief Optimized rolling Min-Hashers via floats, @b constrained to a certain dimensionality & window width.
 *  @note Window width can't be too big to fit on the stack! 16 or 64 is the sweet spot.
 *
 *  This set of CPU kernels is likely to be composed into combinations for different dimensionalities and window
 *  widths, thus covering a subset of the dimensions in a final fingerprint. An example would be, having:
 *  - 32 dimensions for 3-grams,
 *  - 32 dimensions for 5-grams,
 *  - 64 dimensions for 7-grams.
 *
 *  @tparam capability_ The CPU capability, e.g., `sz_cap_serial_k`, `sz_cap_skylake_k`, etc.
 *  @tparam window_width_ The width of the rolling window, e.g., 3, 4, 5, 6, etc.
 *  @tparam dimensions_ The number of dimensions in the fingerprint, recommended a multiple of 16, ideally @b 64.
 *  @tparam enable_ A type used to enable or disable this specialization, e.g., `void` for default.
 */
template <                                         //
    sz_capability_t capability_ = sz_cap_serial_k, //
    std::size_t window_width_ = SZ_SIZE_MAX,       //
    std::size_t dimensions_ = 64,                  //
    typename enable_ = void                        //
    >
struct floating_rolling_hashers {

    using hasher_t = floating_rolling_hasher<double>;
    using rolling_state_t = double;
    using min_hash_t = std::uint32_t;
    using min_count_t = std::uint32_t;

    static constexpr std::size_t window_width_k = window_width_;
    static constexpr std::size_t dimensions_k = dimensions_;
    static constexpr rolling_state_t skipped_rolling_state_k = std::numeric_limits<rolling_state_t>::max();
    static constexpr rolling_state_t skipped_rolling_hash_k = std::numeric_limits<rolling_state_t>::max();
    static constexpr min_hash_t max_hash_k = std::numeric_limits<min_hash_t>::max();

    using min_hashes_span_t = span<min_hash_t, dimensions_k>;
    using min_counts_span_t = span<min_count_t, dimensions_k>;

  private:
    rolling_state_t multipliers_[dimensions_k];
    rolling_state_t modulos_[dimensions_k];
    rolling_state_t inverse_modulos_[dimensions_k];
    rolling_state_t negative_discarding_multipliers_[dimensions_k];

  public:
    constexpr std::size_t window_width() const noexcept { return window_width_k; }
    constexpr std::size_t dimensions() const noexcept { return dimensions_k; }

    /**
     *  @brief Initializes several rolling hashers with different multipliers and modulos.
     *  @param[in] alphabet_size Size of the alphabet, typically 256 for UTF-8, 4 for DNA, or 20 for proteins.
     */
    status_t try_seed(std::size_t alphabet_size = 256) noexcept {
        for (unsigned dim = 0; dim < dimensions_k; ++dim) {
            hasher_t hasher(window_width_k, alphabet_size + dim, hasher_t::default_modulo_base_k);
            multipliers_[dim] = hasher.multiplier();
            modulos_[dim] = hasher.modulo();
            inverse_modulos_[dim] = hasher.inverse_modulo();
            negative_discarding_multipliers_[dim] = hasher.negative_discarding_multiplier();
        }
        return status_t::success_k;
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     */
    void fingerprint(span<byte_t const> text, min_hashes_span_t min_hashes,
                     min_counts_span_t min_counts) const noexcept {

        if (text.size() < window_width_k) {
            for (auto &min_hash : min_hashes) min_hash = max_hash_k;
            for (auto &min_count : min_counts) min_count = 0;
            return;
        }

        rolling_state_t rolling_states[dimensions_k];
        rolling_state_t rolling_minimums[dimensions_k];
        for (std::size_t dim = 0; dim < dimensions_k; ++dim)
            rolling_states[dim] = 0, rolling_minimums[dim] = skipped_rolling_hash_k;
        fingerprint_chunk(text, rolling_states, rolling_minimums, min_hashes, min_counts);
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     */
    status_t try_fingerprint(span<byte_t const> text, min_hashes_span_t min_hashes,
                             min_counts_span_t min_counts) const noexcept {
        fingerprint(text, min_hashes, min_counts);
        return status_t::success_k;
    }

    /**
     *  @brief Underlying machinery of `fingerprint` that fills the states of the hashers.
     *  @param[in] text_chunk A chunk of text to update the @p `last_states` with.
     *  @param[inout] last_states The last computed floats for each hasher; start with @b zeroes.
     *  @param[inout] rolling_minimums The minimum floats for each hasher; start with @b `skipped_rolling_hash_k`.
     *  @param[out] min_hashes The @b optional output for minimum hashes, which are the final fingerprints.
     *  @param[out] min_counts The frequencies of @p `rolling_minimums` and optional @p `min_hashes` hashes.
     *  @param[in] passed_progress The offset of the received @p `text_chunk` in the whole text; defaults to 0.
     *
     *  Unlike the `fingerprint` method, this function can be used in a @b rolling fashion, i.e., it can be called
     *  multiple times with different chunks of text, and it will update the states accordingly. In the end, it
     *  will anyways export the composing Count-Min-Sketch fingerprint into the @p `min_hashes` and @p `min_counts`,
     *  as its a relatively cheap operation.
     */
    void fingerprint_chunk(                                   //
        span<byte_t const> text_chunk,                        //
        span<rolling_state_t, dimensions_k> last_states,      //
        span<rolling_state_t, dimensions_k> rolling_minimums, //
        span<min_hash_t, dimensions_k> min_hashes,            //
        span<min_count_t, dimensions_k> min_counts,           //
        std::size_t const passed_progress = 0) const noexcept {

        // Until we reach the maximum window length, use a branching code version
        std::size_t const prefix_length = (std::min)(text_chunk.size(), window_width_k);
        std::size_t new_char_offset = passed_progress;
        for (; new_char_offset < prefix_length; ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            rolling_state_t const new_term = static_cast<rolling_state_t>(new_char) + 1.0;
            for (std::size_t dim = 0; dim < dimensions_k; ++dim) {
                rolling_state_t &last_state = last_states[dim];
                last_state = std::fma(last_state, multipliers_[dim], new_term); // Add head
                last_state = barrett_mod(last_state, dim);
            }
        }

        // We now have our first minimum hashes
        if (new_char_offset == window_width_k)
            for (std::size_t dim = 0; dim < dimensions_k; ++dim)
                rolling_minimums[dim] = (std::min)(rolling_minimums[dim], last_states[dim]),
                min_counts[dim] = 1; // First occurrence of this hash

        // Now we can avoid a branch in the nested loop, as we are passed the longest window width
        for (; new_char_offset < text_chunk.size(); ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            byte_t const old_char = text_chunk[new_char_offset - window_width_k];
            rolling_state_t const new_term = static_cast<rolling_state_t>(new_char) + 1.0;
            rolling_state_t const old_term = static_cast<rolling_state_t>(old_char) + 1.0;
            for (std::size_t dim = 0; dim < dimensions_k; ++dim) {
                rolling_state_t &last_state = last_states[dim];
                rolling_state_t &rolling_minimum = rolling_minimums[dim];
                min_count_t &min_count = min_counts[dim];

                last_state = std::fma(negative_discarding_multipliers_[dim], old_term, last_state); // Remove tail
                last_state = barrett_mod(last_state, dim);
                last_state = std::fma(last_state, multipliers_[dim], new_term); // Add head
                last_state = barrett_mod(last_state, dim);

                if (rolling_minimum == last_state) { min_count++; }
                else if (last_state < rolling_minimum) { rolling_minimum = last_state, min_count = 1; }
            }
        }

        // Finally, export the minimum hashes into the smaller representations
        if (min_hashes)
            for (std::size_t dim = 0; dim < dimensions_k; ++dim) {
                rolling_state_t const &rolling_minimum = rolling_minimums[dim];
                min_hash_t &min_hash = min_hashes[dim];
                auto const rolling_minimum_as_uint = static_cast<std::uint64_t>(rolling_minimum);
                min_hash = rolling_minimum == skipped_rolling_hash_k
                               ? max_hash_k // If the rolling minimum is not set, use the maximum hash value
                               : static_cast<min_hash_t>(rolling_minimum_as_uint & max_hash_k);
            }
    }

    /**
     *  @brief Computes many fingerprints in parallel for input @p texts via an @p executor.
     *  @param[in] texts The input texts to hash, typically a sequential container of UTF-8 encoded strings.
     *  @param[out] min_hashes_per_text The output fingerprints, an array of vectors of minimum hashes.
     *  @param[out] min_counts_per_text The output frequencies of @p `min_hashes_per_text` hashes.
     *  @param[in] executor The executor to use for parallel processing, defaults to a dummy executor.
     *  @param[in] specs The CPU specifications to use, defaults to an empty `cpu_specs_t`.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     */
    template <typename texts_type_, typename min_hashes_per_text_type_, typename min_counts_per_text_type_,
              typename executor_type_ = dummy_executor_t>
#if SZ_IS_CPP20_
        requires executor_like<executor_type_>
#endif
    status_t operator()(texts_type_ const &texts, min_hashes_per_text_type_ &&min_hashes, //
                        min_counts_per_text_type_ &&min_counts, executor_type_ &&executor = {},
                        cpu_specs_t specs = {}) noexcept {
        return floating_rolling_hashers_in_parallel_(            //
            *this, texts,                                        //
            std::forward<min_hashes_per_text_type_>(min_hashes), //
            std::forward<min_counts_per_text_type_>(min_counts), //
            std::forward<executor_type_>(executor), specs);
    }

  private:
    inline rolling_state_t barrett_mod(rolling_state_t x, std::size_t dim) const noexcept {
        rolling_state_t const modulo = modulos_[dim];
        rolling_state_t const inverse_modulo = inverse_modulos_[dim];

        // Use STL-based modulo reduction like `floating_rolling_hasher`
        rolling_state_t q = std::floor(x * inverse_modulo);
        rolling_state_t result = x - q * modulo;

        // Clamp into the [0, modulo) range.
        result += modulo * (result < 0.0);
        result -= modulo * (result >= modulo);
        return result;
    }
};

/*  AVX512 implementation of the string hashing algorithms for Skylake and newer CPUs.
 *  Includes extensions: F, CD, ER, PF, VL, DQ, BW.
 *
 *  This is the "starting level" for the advanced algorithms using K-mask registers on x86.
 */
#pragma region Skylake Implementation
#if SZ_USE_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512dq", "avx512bw", "bmi", "bmi2")
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512dq,avx512bw,bmi,bmi2"))), \
                             apply_to = function)

/**
 *  @brief Optimized rolling Min-Hashers built around floating-point numbers.
 *
 *  In a single ZMM register we can store 8 `double` values, so we can process 8 hashes per register.
 *  Assuming 32x ZMM registers, and roughly 10ish scalars for intermediaries per hash, we can unroll
 *  2-3x times, and process 16-24 hashes in parallel.
 */
template <std::size_t window_width_, std::size_t dimensions_>
struct floating_rolling_hashers<sz_cap_skylake_k, window_width_, dimensions_> {

    using hasher_t = floating_rolling_hasher<double>;
    using rolling_state_t = double;
    using min_hash_t = std::uint32_t;
    using min_count_t = std::uint32_t;

    static constexpr std::size_t window_width_k = window_width_;
    static constexpr std::size_t dimensions_k = dimensions_;
    static constexpr rolling_state_t skipped_rolling_hash_k = std::numeric_limits<rolling_state_t>::max();
    static constexpr min_hash_t max_hash_k = std::numeric_limits<min_hash_t>::max();

    using min_hashes_span_t = span<min_hash_t, dimensions_k>;
    using min_counts_span_t = span<min_count_t, dimensions_k>;

    static constexpr unsigned unroll_factor_k = 2;
    static constexpr unsigned hashes_per_zmm_k = sizeof(sz_u512_vec_t) / sizeof(rolling_state_t);
    static constexpr unsigned hashes_per_unrolled_group_k = unroll_factor_k * hashes_per_zmm_k;
    static constexpr bool has_incomplete_tail_group_k = dimensions_k % hashes_per_unrolled_group_k;
    static constexpr std::size_t aligned_dimensions_k =
        has_incomplete_tail_group_k ? (dimensions_k / hashes_per_unrolled_group_k + 1) * hashes_per_unrolled_group_k
                                    : (dimensions_k);
    static constexpr unsigned groups_count_k = aligned_dimensions_k / hashes_per_unrolled_group_k;

    static_assert(dimensions_k <= 256, "Too many dimensions to keep on stack");

  private:
    rolling_state_t multipliers_[aligned_dimensions_k];
    rolling_state_t modulos_[aligned_dimensions_k];
    rolling_state_t inverse_modulos_[aligned_dimensions_k];
    rolling_state_t negative_discarding_multipliers_[aligned_dimensions_k];

  public:
    constexpr std::size_t window_width() const noexcept { return window_width_k; }
    constexpr std::size_t dimensions() const noexcept { return dimensions_k; }

    /**
     *  @brief Initializes several rolling hashers with different multipliers and modulos.
     *  @param[in] alphabet_size Size of the alphabet, typically 256 for UTF-8, 4 for DNA, or 20 for proteins.
     */
    status_t try_seed(std::size_t alphabet_size = 256) noexcept {
        for (unsigned dim = 0; dim < dimensions_k; ++dim) {
            hasher_t hasher(window_width_k, alphabet_size + dim, hasher_t::default_modulo_base_k);
            multipliers_[dim] = hasher.multiplier();
            modulos_[dim] = hasher.modulo();
            inverse_modulos_[dim] = hasher.inverse_modulo();
            negative_discarding_multipliers_[dim] = hasher.negative_discarding_multiplier();
        }
        return status_t::success_k;
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     */
    void fingerprint(span<byte_t const> text, min_hashes_span_t min_hashes,
                     min_counts_span_t min_counts) const noexcept {

        if (text.size() < window_width_k) {
            for (auto &min_hash : min_hashes) min_hash = max_hash_k;
            for (auto &min_count : min_counts) min_count = 0;
            return;
        }

        rolling_state_t rolling_states[dimensions_k];
        rolling_state_t rolling_minimums[dimensions_k];
        for (std::size_t dim = 0; dim < dimensions_k; ++dim)
            rolling_states[dim] = 0, rolling_minimums[dim] = skipped_rolling_hash_k;
        fingerprint_chunk(text, &rolling_states[0], &rolling_minimums[0], min_hashes, min_counts);
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     */
    status_t try_fingerprint(span<byte_t const> text, min_hashes_span_t min_hashes,
                             min_counts_span_t min_counts) const noexcept {
        fingerprint(text, min_hashes, min_counts);
        return status_t::success_k;
    }

    /**
     *  @brief Underlying machinery of `fingerprint` that fills the states of the hashers.
     *  @param[in] text_chunk A chunk of text to update the @p `last_states` with.
     *  @param[inout] last_states The last computed floats for each hasher; start with @b zeroes.
     *  @param[inout] rolling_minimums The minimum floats for each hasher; start with @b `skipped_rolling_hash_k`.
     *  @param[out] min_hashes The @b optional output for minimum hashes, which are the final fingerprints.
     *  @param[out] min_counts The frequencies of @p `rolling_minimums` and optional @p `min_hashes` hashes.
     *  @param[in] passed_progress The offset of the received @p `text_chunk` in the whole text; defaults to 0.
     *
     *  Unlike the `fingerprint` method, this function can be used in a @b rolling fashion, i.e., it can be called
     *  multiple times with different chunks of text, and it will update the states accordingly. In the end, it
     *  will anyways export the composing Count-Min-Sketch fingerprint into the @p `min_hashes` and @p `min_counts`,
     *  as its a relatively cheap operation.
     */
    void fingerprint_chunk(                                   //
        span<byte_t const> text_chunk,                        //
        span<rolling_state_t, dimensions_k> last_states,      //
        span<rolling_state_t, dimensions_k> rolling_minimums, //
        span<min_hash_t, dimensions_k> min_hashes,            //
        span<min_count_t, dimensions_k> min_counts,           //
        std::size_t const passed_progress = 0) const noexcept {

        for (unsigned group_index = 0; group_index < groups_count_k; ++group_index)
            roll_group(text_chunk, group_index, last_states, rolling_minimums, min_counts, passed_progress);

        // Finally, export the minimum hashes into the smaller representations
        if (min_hashes)
            for (std::size_t dim = 0; dim < dimensions_k; ++dim) {
                rolling_state_t const &rolling_minimum = rolling_minimums[dim];
                min_hash_t &min_hash = min_hashes[dim];
                auto const rolling_minimum_as_uint = static_cast<std::uint64_t>(rolling_minimum);
                min_hash = rolling_minimum == skipped_rolling_hash_k
                               ? max_hash_k // If the rolling minimum is not set, use the maximum hash value
                               : static_cast<min_hash_t>(rolling_minimum_as_uint & max_hash_k);
            }
    }

    /**
     *  @brief Computes many fingerprints in parallel for input @p texts via an @p executor.
     *  @param[in] texts The input texts to hash, typically a sequential container of UTF-8 encoded strings.
     *  @param[out] min_hashes_per_text The output fingerprints, an array of vectors of minimum hashes.
     *  @param[out] min_counts_per_text The output frequencies of @p `min_hashes_per_text` hashes.
     *  @param[in] executor The executor to use for parallel processing, defaults to a dummy executor.
     *  @param[in] specs The CPU specifications to use, defaults to an empty `cpu_specs_t`.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     */
    template <typename texts_type_, typename min_hashes_per_text_type_, typename min_counts_per_text_type_,
              typename executor_type_ = dummy_executor_t>
#if SZ_IS_CPP20_
        requires executor_like<executor_type_>
#endif
    status_t operator()(texts_type_ const &texts, min_hashes_per_text_type_ &&min_hashes_per_text, //
                        min_counts_per_text_type_ &&min_counts_per_text, executor_type_ &&executor = {},
                        cpu_specs_t specs = {}) noexcept {
        return floating_rolling_hashers_in_parallel_(                     //
            *this, texts,                                                 //
            std::forward<min_hashes_per_text_type_>(min_hashes_per_text), //
            std::forward<min_counts_per_text_type_>(min_counts_per_text), //
            std::forward<executor_type_>(executor), specs);
    }

  private:
    // TODO: We can probably shave a few ore cycles here:
    SZ_INLINE __m512d barrett_mod(__m512d xs, __m512d modulos, __m512d inverse_modulos) const noexcept {
        // Use rounding SIMD arithmetic
        __m512d qs = _mm512_roundscale_pd(      // ! The rounding operation is extremely expensive,
            _mm512_mul_pd(xs, inverse_modulos), // ! so alternatives should be considered.
            _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        __m512d results = _mm512_fnmadd_pd(qs, modulos, xs);

        // Clamp into the [0, modulo) range.
        __mmask8 overflow_mask = _mm512_cmp_pd_mask(results, modulos, _CMP_GE_OQ);
        results = _mm512_mask_sub_pd(results, overflow_mask, results, modulos);
        __mmask8 negative_mask = _mm512_fpclass_pd_mask(results, 0x44); // Negative
        results = _mm512_mask_add_pd(results, negative_mask, results, modulos);

        sz_assert_(modulos[0] == 0 || absolute_umod(xs[0], modulos[0]) == static_cast<u64_t>(results[0]));
        sz_assert_(modulos[1] == 0 || absolute_umod(xs[1], modulos[1]) == static_cast<u64_t>(results[1]));
        sz_assert_(modulos[2] == 0 || absolute_umod(xs[2], modulos[2]) == static_cast<u64_t>(results[2]));
        sz_assert_(modulos[3] == 0 || absolute_umod(xs[3], modulos[3]) == static_cast<u64_t>(results[3]));
        sz_assert_(modulos[4] == 0 || absolute_umod(xs[4], modulos[4]) == static_cast<u64_t>(results[4]));
        sz_assert_(modulos[5] == 0 || absolute_umod(xs[5], modulos[5]) == static_cast<u64_t>(results[5]));
        sz_assert_(modulos[6] == 0 || absolute_umod(xs[6], modulos[6]) == static_cast<u64_t>(results[6]));
        sz_assert_(modulos[7] == 0 || absolute_umod(xs[7], modulos[7]) == static_cast<u64_t>(results[7]));

        return results;
    }

    __attribute__((no_sanitize("address"))) void roll_group(       //
        span<byte_t const> text_chunk, unsigned const group_index, //
        span<rolling_state_t, dimensions_k> last_states,           //
        span<rolling_state_t, dimensions_k> rolling_minimums,      //
        span<min_count_t, dimensions_k> rolling_counts,            //
        std::size_t const passed_progress = 0) const noexcept {

        // Register space for in-out variables
        sz_u512_vec_t last_states_vec[unroll_factor_k];
        sz_u512_vec_t rolling_minimums_vec[unroll_factor_k];
        sz_u256_vec_t rolling_counts_vec[unroll_factor_k];

        // Use masked loads for the incomplete tail group
        if (has_incomplete_tail_group_k && group_index + 1 == groups_count_k)
#pragma unroll(unroll_factor_k)
            for (unsigned index_in_group = 0; index_in_group < unroll_factor_k; ++index_in_group) {
                unsigned const dim = group_index * hashes_per_unrolled_group_k + index_in_group * hashes_per_zmm_k;
                __mmask8 const load_mask = dimensions_k > dim ? sz_u8_mask_until_(dimensions_k - dim) : (__mmask8)0;
                last_states_vec[index_in_group].zmm_pd = _mm512_maskz_loadu_pd(load_mask, &last_states[dim]);
                rolling_minimums_vec[index_in_group].zmm_pd = _mm512_maskz_loadu_pd(load_mask, &rolling_minimums[dim]);
                rolling_counts_vec[index_in_group].ymm = _mm256_maskz_loadu_epi32(load_mask, &rolling_counts[dim]);
            }
        // Otherwise, everything is easy
        else
#pragma unroll(unroll_factor_k)
            for (unsigned index_in_group = 0; index_in_group < unroll_factor_k; ++index_in_group) {
                unsigned const dim = group_index * hashes_per_unrolled_group_k + index_in_group * hashes_per_zmm_k;
                last_states_vec[index_in_group].zmm_pd = _mm512_loadu_pd(&last_states[dim]);
                rolling_minimums_vec[index_in_group].zmm_pd = _mm512_loadu_pd(&rolling_minimums[dim]);
                rolling_counts_vec[index_in_group].ymm =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(&rolling_counts[dim]));
            }

        // Temporary variables for the rolling state
        sz_u512_vec_t multipliers_vec[unroll_factor_k], negative_discarding_multipliers_vec[unroll_factor_k],
            modulos_vec[unroll_factor_k], inverse_modulos_vec[unroll_factor_k];

#pragma unroll(unroll_factor_k)
        for (unsigned index_in_group = 0; index_in_group < unroll_factor_k; ++index_in_group) {
            unsigned const dim = group_index * hashes_per_unrolled_group_k + index_in_group * hashes_per_zmm_k;
            multipliers_vec[index_in_group].zmm_pd = _mm512_loadu_pd(&multipliers_[dim]);
            negative_discarding_multipliers_vec[index_in_group].zmm_pd =
                _mm512_loadu_pd(&negative_discarding_multipliers_[dim]);
            modulos_vec[index_in_group].zmm_pd = _mm512_loadu_pd(&modulos_[dim]);
            inverse_modulos_vec[index_in_group].zmm_pd = _mm512_loadu_pd(&inverse_modulos_[dim]);
        }

        // Until we reach the `window_width_k`, we don't need to discard any symbols and can keep the code simpler
        std::size_t const prefix_length = (std::min)(text_chunk.size(), window_width_k);
        std::size_t new_char_offset = passed_progress;
        for (; new_char_offset < prefix_length; ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            rolling_state_t const new_term = static_cast<rolling_state_t>(new_char) + 1.0;
            __m512d new_term_zmm = _mm512_set1_pd(new_term);

#pragma unroll(unroll_factor_k)
            for (unsigned index_in_group = 0; index_in_group < unroll_factor_k; ++index_in_group) {
                last_states_vec[index_in_group].zmm_pd = _mm512_fmadd_pd(
                    last_states_vec[index_in_group].zmm_pd, multipliers_vec[index_in_group].zmm_pd, new_term_zmm);
                last_states_vec[index_in_group].zmm_pd = barrett_mod( //
                    last_states_vec[index_in_group].zmm_pd,           //
                    modulos_vec[index_in_group].zmm_pd,               //
                    inverse_modulos_vec[index_in_group].zmm_pd);
            }
        }

        // We now have our first minimum hashes
        __m256i const ones_ymm = _mm256_set1_epi32(1);
        if (new_char_offset == window_width_k && passed_progress < prefix_length)
#pragma unroll(unroll_factor_k)
            for (unsigned index_in_group = 0; index_in_group < unroll_factor_k; ++index_in_group)
                rolling_minimums_vec[index_in_group].zmm_pd = last_states_vec[index_in_group].zmm_pd,
                rolling_counts_vec[index_in_group].ymm = ones_ymm;

        // Now we can avoid a branch in the nested loop, as we are passed the longest window width
        for (; new_char_offset < text_chunk.size(); ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            byte_t const old_char = text_chunk[new_char_offset - window_width_k];
            rolling_state_t const new_term = static_cast<rolling_state_t>(new_char) + 1.0;
            rolling_state_t const old_term = static_cast<rolling_state_t>(old_char) + 1.0;
            __m512d new_term_zmm = _mm512_set1_pd(new_term);
            __m512d old_term_zmm = _mm512_set1_pd(old_term);

#pragma unroll(unroll_factor_k)
            for (unsigned index_in_group = 0; index_in_group < unroll_factor_k; ++index_in_group) {

                // Discard the old term
                last_states_vec[index_in_group].zmm_pd =
                    _mm512_fmadd_pd(negative_discarding_multipliers_vec[index_in_group].zmm_pd, old_term_zmm,
                                    last_states_vec[index_in_group].zmm_pd);
                last_states_vec[index_in_group].zmm_pd = barrett_mod( //
                    last_states_vec[index_in_group].zmm_pd,           //
                    modulos_vec[index_in_group].zmm_pd,               //
                    inverse_modulos_vec[index_in_group].zmm_pd);

                // Add the new term
                last_states_vec[index_in_group].zmm_pd = _mm512_fmadd_pd(
                    last_states_vec[index_in_group].zmm_pd, multipliers_vec[index_in_group].zmm_pd, new_term_zmm);
                last_states_vec[index_in_group].zmm_pd = barrett_mod( //
                    last_states_vec[index_in_group].zmm_pd,           //
                    modulos_vec[index_in_group].zmm_pd,               //
                    inverse_modulos_vec[index_in_group].zmm_pd);

                // To keep the right comparison mask, check out: https://stackoverflow.com/q/16988199
                __mmask8 same_mask = _mm512_cmp_pd_mask(rolling_minimums_vec[index_in_group].zmm_pd,
                                                        last_states_vec[index_in_group].zmm_pd, _CMP_EQ_OQ);
                rolling_minimums_vec[index_in_group].zmm_pd =
                    _mm512_min_pd(rolling_minimums_vec[index_in_group].zmm_pd, last_states_vec[index_in_group].zmm_pd);
                rolling_counts_vec[index_in_group].ymm =
                    _mm256_mask_add_epi32(rolling_counts_vec[index_in_group].ymm, same_mask,
                                          rolling_counts_vec[index_in_group].ymm, ones_ymm);
            }
        }

        // Dump back the results from registers into our spans
        if (has_incomplete_tail_group_k && group_index + 1 == groups_count_k)
#pragma unroll(unroll_factor_k)
            for (unsigned index_in_group = 0; index_in_group < unroll_factor_k; ++index_in_group) {
                unsigned const dim = group_index * hashes_per_unrolled_group_k + index_in_group * hashes_per_zmm_k;
                __mmask8 const store_mask = dimensions_k > dim ? sz_u8_mask_until_(dimensions_k - dim) : (__mmask8)0;
                _mm512_mask_storeu_pd(&last_states[dim], store_mask, last_states_vec[index_in_group].zmm_pd);
                _mm512_mask_storeu_pd(&rolling_minimums[dim], store_mask, rolling_minimums_vec[index_in_group].zmm_pd);
                _mm256_mask_storeu_epi32(&rolling_counts[dim], store_mask, rolling_counts_vec[index_in_group].ymm);
            }
        else
            for (unsigned index_in_group = 0; index_in_group < unroll_factor_k; ++index_in_group) {
                unsigned const dim = group_index * hashes_per_unrolled_group_k + index_in_group * hashes_per_zmm_k;
                _mm512_storeu_pd(&last_states[dim], last_states_vec[index_in_group].zmm_pd);
                _mm512_storeu_pd(&rolling_minimums[dim], rolling_minimums_vec[index_in_group].zmm_pd);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(&rolling_counts[dim]),
                                    rolling_counts_vec[index_in_group].ymm);
            }
    }
};

#pragma clang attribute pop
#pragma GCC pop_options
#endif // SZ_USE_SKYLAKE

#pragma endregion - Optimized Rolling MinHashers

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_FINGERPRINT_HPP_
