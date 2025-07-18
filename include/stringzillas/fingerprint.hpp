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
 *  @section Fingerprinting/Sketching via @b Weighted-Min-Hashing
 *
 *  Computing one such hash won't help us much in large-scale retrieval tasks, but there is a common technique
 *  called "Min-Hashing" that can. The idea is built around the T
 */
#ifndef STRINGZILLAS_FINGERPRINT_HPP_
#define STRINGZILLAS_FINGERPRINT_HPP_

#include "stringzilla/types.hpp"  // `sz::error_cost_t`
#include "stringzilla/memory.h"   // `sz_move`
#include "stringzillas/types.hpp" // `sz::executor_like`

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
    using hash_t = hash_type_;

    explicit multiplying_rolling_hasher(std::size_t window_width, hash_t multiplier = static_cast<hash_t>(257)) noexcept
        : window_width_ {window_width}, multiplier_ {multiplier}, highest_power_ {1} {

        _sz_assert(window_width_ > 1 && "Window width must be > 1");
        _sz_assert(multiplier_ > 0 && "Multiplier must be positive");

        for (std::size_t i = 0; i + 1 < window_width_; ++i) highest_power_ = highest_power_ * multiplier_;
    }

    inline std::size_t window_width() const noexcept { return window_width_; }

    inline hash_t update(hash_t old_hash, byte_t new_char) const noexcept { return old_hash * multiplier_ + new_char; }

    inline hash_t update(hash_t const old_hash, byte_t const old_char, byte_t const new_char) const noexcept {
        hash_t const without_head = old_hash - old_char * highest_power_;
        return without_head * multiplier_ + new_char;
    }

  private:
    std::size_t window_width_;
    hash_t multiplier_;
    hash_t highest_power_;
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
    using accumulator_t = accumulator_type_;

    static_assert(std::is_same<hash_t, std::uint16_t>::value || std::is_same<hash_t, std::uint32_t>::value ||
                      std::is_same<hash_t, std::uint64_t>::value,
                  "Unsupported hash type");

    static constexpr hash_t default_alphabet_size_k = 256u;
    static constexpr hash_t default_modulo_base_k = //
        std::is_same_v<hash_t, std::uint16_t>   ? SZ_U16_MAX_PRIME
        : std::is_same_v<hash_t, std::uint32_t> ? SZ_U32_MAX_PRIME
                                                : SZ_U64_MAX_PRIME;

    explicit rabin_karp_rolling_hasher(              //
        std::size_t window_width,                    //
        hash_t multiplier = default_alphabet_size_k, //
        hash_t modulo = default_modulo_base_k) noexcept
        : window_width_ {window_width}, modulo_ {modulo}, multiplier_ {multiplier}, discarding_multiplier_ {1} {

        _sz_assert(window_width_ > 1 && "Window width must be > 1");
        _sz_assert(multiplier_ > 0 && "Multiplier must be positive");
        _sz_assert(modulo_ > 1 && "Modulo base must be > 1");

        for (std::size_t i = 0; i + 1 < window_width_; ++i)
            discarding_multiplier_ = mul_mod(discarding_multiplier_, multiplier_);
    }

    inline std::size_t window_width() const noexcept { return window_width_; }

    inline hash_t update(hash_t old_hash, byte_t new_char) const noexcept {
        return add_mod(mul_mod(old_hash, multiplier_), new_char);
    }

    inline hash_t update(hash_t const old_hash, byte_t const old_char, byte_t const new_char) const noexcept {
        hash_t const term_to_subtract = mul_mod(old_char, discarding_multiplier_);
        hash_t const without_head = sub_mod(old_hash, term_to_subtract);
        return add_mod(mul_mod(without_head, multiplier_), new_char);
    }

  private:
    inline hash_t mul_mod(hash_t const a, hash_t const b) const noexcept {
        accumulator_t const prod = accumulator_t {a} * accumulator_t {b};
        return static_cast<hash_t>(prod % modulo_);
    }

    inline hash_t add_mod(hash_t const a, hash_t const b) const noexcept {
        accumulator_t const sum = accumulator_t {a} + accumulator_t {b};
        return static_cast<hash_t>(sum % modulo_);
    }

    inline hash_t sub_mod(hash_t const a, hash_t const b) const noexcept {
        accumulator_t diff = accumulator_t {a} + modulo_ - accumulator_t {b};
        return static_cast<hash_t>(diff % modulo_);
    }

    std::size_t window_width_;
    hash_t modulo_;
    hash_t multiplier_;
    hash_t discarding_multiplier_;
};

/**
 *  @brief BuzHash rolling hash function leveraging a fixed-size lookup table and bitwise operations.
 *  @tparam hash_type_ Type of the hash value, e.g., `std::uint64_t`.
 *  @sa `multiplying_rolling_hasher`, `rabin_karp_rolling_hasher`
 */
template <typename hash_type_ = std::uint64_t>
struct buz_rolling_hasher {
    using hash_t = hash_type_;

    explicit buz_rolling_hasher(std::size_t window_width, std::uint64_t seed = 0x9E3779B97F4A7C15ull) noexcept
        : window_width_ {window_width} {

        _sz_assert(window_width_ > 1 && "Window width must be > 1");
        for (std::size_t i = 0; i < 256; ++i) table_[i] = split_mix64(seed);
    }

    inline std::size_t window_width() const noexcept { return window_width_; }

    inline hash_t update(hash_t old_hash, byte_t new_char) const noexcept {
        return rotl(old_hash, 1) ^ table_[new_char & 0xFFu];
    }

    inline hash_t update(hash_t const old_hash, byte_t const old_char, byte_t const new_char) const noexcept {
        constexpr unsigned bits_k = sizeof(hash_t) * 8u;

        hash_t const rolled = rotl(old_hash, 1);
        hash_t const remove_term = rotl(table_[old_char & 0xFFu], window_width_ & (bits_k - 1u));
        return rolled ^ remove_term ^ table_[new_char & 0xFFu];
    }

  private:
    static inline hash_t rotl(hash_t const v, unsigned const r) noexcept {
        constexpr unsigned bits_k = sizeof(hash_t) * 8u;
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
    hash_t table_[256];
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
        if (std::gcd(p, multiplier) != 1) continue;

    return 0;
}

template <typename float_type_ = float>
struct floating_rolling_hasher;

/**
 *  @brief Rabin-Karp-style Rolling hash function for single-precision floating-point numbers.
 *  @tparam float_type_ Type of the floating-point number, e.g., `float`.
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
    using hash_t = std::uint32_t;
    using float_t = float;

    static constexpr float_t limit_k = 8'388'607.0f;
    static constexpr hash_t default_alphabet_size_k = 256u;
    static constexpr hash_t default_modulo_base_k = 8123u;

    explicit floating_rolling_hasher(                      //
        std::size_t const window_width,                    //
        hash_t const multiplier = default_alphabet_size_k, //
        hash_t const modulo = default_modulo_base_k) noexcept
        : window_width_ {window_width}, multiplier_ {static_cast<float_t>(multiplier)},
          modulo_ {static_cast<float_t>(modulo)}, inverse_modulo_ {1.0f / modulo_},
          negative_discarding_multiplier_ {1.0f} {

        _sz_assert(window_width_ > 1 && "Window width must be > 1");
        _sz_assert(multiplier_ > 0 && "Multiplier must be positive");
        _sz_assert(modulo_ > 1 && "Modulo must be > 1");

        // If we want to avoid hitting +inf or NaN, we need to make sure that the product of our post-modulo
        // normalized number with the multiplier and added subsequent term stays within the exactly representable range.
        float_t const largest_input_term = std::numeric_limits<byte_t>::max() + 1.0f;
        float_t const largest_normalized_state = modulo_ - 1;
        float_t const largest_intermediary = largest_normalized_state * multiplier_ + largest_input_term;
        _sz_assert(largest_intermediary < limit_k && "Intermediate state overflows the limit");

        // ! The GCC header misses the `std::fmodf` overload, so we use the underlying C version
        for (std::size_t i = 0; i + 1 < window_width_; ++i)
            negative_discarding_multiplier_ = ::fmodf(negative_discarding_multiplier_ * multiplier_, modulo_);
        negative_discarding_multiplier_ = -negative_discarding_multiplier_;
    }

    inline std::size_t window_width() const noexcept { return window_width_; }

    inline hash_t update(hash_t const old_hash, byte_t const new_char) const noexcept {

        float_t state = sz_bitcast(float_t, old_hash);
        float_t new_term = float_t(new_char) + 1.0f;

        state = std::fmaf(state, multiplier_, new_term);
        state = reduce(state);

        return sz_bitcast(hash_t, state);
    }

    inline hash_t update(hash_t const old_hash, byte_t const old_char, byte_t const new_char) const noexcept {

        float_t state = sz_bitcast(float_t, old_hash);
        float_t old_term = float_t(old_char) + 1.0f;
        float_t new_term = float_t(new_char) + 1.0f;

        state = std::fmaf(state, negative_discarding_multiplier_, old_term); // Remove tail
        state = std::fmaf(state, multiplier_, new_term);                     // Add head
        state = reduce(state);

        return sz_bitcast(hash_t, state);
    }

  private:
    /**
     *  @brief Barrett-style `std::fmodf` alternative to avoid overflow.
     *  @see https://en.cppreference.com/w/cpp/numeric/math/fmod
     */
    inline float_t reduce(float_t h) const noexcept {
        h -= modulo_ * std::floor(h * inverse_modulo_);
        // Clamp into the [0, modulo_) range.
        h += modulo_ * (h < 0.0f);
        h -= modulo_ * (h >= modulo_);
        return h;
    }

    std::size_t window_width_;
    float_t multiplier_;
    float_t modulo_;
    float_t inverse_modulo_;
    float_t negative_discarding_multiplier_;
};

/**
 *  @brief Rabin-Karp-style Rolling hash function for double-precision floating-point numbers.
 *  @tparam float_type_ Type of the floating-point number, e.g., `float`.
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
 *
 *      4'503'599'626'781, 4'503'599'626'783, 4'503'599'626'807,
 *      4'503'599'626'907, 4'503'599'626'957, 4'503'599'626'977.
 *
 *  @sa `floating_rolling_hasher<double>` for 52 bit variant.
 */
template <>
struct floating_rolling_hasher<double> {
    using hash_t = std::uint64_t;
    using float_t = double;

    static constexpr float_t limit_k = 4503599627370495.0;
    static constexpr hash_t default_alphabet_size_k = 256u;
    static constexpr hash_t default_modulo_base_k = 4503599626977u;

    explicit floating_rolling_hasher(                      //
        std::size_t const window_width,                    //
        hash_t const multiplier = default_alphabet_size_k, //
        hash_t const modulo = default_modulo_base_k) noexcept
        : window_width_ {window_width}, multiplier_ {static_cast<float_t>(multiplier)},
          modulo_ {static_cast<float_t>(modulo)}, inverse_modulo_ {1.0 / modulo_},
          negative_discarding_multiplier_ {1.0} {

        _sz_assert(window_width_ > 1 && "Window width must be > 1");
        _sz_assert(multiplier_ > 0 && "Multiplier must be positive");
        _sz_assert(modulo_ > 1 && "Modulo must be > 1");

        // If we want to avoid hitting +inf or NaN, we need to make sure that the product of our post-modulo
        // normalized number with the multiplier and added subsequent term stays within the exactly representable range.
        float_t const largest_input_term = std::numeric_limits<byte_t>::max() + 1.0;
        float_t const largest_normalized_state = modulo_ - 1;
        float_t const largest_intermediary = largest_normalized_state * multiplier_ + largest_input_term;
        _sz_assert(largest_intermediary < limit_k && "Intermediate state overflows the limit");

        for (std::size_t i = 0; i + 1 < window_width_; ++i)
            negative_discarding_multiplier_ = std::fmod(negative_discarding_multiplier_ * multiplier_, modulo_);
        negative_discarding_multiplier_ = -negative_discarding_multiplier_;
    }

    inline std::size_t window_width() const noexcept { return window_width_; }

    inline hash_t update(hash_t const old_hash, byte_t const new_char) const noexcept {

        float_t state = sz_bitcast(float_t, old_hash);
        float_t new_term = float_t(new_char) + 1.0;

        state = std::fma(state, multiplier_, new_term);
        state = reduce(state);

        return sz_bitcast(hash_t, state);
    }

    inline hash_t update(hash_t const old_hash, byte_t const old_char, byte_t const new_char) const noexcept {

        float_t state = sz_bitcast(float_t, old_hash);
        float_t old_term = float_t(old_char) + 1.0;
        float_t new_term = float_t(new_char) + 1.0;

        state = std::fma(state, negative_discarding_multiplier_, old_term); // Remove tail
        state = std::fma(state, multiplier_, new_term);                     // Add head
        state = reduce(state);

        return sz_bitcast(hash_t, state);
    }

    inline float_t multiplier() const noexcept { return multiplier_; }
    inline float_t modulo() const noexcept { return modulo_; }
    inline float_t inverse_modulo() const noexcept { return inverse_modulo_; }
    inline float_t negative_discarding_multiplier() const noexcept { return negative_discarding_multiplier_; }

  private:
    /**
     *  @brief Barrett-style `std::fmod` alternative to avoid overflow.
     *  @see https://en.cppreference.com/w/cpp/numeric/math/fmod
     */
    inline float_t reduce(float_t h) const noexcept {
        h -= modulo_ * std::floor(h * inverse_modulo_);
        // Clamp into the [0, modulo_) range.
        h += modulo_ * (h < 0.0);
        h -= modulo_ * (h >= modulo_);
        return h;
    }

    std::size_t window_width_;
    float_t multiplier_;
    float_t modulo_;
    float_t inverse_modulo_;
    float_t negative_discarding_multiplier_;
};

#pragma endregion - Baseline Rolling Hashers

#pragma region - Optimized Rolling MinHashers

/**
 *  @brief Boring Min-Hash implementation on top of the baseline Rabin Karp algorithm just for benchmarking.
 *  @tparam hasher_type_ Can be the Rabin-Karp, BuzHash, or anything else compatible.
 */
template <                                                                           //
    typename hasher_type_ = rabin_karp_rolling_hasher<std::uint32_t, std::uint64_t>, //
    typename allocator_type_ = std::allocator<hasher_type_>,                         //
    typename scalar_type_ = typename hasher_type_::hash_t                            //
    >
struct basic_rolling_hashers {

    using hasher_t = hasher_type_;
    using rolling_hash_t = typename hasher_t::hash_t;
    using result_scalar_t = scalar_type_;
    using allocator_t = allocator_type_;

    static constexpr rolling_hash_t skipped_rolling_hash_k = std::numeric_limits<rolling_hash_t>::max();
    static constexpr result_scalar_t max_result_scalar_k = std::numeric_limits<result_scalar_t>::max();

    struct state_t {
        rolling_hash_t last = 0;
        rolling_hash_t minimum = skipped_rolling_hash_k;
    };

  private:
    using allocator_traits_t = std::allocator_traits<allocator_type_>;
    using hasher_allocator_t = typename allocator_traits_t::template rebind_alloc<hasher_t>;
    using state_allocator_t = typename allocator_traits_t::template rebind_alloc<state_t>;

    using hashers_t = safe_vector<hasher_t, hasher_allocator_t>;
    using states_t = safe_vector<state_t, state_allocator_t>;

    allocator_t allocator_;
    hashers_t hashers_;
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
            _sz_assert(status == status_t::success_k && "Couldn't fail after the reserve");
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
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] result The output fingerprint, a vector of minimum hashes.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     */
    template <size_t dimensions_ = SZ_SIZE_MAX>
    status_t try_fingerprint(span<byte_t const> text, span<result_scalar_t, dimensions_> result) const noexcept {
        _sz_assert(result.size() == dimensions() && "Dimensions number & hashers number mismatch");

        // Allocate temporary states
        states_t states(allocator_traits_t::select_on_container_copy_construction(allocator_));
        if (!states.try_resize(dimensions())) return status_t::bad_alloc_k;

        fingerprint(text, states, result);
        return status_t::success_k;
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[inout] states The states of the hashers, used to avoid re-allocating temporary buffers.
     *  @param[out] result The output fingerprint, a vector of minimum hashes.
     */
    template <size_t dimensions_ = SZ_SIZE_MAX>
    void fingerprint(span<byte_t const> text, span<state_t, dimensions_> states,
                     span<result_scalar_t, dimensions_> result) const noexcept {

        _sz_assert(result.size() == dimensions() && "Dimensions number & hashers number mismatch");
        _sz_assert(states.size() == dimensions() && "Dimensions number & states number mismatch");

        fill_states_(text, states);

        // Export the minimum hashes to the fingerprint
        for (std::size_t dim = 0; dim < result.size(); ++dim) {
            hasher_t const &hasher = hashers_[dim];
            rolling_hash_t minimum = hasher.window_width() < text.size() ? skipped_rolling_hash_k : states[dim].minimum;
            result[dim] = static_cast<result_scalar_t>(minimum & max_result_scalar_k);
        }
    }

    /**
     *  @brief Computes many fingerprints in parallel for input @p texts via an @p executor.
     *  @param[in] texts The input texts to hash, typically a UTF-8 encoded string.
     *  @param[out] results The output fingerprints, a array of vectors of minimum hashes.
     *  @param[in] executor The executor to use for parallel processing, defaults to a dummy executor.
     *  @param[in] specs The CPU specifications to use, defaults to an empty `cpu_specs_t`.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     */
    template <typename texts_type_, typename fingerprints_type_, typename executor_type_ = dummy_executor_t>
#if _SZ_IS_CPP20
        requires executor_like<executor_type_>
#endif
    status_t operator()(                                        //
        texts_type_ const &texts, fingerprints_type_ &&results, //
        executor_type_ &&executor = {}, cpu_specs_t specs = {}) const noexcept {

        // Depending on document sizes, choose the appropriate parallelization strategy
        // - Either split each text into chunks across threads
        // - Or split the texts themselves across threads
        std::size_t const text_size_threshold = specs.l2_bytes * executor.threads_count();
        std::size_t const dims = dimensions();

        // Allocate enough temporary states for all cores to have individual states
        states_t states(allocator_traits_t::select_on_container_copy_construction(allocator_));
        if (states.try_resize(executor.threads_count() * dims) != status_t::success_k) return status_t::bad_alloc_k;

        // Process small texts by individual threads
        executor.for_n_dynamic(texts.size(), [&](auto prong) noexcept {
            auto const text_index = prong.task;
            auto const thread_index = prong.thread;

            auto const &text = texts[text_index];
            if (text.size() >= text_size_threshold) return;

            auto text_view = to_bytes_view(text);
            auto thread_local_states = to_span(states).subspan(thread_index * dims, dims);
            auto result = to_span(results[text_index]);
            fingerprint<SZ_SIZE_MAX>(text_view, thread_local_states, result);
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
                auto text_start = text_view.data() + std::min(text_view.size(), thread_index * chunk_size);
                // ? This overlap will be different for different window widths, but assuming we are
                // ? computing the non-weighted Min-Hash, recomputing & comparing a few hashes for the
                // ? same slices isn't a big deal.
                auto overlapping_text_end = std::min(text_start + chunk_size + max_window_width_ - 1, text_view.end());
                auto thread_local_text = span<byte_t const>(text_start, overlapping_text_end);
                auto thread_local_states = to_span(states).subspan(thread_index * dims, dims);
                fill_states_<SZ_SIZE_MAX>(thread_local_text, thread_local_states);
            });

            // Compute the minimums of each thread's local states
            auto &result = results[text_index];
            for (std::size_t dim = 0; dim < result.size(); ++dim) {
                rolling_hash_t minimum_across_thread_local_chunks = skipped_rolling_hash_k;
                for (std::size_t thread_index = 0; thread_index < executor.threads_count(); ++thread_index) {
                    auto const &state = states[thread_index * dims + dim];
                    minimum_across_thread_local_chunks = (std::min)(minimum_across_thread_local_chunks, state.minimum);
                }
                minimum_across_thread_local_chunks &= max_result_scalar_k; // Clamp to the result scalar range
                result[dim] = static_cast<result_scalar_t>(minimum_across_thread_local_chunks);
            }
        }

        return status_t::success_k;
    }

  private:
    template <size_t dimensions_ = SZ_SIZE_MAX>
    void fill_states_(span<byte_t const> text, span<state_t, dimensions_> states) const noexcept {

        _sz_assert(states.size() >= hashers_.size() && "Dimensions number & states number mismatch");

        // Clear the states
        for (auto &state : states) state = state_t {};

        // Until we reach the maximum window length, use a branching code version
        std::size_t const prefix_length = (std::min)(text.size(), max_window_width_);
        for (std::size_t new_char_offset = 0; new_char_offset < prefix_length; ++new_char_offset) {
            byte_t const new_char = text[new_char_offset];
            for (std::size_t dim = 0; dim < states.size(); ++dim) {
                auto &hasher = hashers_[dim];
                auto &state = states[dim];
                if (hasher.window_width() > new_char_offset) { state.last = hasher.update(state.last, new_char); }
                else {
                    auto const old_char = text[new_char_offset - hasher.window_width()];
                    state.last = hasher.update(state.last, old_char, new_char);
                    state.minimum = (std::min)(state.minimum, state.last);
                }
            }
        }

        // Now we can avoid a branch in the nested loop, as we are passed the longest window width
        for (std::size_t new_char_offset = max_window_width_; new_char_offset < text.size(); ++new_char_offset) {
            byte_t const new_char = text[new_char_offset];
            for (std::size_t dim = 0; dim < states.size(); ++dim) {
                auto &hasher = hashers_[dim];
                auto &state = states[dim];
                auto const old_char = text[new_char_offset - hasher.window_width()];
                state.last = hasher.update(state.last, old_char, new_char);
                state.minimum = (std::min)(state.minimum, state.last);
            }
        }
    }
};

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
 *  @tparam capability_ The CPU capability, e.g., `sz_cap_serial_k`, `sz_cap_ice_k`, etc.
 *  @tparam window_width_ The width of the rolling window, e.g., 3, 4, 5, 6, etc.
 *  @tparam dimensions_ The number of dimensions in the fingerprint, ideally a multiple of 16, like 64 or 80.
 *  @tparam enable_ A type used to enable or disable this specialization, e.g., `void` for default.
 */
template <                                         //
    sz_capability_t capability_ = sz_cap_serial_k, //
    std::size_t window_width_ = SZ_SIZE_MAX,       //
    std::size_t dimensions_ = 16,                  //
    typename enable_ = void                        //
    >
struct floating_rolling_hashers {

    using hasher_t = floating_rolling_hasher<double>;
    using rolling_hash_t = typename hasher_t::hash_t;
    using result_scalar_t = std::uint32_t;

    static constexpr std::size_t window_width_k = window_width_;
    static constexpr std::size_t dimensions_k = dimensions_;
    static constexpr result_scalar_t skipped_rolling_hash_k = std::numeric_limits<result_scalar_t>::max();

    using fingerprint_span_t = span<result_scalar_t, dimensions_k>;

  private:
    using float_t = typename hasher_t::float_t;

    float_t multipliers_[dimensions_k];
    float_t modulos_[dimensions_k];
    float_t inverse_modulos_[dimensions_k];
    float_t negative_discarding_multipliers_[dimensions_k];

  public:
    constexpr std::size_t window_width() const noexcept { return window_width_k; }
    constexpr std::size_t dimensions() const noexcept { return dimensions_k; }

    /**
     *  @brief Initializes several rolling hashers with different multipliers and modulos.
     *  @param[in] alphabet_size Size of the alphabet, typically 256 for UTF-8, 4 for DNA, or 20 for proteins.
     */
    status_t try_seed(std::size_t alphabet_size = 256) noexcept {
        for (std::size_t j = 0; j < dimensions_k; ++j) {
            hasher_t hasher(window_width_k, alphabet_size + j, hasher_t::default_modulo_base_k);
            multipliers_[j] = hasher.multiplier();
            modulos_[j] = hasher.modulo();
            inverse_modulos_[j] = hasher.inverse_modulo();
            negative_discarding_multipliers_[j] = hasher.negative_discarding_multiplier();
        }
        return status_t::success_k;
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] result The output fingerprint, a vector of minimum hashes.
     */
    void fingerprint(span<byte_t const> text, fingerprint_span_t result) const noexcept {
        if (text.size() < window_width_k) return;

        // Fill the states
        rolling_hash_t last_hashes[dimensions_k];
        rolling_hash_t minimum_hashes[dimensions_k];
        fill_states_(text, last_hashes, minimum_hashes);

        // Export the minimum hashes to the fingerprint
        for (std::size_t dim = 0; dim < dimensions_k; ++dim)
            result[dim] = static_cast<result_scalar_t>(minimum_hashes[dim]);
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] result The output fingerprint, a vector of minimum hashes.
     */
    status_t try_fingerprint(span<byte_t const> text, fingerprint_span_t result) const noexcept {
        fingerprint(text, result);
        return status_t::success_k;
    }

    /**
     *  @brief Computes many fingerprints in parallel for input @p texts via an @p executor.
     *  @param[in] texts The input texts to hash, typically a UTF-8 encoded string.
     *  @param[out] results The output fingerprints, a array of vectors of minimum hashes.
     *  @param[in] executor The executor to use for parallel processing, defaults to a dummy executor.
     *  @param[in] specs The CPU specifications to use, defaults to an empty `cpu_specs_t`.
     *  @retval status_t::success_k on success, or an error code otherwise.
     *  @retval status_t::bad_alloc_k if the memory allocation fails.
     */
    template <typename texts_type_, typename fingerprints_type_, typename executor_type_ = dummy_executor_t>
#if _SZ_IS_CPP20
        requires executor_like<executor_type_>
#endif
    status_t operator()(                                        //
        texts_type_ const &texts, fingerprints_type_ &&results, //
        executor_type_ &&executor = {}, cpu_specs_t specs = {}) const noexcept {

        // Depending on document sizes, choose the appropriate parallelization strategy
        // - Either split each text into chunks across threads
        // - Or split the texts themselves across threads
        std::size_t const text_size_threshold = specs.l2_bytes * executor.threads_count();

        // Process small texts by individual threads
        executor.for_n_dynamic(texts.size(), [&](auto prong) noexcept {
            auto const text_index = prong.task;

            auto const &text = texts[text_index];
            if (text.size() >= text_size_threshold) return;

            auto text_view = to_bytes_view(text);
            auto result = to_span<dimensions_k>(results[text_index]);
            fingerprint(text_view, result);
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

            auto gather_mutex = executor.make_mutex();
            rolling_hash_t minimum_hashes[dimensions_k];

            // Distribute overlapping chunks across threads
            executor.for_threads([&](std::size_t thread_index) noexcept {
                auto text_start = text_view.data() + std::min(text_view.size(), thread_index * chunk_size);
                // ? This overlap will be different for different window widths, but assuming we are
                // ? computing the non-weighted Min-Hash, recomputing & comparing a few hashes for the
                // ? same slices isn't a big deal.
                auto overlapping_text_end = std::min(text_start + chunk_size + window_width_k - 1, text_view.end());
                auto thread_local_text = span<byte_t const>(text_start, overlapping_text_end);

                rolling_hash_t thread_local_last_hashes[dimensions_k];
                rolling_hash_t thread_local_minimum_hashes[dimensions_k];
                fill_states_(thread_local_text, thread_local_last_hashes, thread_local_minimum_hashes);

                lock_guard lock(gather_mutex);
                for (std::size_t dim = 0; dim < dimensions_k; ++dim)
                    minimum_hashes[dim] = (std::min)(minimum_hashes[dim], thread_local_minimum_hashes[dim]);
            });

            // Compute the minimums of each thread's local states
            auto &result = results[text_index];
            for (std::size_t dim = 0; dim < dimensions_k; ++dim)
                result[dim] = static_cast<result_scalar_t>(minimum_hashes[dim]);
        }

        return status_t::success_k;
    }

  private:
    inline float_t reduce(float_t h, std::size_t dim) const noexcept {
        float_t const modulo = modulos_[dim];
        float_t const inverse_modulo = inverse_modulos_[dim];
        // Using `trunc` instead of `floor` for modulo.
        // This is branchless and avoids STL.
        h -= modulo * static_cast<float_t>(static_cast<long long>(h * inverse_modulo));

        // Branchless clamp to [0, modulo).
        // `h` is in (-modulo, modulo).
        // If h is negative, add modulo.
        h += modulo * static_cast<float_t>(sz_bitcast(std::uint64_t, h) >> 63);
        // Now `h` is in [0, 2*modulo).
        // If h is >= modulo, subtract modulo.
        long long is_ge = static_cast<long long>(h * inverse_modulo);
        h -= modulo * static_cast<float_t>(is_ge);
        return h;
    }

    void fill_states_(span<byte_t const> text, span<rolling_hash_t, dimensions_k> last_hashes,
                      span<rolling_hash_t, dimensions_k> minimum_hashes) const noexcept {

        for (std::size_t dim = 0; dim < dimensions_k; ++dim) {
            last_hashes[dim] = 0;
            minimum_hashes[dim] = skipped_rolling_hash_k;
        }

        // Until we reach the maximum window length, use a branching code version
        for (std::size_t new_char_offset = 0; new_char_offset < window_width_k; ++new_char_offset) {
            byte_t const new_char = text[new_char_offset];
            float_t const new_term = static_cast<float_t>(new_char) + 1.0;
            for (std::size_t dim = 0; dim < dimensions_k; ++dim) {
                rolling_hash_t &hash = last_hashes[dim];
                float_t state = sz_bitcast(float_t, hash);
                state += multipliers_[dim] * new_term;
                state = reduce(state, dim);

                // Save back
                hash = sz_bitcast(rolling_hash_t, state);
                minimum_hashes[dim] = std::min(minimum_hashes[dim], hash);
            }
        }

        // Now we can avoid a branch in the nested loop, as we are passed the longest window width
        for (std::size_t new_char_offset = window_width_k; new_char_offset < text.size(); ++new_char_offset) {
            byte_t const new_char = text[new_char_offset];
            byte_t const old_char = text[new_char_offset - window_width_k];
            float_t const new_term = static_cast<float_t>(new_char) + 1.0;
            float_t const old_term = static_cast<float_t>(old_char) + 1.0;
            for (std::size_t dim = 0; dim < dimensions_k; ++dim) {
                rolling_hash_t &hash = last_hashes[dim];
                float_t state = sz_bitcast(float_t, hash);
                state += negative_discarding_multipliers_[dim] * old_term; // Remove tail
                state += multipliers_[dim] * new_term;                     // Add head
                state = reduce(state, dim);

                // Save back
                hash = sz_bitcast(rolling_hash_t, state);
                minimum_hashes[dim] = std::min(minimum_hashes[dim], hash);
            }
        }
    }
};

/**
 *  @brief Optimized rolling Min-Hashers built around floating-point numbers.
 *
 *  In a single ZMM register we can store 8 `double` values, so we can process 8 hashes per register.
 *  Assuming 32x ZMM registers, and roughly 10ish scalars for intermediaries per hash, we can unroll
 *  2-3x times, and process 16-24 hashes in parallel.
 */
template <std::size_t dimensions_>
struct floating_rolling_hashers<sz_cap_ice_k, dimensions_> {

    using hasher_t = floating_rolling_hasher<double>;
    using rolling_hash_t = typename hasher_t::hash_t;
    using result_scalar_t = std::uint32_t;
    static constexpr std::size_t dimensions_k = dimensions_;

    void operator()(span<byte_t const> text, span<hasher_t const, dimensions_k> hashers,
                    span<result_scalar_t, dimensions_k> fingerprint) const noexcept {

        constexpr std::size_t unroll_factor_k = 2;
        constexpr std::size_t unrolled_hashes_k = unroll_factor_k * sizeof(sz_u512_vec_t) / sizeof(rolling_hash_t);

        sz_u512_vec_t window_widths[unroll_factor_k], multipliers[unroll_factor_k],
            negative_discarding_multipliers[unroll_factor_k], modulos[unroll_factor_k],
            inverse_modulos[unroll_factor_k], states_[unroll_factor_k], min_hashes[unroll_factor_k];
    }
};

#pragma endregion - Optimized Rolling MinHashers

} // namespace stringzillas
} // namespace ashvardanian

#if 0
#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief  Computes the Karp-Rabin rolling hashes of a string supplying them to the provided `callback`.
 *          Can be used for similarity scores, search, ranking, etc.
 *
 *  Rabin-Karp-like rolling hashes can have very high-level of collisions and depend
 *  on the choice of bases and the prime number. That's why, often two hashes from the same
 *  family are used with different bases.
 *
 *       1. Kernighan and Ritchie's function uses 31, a prime close to the size of English alphabet.
 *       2. To be friendlier to byte-arrays and UTF8, we use 257 for the second function.
 *
 *
 *  @param text             String to hash.
 *  @param length           Number of bytes in the string.
 *  @param window_width    Length of the rolling window in bytes.
 *  @param window_step      Step of reported hashes. @b Must be power of two. Should be smaller than `window_width`.
 *  @param callback         Function receiving the start & length of a substring, the hash, and the `callback_handle`.
 *  @param callback_handle  Optional user-provided pointer to be passed to the `callback`.
 *  @see                    sz_hashes_fingerprint, sz_hashes_intersection
 */
SZ_DYNAMIC void sz_hashes(                                                            //
    sz_cptr_t text, sz_size_t length, sz_size_t window_width, sz_size_t window_step, //
    sz_hash_callback_t callback, void *callback_handle);

/**
 *  @brief  Computes the Karp-Rabin rolling hashes of a string outputting a binary fingerprint.
 *          Such fingerprints can be compared with Hamming or Jaccard (Tanimoto) distance for similarity.
 *
 *  The algorithm doesn't clear the fingerprint buffer on start, so it can be invoked multiple times
 *  to produce a fingerprint of a longer string, by passing the previous fingerprint as the ::fingerprint.
 *  It can also be reused to produce multi-resolution fingerprints by changing the ::window_width
 *  and calling the same function multiple times for the same input ::text.
 *
 *  Processes large strings in parts to maximize the cache utilization, using a small on-stack buffer,
 *  avoiding cache-coherency penalties of remote on-heap buffers.
 *
 *  @param text                 String to hash.
 *  @param length               Number of bytes in the string.
 *  @param fingerprint          Output fingerprint buffer.
 *  @param fingerprint_bytes    Number of bytes in the fingerprint buffer.
 *  @param window_width        Length of the rolling window in bytes.
 *  @see                        sz_hashes, sz_hashes_intersection
 */
SZ_PUBLIC void sz_hashes_fingerprint(                          //
    sz_cptr_t text, sz_size_t length, sz_size_t window_width, //
    sz_ptr_t fingerprint, sz_size_t fingerprint_bytes) {
    sz_unused(text && length && window_width && fingerprint && fingerprint_bytes);
}

/**
 *  @brief  Given a hash-fingerprint of a textual document, computes the number of intersecting hashes
 *          of the incoming document. Can be used for document scoring and search.
 *
 *  Processes large strings in parts to maximize the cache utilization, using a small on-stack buffer,
 *  avoiding cache-coherency penalties of remote on-heap buffers.
 *
 *  @param text                 Input document.
 *  @param length               Number of bytes in the input document.
 *  @param fingerprint          Reference document fingerprint.
 *  @param fingerprint_bytes    Number of bytes in the reference documents fingerprint.
 *  @param window_width        Length of the rolling window in bytes.
 *  @see                        sz_hashes, sz_hashes_fingerprint
 */
SZ_PUBLIC sz_size_t sz_hashes_intersection(                    //
    sz_cptr_t text, sz_size_t length, sz_size_t window_width, //
    sz_cptr_t fingerprint, sz_size_t fingerprint_bytes);

/** @copydoc sz_hashes */
SZ_PUBLIC void sz_hashes_serial(                                                      //
    sz_cptr_t text, sz_size_t length, sz_size_t window_width, sz_size_t window_step, //
    sz_hash_callback_t callback, void *callback_handle);
}

#pragma endregion // Core API

#pragma region Serial Implementation

/*
 *  One hardware-accelerated way of mixing hashes can be CRC, but it's only implemented for 32-bit values.
 *  Using a Boost-like mixer works very poorly in such case:
 *
 *       hash_first ^ (hash_second + 0x517cc1b727220a95 + (hash_first << 6) + (hash_first >> 2));
 *
 *  Let's stick to the Fibonacci hash trick using the golden ratio.
 *  https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
 */
#define _sz_hash_mix(first, second) ((first * 11400714819323198485ull) ^ (second * 11400714819323198485ull))
#define _sz_shift_low(x) (x)
#define _sz_shift_high(x) ((x + 77ull) & 0xFFull)
#define _sz_prime_mod(x) (x % SZ_U64_MAX_PRIME)

SZ_PUBLIC void sz_hashes_serial(sz_cptr_t start, sz_size_t length, sz_size_t window_width, sz_size_t step, //
                                sz_hash_callback_t callback, void *callback_handle) {

    if (length < window_width || !window_width) return;
    sz_u8_t const *text = (sz_u8_t const *)start;
    sz_u8_t const *text_end = text + length;

    // Prepare the `prime ^ window_width` values, that we are going to use for modulo arithmetic.
    sz_u64_t prime_power_low = 1, prime_power_high = 1;
    for (sz_size_t i = 0; i + 1 < window_width; ++i)
        prime_power_low = (prime_power_low * 31ull) % SZ_U64_MAX_PRIME,
        prime_power_high = (prime_power_high * 257ull) % SZ_U64_MAX_PRIME;

    // Compute the initial hash value for the first window.
    sz_u64_t hash_low = 0, hash_high = 0, hash_mix;
    for (sz_u8_t const *first_end = text + window_width; text < first_end; ++text)
        hash_low = (hash_low * 31ull + _sz_shift_low(*text)) % SZ_U64_MAX_PRIME,
        hash_high = (hash_high * 257ull + _sz_shift_high(*text)) % SZ_U64_MAX_PRIME;

    // In most cases the fingerprint length will be a power of two.
    hash_mix = _sz_hash_mix(hash_low, hash_high);
    callback((sz_cptr_t)text, window_width, hash_mix, callback_handle);

    // Compute the hash value for every window, exporting into the fingerprint,
    // using the expensive modulo operation.
    sz_size_t cycles = 1;
    sz_size_t const step_mask = step - 1;
    for (; text < text_end; ++text, ++cycles) {
        // Discard one character:
        hash_low -= _sz_shift_low(*(text - window_width)) * prime_power_low;
        hash_high -= _sz_shift_high(*(text - window_width)) * prime_power_high;
        // And add a new one:
        hash_low = 31ull * hash_low + _sz_shift_low(*text);
        hash_high = 257ull * hash_high + _sz_shift_high(*text);
        // Wrap the hashes around:
        hash_low = _sz_prime_mod(hash_low);
        hash_high = _sz_prime_mod(hash_high);
        // Mix only if we've skipped enough hashes.
        if ((cycles & step_mask) == 0) {
            hash_mix = _sz_hash_mix(hash_low, hash_high);
            callback((sz_cptr_t)text, window_width, hash_mix, callback_handle);
        }
    }
}

/** @brief  An internal callback used to set a bit in a power-of-two length binary fingerprint of a string. */
SZ_INTERNAL void _sz_hashes_fingerprint_pow2_callback(sz_cptr_t start, sz_size_t length, sz_u64_t hash, void *handle) {
    sz_string_view_t *fingerprint_buffer = (sz_string_view_t *)handle;
    sz_u8_t *fingerprint_u8s = (sz_u8_t *)fingerprint_buffer->start;
    sz_size_t fingerprint_bytes = fingerprint_buffer->length;
    fingerprint_u8s[(hash / 8) & (fingerprint_bytes - 1)] |= (1 << (hash & 7));
    sz_unused(start && length);
}

/** @brief  An internal callback used to set a bit in a @b non power-of-two length binary fingerprint of a string. */
SZ_INTERNAL void _sz_hashes_fingerprint_non_pow2_callback( //
    sz_cptr_t start, sz_size_t length, sz_u64_t hash, void *handle) {
    sz_string_view_t *fingerprint_buffer = (sz_string_view_t *)handle;
    sz_u8_t *fingerprint_u8s = (sz_u8_t *)fingerprint_buffer->start;
    sz_size_t fingerprint_bytes = fingerprint_buffer->length;
    fingerprint_u8s[(hash / 8) % fingerprint_bytes] |= (1 << (hash & 7));
    sz_unused(start && length);
}

/** @brief  An internal callback, used to mix all the running hashes into one pointer-size value. */
SZ_INTERNAL void _sz_hashes_fingerprint_scalar_callback( //
    sz_cptr_t start, sz_size_t length, sz_u64_t hash, void *scalar_handle) {
    sz_unused(start && length && hash && scalar_handle);
    sz_size_t *scalar_ptr = (sz_size_t *)scalar_handle;
    *scalar_ptr ^= hash;
}

#undef _sz_shift_low
#undef _sz_shift_high
#undef _sz_hash_mix
#undef _sz_prime_mod

#pragma endregion // Serial Implementation

/*  AVX2 implementation of the string search algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2")
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)

/**
 *  @brief  There is no AVX2 instruction for fast multiplication of 64-bit integers.
 *          This implementation is coming from Agner Fog's Vector Class Library.
 */
SZ_INTERNAL __m256i _mm256_mul_epu64(__m256i a, __m256i b) {
    __m256i bswap = _mm256_shuffle_epi32(b, 0xB1);
    __m256i prodlh = _mm256_mullo_epi32(a, bswap);
    __m256i zero = _mm256_setzero_si256();
    __m256i prodlh2 = _mm256_hadd_epi32(prodlh, zero);
    __m256i prodlh3 = _mm256_shuffle_epi32(prodlh2, 0x73);
    __m256i prodll = _mm256_mul_epu32(a, b);
    __m256i prod = _mm256_add_epi64(prodll, prodlh3);
    return prod;
}

SZ_PUBLIC void sz_hashes_haswell(sz_cptr_t start, sz_size_t length, sz_size_t window_width, sz_size_t step, //
                                 sz_hash_callback_t callback, void *callback_handle) {

    if (length < window_width || !window_width) return;
    if (length < 4 * window_width) {
        sz_hashes_serial(start, length, window_width, step, callback, callback_handle);
        return;
    }

    // Using AVX2, we can perform 4 long integer multiplications and additions within one register.
    // So let's slice the entire string into 4 overlapping windows, to slide over them in parallel.
    sz_size_t const max_hashes = length - window_width + 1;
    sz_size_t const min_hashes_per_thread = max_hashes / 4; // At most one sequence can overlap between 2 threads.
    sz_u8_t const *text_first = (sz_u8_t const *)start;
    sz_u8_t const *text_second = text_first + min_hashes_per_thread;
    sz_u8_t const *text_third = text_first + min_hashes_per_thread * 2;
    sz_u8_t const *text_fourth = text_first + min_hashes_per_thread * 3;
    sz_u8_t const *text_end = text_first + length;

    // Prepare the `prime ^ window_width` values, that we are going to use for modulo arithmetic.
    sz_u64_t prime_power_low = 1, prime_power_high = 1;
    for (sz_size_t i = 0; i + 1 < window_width; ++i)
        prime_power_low = (prime_power_low * 31ull) % SZ_U64_MAX_PRIME,
        prime_power_high = (prime_power_high * 257ull) % SZ_U64_MAX_PRIME;

    // Broadcast the constants into the registers.
    sz_u256_vec_t prime_vec, golden_ratio_vec;
    sz_u256_vec_t base_low_vec, base_high_vec, prime_power_low_vec, prime_power_high_vec, shift_high_vec;
    base_low_vec.ymm = _mm256_set1_epi64x(31ull);
    base_high_vec.ymm = _mm256_set1_epi64x(257ull);
    shift_high_vec.ymm = _mm256_set1_epi64x(77ull);
    prime_vec.ymm = _mm256_set1_epi64x(SZ_U64_MAX_PRIME);
    golden_ratio_vec.ymm = _mm256_set1_epi64x(11400714819323198485ull);
    prime_power_low_vec.ymm = _mm256_set1_epi64x(prime_power_low);
    prime_power_high_vec.ymm = _mm256_set1_epi64x(prime_power_high);

    // Compute the initial hash values for every one of the four windows.
    sz_u256_vec_t hash_low_vec, hash_high_vec, hash_mix_vec, chars_low_vec, chars_high_vec;
    hash_low_vec.ymm = _mm256_setzero_si256();
    hash_high_vec.ymm = _mm256_setzero_si256();
    for (sz_u8_t const *prefix_end = text_first + window_width; text_first < prefix_end;
         ++text_first, ++text_second, ++text_third, ++text_fourth) {

        // 1. Multiply the hashes by the base.
        hash_low_vec.ymm = _mm256_mul_epu64(hash_low_vec.ymm, base_low_vec.ymm);
        hash_high_vec.ymm = _mm256_mul_epu64(hash_high_vec.ymm, base_high_vec.ymm);

        // 2. Load the four characters from `text_first`, `text_first + max_hashes_per_thread`,
        //   `text_first + max_hashes_per_thread * 2`, `text_first + max_hashes_per_thread * 3`.
        chars_low_vec.ymm = _mm256_set_epi64x(text_fourth[0], text_third[0], text_second[0], text_first[0]);
        chars_high_vec.ymm = _mm256_add_epi8(chars_low_vec.ymm, shift_high_vec.ymm);

        // 3. Add the incoming characters.
        hash_low_vec.ymm = _mm256_add_epi64(hash_low_vec.ymm, chars_low_vec.ymm);
        hash_high_vec.ymm = _mm256_add_epi64(hash_high_vec.ymm, chars_high_vec.ymm);

        // 4. Compute the modulo. Assuming there are only 59 values between our prime
        //    and the 2^64 value, we can simply compute the modulo by conditionally subtracting the prime.
        hash_low_vec.ymm = _mm256_blendv_epi8( //
            hash_low_vec.ymm, _mm256_sub_epi64(hash_low_vec.ymm, prime_vec.ymm),
            _mm256_cmpgt_epi64(hash_low_vec.ymm, prime_vec.ymm));
        hash_high_vec.ymm = _mm256_blendv_epi8( //
            hash_high_vec.ymm, _mm256_sub_epi64(hash_high_vec.ymm, prime_vec.ymm),
            _mm256_cmpgt_epi64(hash_high_vec.ymm, prime_vec.ymm));
    }

    // 5. Compute the hash mix, that will be used to index into the fingerprint.
    //    This includes a serial step at the end.
    hash_low_vec.ymm = _mm256_mul_epu64(hash_low_vec.ymm, golden_ratio_vec.ymm);
    hash_high_vec.ymm = _mm256_mul_epu64(hash_high_vec.ymm, golden_ratio_vec.ymm);
    hash_mix_vec.ymm = _mm256_xor_si256(hash_low_vec.ymm, hash_high_vec.ymm);
    callback((sz_cptr_t)text_first, window_width, hash_mix_vec.u64s[0], callback_handle);
    callback((sz_cptr_t)text_second, window_width, hash_mix_vec.u64s[1], callback_handle);
    callback((sz_cptr_t)text_third, window_width, hash_mix_vec.u64s[2], callback_handle);
    callback((sz_cptr_t)text_fourth, window_width, hash_mix_vec.u64s[3], callback_handle);

    // Now repeat that operation for the remaining characters, discarding older characters.
    sz_size_t cycle = 1;
    sz_size_t const step_mask = step - 1;
    for (; text_fourth != text_end; ++text_first, ++text_second, ++text_third, ++text_fourth, ++cycle) {
        // 0. Load again the four characters we are dropping, shift them, and subtract.
        chars_low_vec.ymm = _mm256_set_epi64x( //
            text_fourth[-window_width], text_third[-window_width], text_second[-window_width],
            text_first[-window_width]);
        chars_high_vec.ymm = _mm256_add_epi8(chars_low_vec.ymm, shift_high_vec.ymm);
        hash_low_vec.ymm =
            _mm256_sub_epi64(hash_low_vec.ymm, _mm256_mul_epu64(chars_low_vec.ymm, prime_power_low_vec.ymm));
        hash_high_vec.ymm =
            _mm256_sub_epi64(hash_high_vec.ymm, _mm256_mul_epu64(chars_high_vec.ymm, prime_power_high_vec.ymm));

        // 1. Multiply the hashes by the base.
        hash_low_vec.ymm = _mm256_mul_epu64(hash_low_vec.ymm, base_low_vec.ymm);
        hash_high_vec.ymm = _mm256_mul_epu64(hash_high_vec.ymm, base_high_vec.ymm);

        // 2. Load the four characters from `text_first`, `text_first + max_hashes_per_thread`,
        //   `text_first + max_hashes_per_thread * 2`, `text_first + max_hashes_per_thread * 3`.
        chars_low_vec.ymm = _mm256_set_epi64x(text_fourth[0], text_third[0], text_second[0], text_first[0]);
        chars_high_vec.ymm = _mm256_add_epi8(chars_low_vec.ymm, shift_high_vec.ymm);

        // 3. Add the incoming characters.
        hash_low_vec.ymm = _mm256_add_epi64(hash_low_vec.ymm, chars_low_vec.ymm);
        hash_high_vec.ymm = _mm256_add_epi64(hash_high_vec.ymm, chars_high_vec.ymm);

        // 4. Compute the modulo. Assuming there are only 59 values between our prime
        //    and the 2^64 value, we can simply compute the modulo by conditionally subtracting the prime.
        hash_low_vec.ymm = _mm256_blendv_epi8( //
            hash_low_vec.ymm, _mm256_sub_epi64(hash_low_vec.ymm, prime_vec.ymm),
            _mm256_cmpgt_epi64(hash_low_vec.ymm, prime_vec.ymm));
        hash_high_vec.ymm = _mm256_blendv_epi8( //
            hash_high_vec.ymm, _mm256_sub_epi64(hash_high_vec.ymm, prime_vec.ymm),
            _mm256_cmpgt_epi64(hash_high_vec.ymm, prime_vec.ymm));

        // 5. Compute the hash mix, that will be used to index into the fingerprint.
        //    This includes a serial step at the end.
        hash_low_vec.ymm = _mm256_mul_epu64(hash_low_vec.ymm, golden_ratio_vec.ymm);
        hash_high_vec.ymm = _mm256_mul_epu64(hash_high_vec.ymm, golden_ratio_vec.ymm);
        hash_mix_vec.ymm = _mm256_xor_si256(hash_low_vec.ymm, hash_high_vec.ymm);
        if ((cycle & step_mask) == 0) {
            callback((sz_cptr_t)text_first, window_width, hash_mix_vec.u64s[0], callback_handle);
            callback((sz_cptr_t)text_second, window_width, hash_mix_vec.u64s[1], callback_handle);
            callback((sz_cptr_t)text_third, window_width, hash_mix_vec.u64s[2], callback_handle);
            callback((sz_cptr_t)text_fourth, window_width, hash_mix_vec.u64s[3], callback_handle);
        }
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

/*  AVX512 implementation of the string hashing algorithms for Skylake and newer CPUs.
 *  Includes extensions: F, CD, ER, PF, VL, DQ, BW.
 *
 *  This is the "starting level" for the advanced algorithms using K-mask registers on x86.
 */
#pragma region Skylake Implementation
#if SZ_USE_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_SKYLAKE
#pragma endregion // Skylake Implementation

/*  AVX512 implementation of the string search algorithms for Ice Lake and newer CPUs.
 *  Includes extensions:
 *      - 2017 Skylake: F, CD, ER, PF, VL, DQ, BW,
 *      - 2018 CannonLake: IFMA, VBMI,
 *      - 2019 Ice Lake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES.
 */
#pragma region Ice Lake Implementation
#if SZ_USE_ICE
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "bmi", "bmi2")
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2"))), \
                             apply_to = function)

SZ_PUBLIC void sz_hashes_ice(sz_cptr_t start, sz_size_t length, sz_size_t window_width, sz_size_t step, //
                             sz_hash_callback_t callback, void *callback_handle) {

    if (length < window_width || !window_width) return;
    if (length < 4 * window_width) {
        sz_hashes_serial(start, length, window_width, step, callback, callback_handle);
        return;
    }

    // Using AVX2, we can perform 4 long integer multiplications and additions within one register.
    // So let's slice the entire string into 4 overlapping windows, to slide over them in parallel.
    sz_size_t const max_hashes = length - window_width + 1;
    sz_size_t const min_hashes_per_thread = max_hashes / 4; // At most one sequence can overlap between 2 threads.
    sz_u8_t const *text_first = (sz_u8_t const *)start;
    sz_u8_t const *text_second = text_first + min_hashes_per_thread;
    sz_u8_t const *text_third = text_first + min_hashes_per_thread * 2;
    sz_u8_t const *text_fourth = text_first + min_hashes_per_thread * 3;
    sz_u8_t const *text_end = text_first + length;

    // Broadcast the global constants into the registers.
    // Both high and low hashes will work with the same prime and golden ratio.
    sz_u512_vec_t prime_vec, golden_ratio_vec;
    prime_vec.zmm = _mm512_set1_epi64(SZ_U64_MAX_PRIME);
    golden_ratio_vec.zmm = _mm512_set1_epi64(11400714819323198485ull);

    // Prepare the `prime ^ window_width` values, that we are going to use for modulo arithmetic.
    sz_u64_t prime_power_low = 1, prime_power_high = 1;
    for (sz_size_t i = 0; i + 1 < window_width; ++i)
        prime_power_low = (prime_power_low * 31ull) % SZ_U64_MAX_PRIME,
        prime_power_high = (prime_power_high * 257ull) % SZ_U64_MAX_PRIME;

    // We will be evaluating 4 offsets at a time with 2 different hash functions.
    // We can fit all those 8 state variables in each of the following ZMM registers.
    sz_u512_vec_t base_vec, prime_power_vec, shift_vec;
    base_vec.zmm = _mm512_set_epi64(31ull, 31ull, 31ull, 31ull, 257ull, 257ull, 257ull, 257ull);
    shift_vec.zmm = _mm512_set_epi64(0ull, 0ull, 0ull, 0ull, 77ull, 77ull, 77ull, 77ull);
    prime_power_vec.zmm = _mm512_set_epi64(prime_power_low, prime_power_low, prime_power_low, prime_power_low,
                                           prime_power_high, prime_power_high, prime_power_high, prime_power_high);

    // Compute the initial hash values for every one of the four windows.
    sz_u512_vec_t hash_vec, chars_vec;
    hash_vec.zmm = _mm512_setzero_si512();
    for (sz_u8_t const *prefix_end = text_first + window_width; text_first < prefix_end;
         ++text_first, ++text_second, ++text_third, ++text_fourth) {

        // 1. Multiply the hashes by the base.
        hash_vec.zmm = _mm512_mullo_epi64(hash_vec.zmm, base_vec.zmm);

        // 2. Load the four characters from `text_first`, `text_first + max_hashes_per_thread`,
        //   `text_first + max_hashes_per_thread * 2`, `text_first + max_hashes_per_thread * 3`...
        chars_vec.zmm = _mm512_set_epi64(text_fourth[0], text_third[0], text_second[0], text_first[0], //
                                         text_fourth[0], text_third[0], text_second[0], text_first[0]);
        chars_vec.zmm = _mm512_add_epi8(chars_vec.zmm, shift_vec.zmm);

        // 3. Add the incoming characters.
        hash_vec.zmm = _mm512_add_epi64(hash_vec.zmm, chars_vec.zmm);

        // 4. Compute the modulo. Assuming there are only 59 values between our prime
        //    and the 2^64 value, we can simply compute the modulo by conditionally subtracting the prime.
        hash_vec.zmm = _mm512_mask_blend_epi8(_mm512_cmpgt_epi64_mask(hash_vec.zmm, prime_vec.zmm), hash_vec.zmm,
                                              _mm512_sub_epi64(hash_vec.zmm, prime_vec.zmm));
    }

    // 5. Compute the hash mix, that will be used to index into the fingerprint.
    //    This includes a serial step at the end.
    sz_u512_vec_t hash_mix_vec;
    hash_mix_vec.zmm = _mm512_mullo_epi64(hash_vec.zmm, golden_ratio_vec.zmm);
    hash_mix_vec.ymms[0] = _mm256_xor_si256(_mm512_extracti64x4_epi64(hash_mix_vec.zmm, 1), //
                                            _mm512_extracti64x4_epi64(hash_mix_vec.zmm, 0));

    callback((sz_cptr_t)text_first, window_width, hash_mix_vec.u64s[0], callback_handle);
    callback((sz_cptr_t)text_second, window_width, hash_mix_vec.u64s[1], callback_handle);
    callback((sz_cptr_t)text_third, window_width, hash_mix_vec.u64s[2], callback_handle);
    callback((sz_cptr_t)text_fourth, window_width, hash_mix_vec.u64s[3], callback_handle);

    // Now repeat that operation for the remaining characters, discarding older characters.
    sz_size_t cycle = 1;
    sz_size_t step_mask = step - 1;
    for (; text_fourth != text_end; ++text_first, ++text_second, ++text_third, ++text_fourth, ++cycle) {
        // 0. Load again the four characters we are dropping, shift them, and subtract.
        chars_vec.zmm = _mm512_set_epi64(text_fourth[-window_width], text_third[-window_width],
                                         text_second[-window_width], text_first[-window_width], //
                                         text_fourth[-window_width], text_third[-window_width],
                                         text_second[-window_width], text_first[-window_width]);
        chars_vec.zmm = _mm512_add_epi8(chars_vec.zmm, shift_vec.zmm);
        hash_vec.zmm = _mm512_sub_epi64(hash_vec.zmm, _mm512_mullo_epi64(chars_vec.zmm, prime_power_vec.zmm));

        // 1. Multiply the hashes by the base.
        hash_vec.zmm = _mm512_mullo_epi64(hash_vec.zmm, base_vec.zmm);

        // 2. Load the four characters from `text_first`, `text_first + max_hashes_per_thread`,
        //   `text_first + max_hashes_per_thread * 2`, `text_first + max_hashes_per_thread * 3`.
        chars_vec.zmm = _mm512_set_epi64(text_fourth[0], text_third[0], text_second[0], text_first[0], //
                                         text_fourth[0], text_third[0], text_second[0], text_first[0]);
        chars_vec.zmm = _mm512_add_epi8(chars_vec.zmm, shift_vec.zmm);

        // ... and prefetch the next four characters into Level 2 or higher.
        _mm_prefetch((sz_cptr_t)text_fourth + 1, _MM_HINT_T1);
        _mm_prefetch((sz_cptr_t)text_third + 1, _MM_HINT_T1);
        _mm_prefetch((sz_cptr_t)text_second + 1, _MM_HINT_T1);
        _mm_prefetch((sz_cptr_t)text_first + 1, _MM_HINT_T1);

        // 3. Add the incoming characters.
        hash_vec.zmm = _mm512_add_epi64(hash_vec.zmm, chars_vec.zmm);

        // 4. Compute the modulo. Assuming there are only 59 values between our prime
        //    and the 2^64 value, we can simply compute the modulo by conditionally subtracting the prime.
        hash_vec.zmm = _mm512_mask_blend_epi8(_mm512_cmpgt_epi64_mask(hash_vec.zmm, prime_vec.zmm), hash_vec.zmm,
                                              _mm512_sub_epi64(hash_vec.zmm, prime_vec.zmm));

        // 5. Compute the hash mix, that will be used to index into the fingerprint.
        //    This includes a serial step at the end.
        hash_mix_vec.zmm = _mm512_mullo_epi64(hash_vec.zmm, golden_ratio_vec.zmm);
        hash_mix_vec.ymms[0] = _mm256_xor_si256(_mm512_extracti64x4_epi64(hash_mix_vec.zmm, 1), //
                                                _mm512_castsi512_si256(hash_mix_vec.zmm));

        if ((cycle & step_mask) == 0) {
            callback((sz_cptr_t)text_first, window_width, hash_mix_vec.u64s[0], callback_handle);
            callback((sz_cptr_t)text_second, window_width, hash_mix_vec.u64s[1], callback_handle);
            callback((sz_cptr_t)text_third, window_width, hash_mix_vec.u64s[2], callback_handle);
            callback((sz_cptr_t)text_fourth, window_width, hash_mix_vec.u64s[3], callback_handle);
        }
    }
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

/*  Implementation of the string hashing algorithms using the Arm NEON instruction set, available on 64-bit
 *  Arm processors. Covers billions of mobile CPUs worldwide, including Apple's A-series, and Qualcomm's Snapdragon.
 */
#pragma region NEON Implementation
#if SZ_USE_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_NEON
#pragma endregion // NEON Implementation

/*  Implementation of the string search algorithms using the Arm SVE variable-length registers,
 *  available in Arm v9 processors, like in Apple M4+ and Graviton 3+ CPUs.
 */
#pragma region SVE Implementation
#if SZ_USE_SVE
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_SVE
#pragma endregion // SVE Implementation

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC void sz_hashes(sz_cptr_t text, sz_size_t length, sz_size_t window_width, sz_size_t window_step, //
                          sz_hash_callback_t callback, void *callback_handle) {
#if SZ_USE_ICE
    sz_hashes_ice(text, length, window_width, window_step, callback, callback_handle);
#elif SZ_USE_HASWELL
    sz_hashes_haswell(text, length, window_width, window_step, callback, callback_handle);
#else
    sz_hashes_serial(text, length, window_width, window_step, callback, callback_handle);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLAS_FINGERPRINT_HPP_
#endif