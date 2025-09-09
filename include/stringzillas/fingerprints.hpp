/**
 *  @brief  Hardware-accelerated Min-Hash fingerprinting for string collections.
 *  @file   fingerprints.hpp
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
 *  The significand of a `f64_t` can store at least 52 bits worth of unique values, and the latencies of
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
#ifndef STRINGZILLAS_FINGERPRINTS_HPP_
#define STRINGZILLAS_FINGERPRINTS_HPP_

#include "stringzilla/types.hpp"  // `sz::error_cost_t`
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
 *  @tparam hash_type_ Type of the hash value, e.g., `u64_t`.
 */
template <typename hash_type_ = u64_t>
struct multiplying_rolling_hasher {
    using state_t = hash_type_;
    using hash_t = hash_type_;

    explicit multiplying_rolling_hasher(size_t window_width, hash_t multiplier = static_cast<hash_t>(257)) noexcept
        : window_width_ {window_width}, multiplier_ {multiplier}, highest_power_ {1} {

        sz_assert_(window_width_ > 1 && "Window width must be > 1");
        sz_assert_(multiplier_ > 0 && "Multiplier must be positive");

        for (size_t i = 0; i + 1 < window_width_; ++i) highest_power_ = highest_power_ * multiplier_;
    }

    constexpr size_t window_width() const noexcept { return window_width_; }

    constexpr state_t push(state_t state, byte_t new_char) const noexcept { return state * multiplier_ + new_char; }

    constexpr state_t roll(state_t state, byte_t old_char, byte_t new_char) const noexcept {
        state_t const without_head = state - old_char * highest_power_;
        return without_head * multiplier_ + new_char;
    }

    constexpr hash_t digest(state_t const state) const noexcept { return state; }

  private:
    size_t window_width_;
    state_t multiplier_;
    state_t highest_power_;
};

/**
 *  @brief Rabin-Karp-style rolling polynomial hash function.
 *  @tparam hash_type_ Type of the hash value, can be `u16`, `u32`, or `u64`.
 *  @tparam accumulator_type_ Type used for modulo arithmetic, e.g., `u64_t`.
 *
 *  Barrett's reduction can be used to avoid overflow in the multiplication and modulo operations.
 *  That, however, is quite tricky and computationally expensive, so this algorithm is provided merely
 *  as a baseline for retrieval benchmarks.
 *  @sa `multiplying_rolling_hasher`
 */
template <typename hash_type_ = u32_t, typename accumulator_type_ = u64_t>
struct rabin_karp_rolling_hasher {
    using hash_t = hash_type_;
    using state_t = accumulator_type_;

    static_assert(is_same_type<hash_t, u16_t>::value || is_same_type<hash_t, u32_t>::value ||
                      is_same_type<hash_t, u64_t>::value,
                  "Unsupported hash type");

    static constexpr hash_t default_alphabet_size_k = 256u;
    static constexpr hash_t default_modulo_base_k = //
        is_same_type<hash_t, u16_t>::value   ? SZ_U16_MAX_PRIME
        : is_same_type<hash_t, u32_t>::value ? SZ_U32_MAX_PRIME
                                             : SZ_U64_MAX_PRIME;

    constexpr rabin_karp_rolling_hasher() noexcept
        : window_width_ {0}, modulo_ {default_modulo_base_k}, multiplier_ {default_alphabet_size_k},
          discarding_multiplier_ {1} {}

    constexpr explicit rabin_karp_rolling_hasher(    //
        size_t window_width,                         //
        hash_t multiplier = default_alphabet_size_k, //
        hash_t modulo = default_modulo_base_k) noexcept
        : window_width_ {window_width}, modulo_ {modulo}, multiplier_ {multiplier}, discarding_multiplier_ {1} {

        sz_assert_(window_width_ > 1 && "Window width must be > 1");
        sz_assert_(multiplier_ > 0 && "Multiplier must be positive");
        sz_assert_(modulo_ > 1 && "Modulo base must be > 1");

        for (size_t i = 0; i + 1 < window_width_; ++i)
            discarding_multiplier_ = mul_mod(discarding_multiplier_, multiplier_);
    }

    constexpr size_t window_width() const noexcept { return window_width_; }

    constexpr state_t push(state_t state, byte_t new_char) const noexcept {
        state_t new_term = static_cast<state_t>(new_char + 1u);
        return add_mod(mul_mod(state, multiplier_), new_term);
    }

    constexpr state_t roll(state_t state, byte_t old_char, byte_t new_char) const noexcept {
        state_t old_term = static_cast<state_t>(old_char + 1u);
        state_t new_term = static_cast<state_t>(new_char + 1u);

        state_t without_old = sub_mod(state, mul_mod(old_term, discarding_multiplier_));
        state_t with_new = add_mod(mul_mod(without_old, multiplier_), new_term);
        return with_new;
    }

    constexpr hash_t digest(state_t state) const noexcept { return static_cast<hash_t>(state); }

  private:
    constexpr state_t mul_mod(state_t a, state_t b) const noexcept { return (a * b) % modulo_; }
    constexpr state_t add_mod(state_t a, state_t b) const noexcept { return (a + b) % modulo_; }
    constexpr state_t sub_mod(state_t a, state_t b) const noexcept { return (a + modulo_ - b) % modulo_; }

    size_t window_width_;
    state_t modulo_;
    state_t multiplier_;
    state_t discarding_multiplier_;
};

/**
 *  @brief BuzHash rolling hash function leveraging a fixed-size lookup table and bitwise operations.
 *  @tparam hash_type_ Type of the hash value, e.g., `u64_t`.
 *  @sa `multiplying_rolling_hasher`, `rabin_karp_rolling_hasher`
 */
template <typename hash_type_ = u64_t>
struct buz_rolling_hasher {
    using state_t = hash_type_;
    using hash_t = hash_type_;

    explicit buz_rolling_hasher(size_t window_width, u64_t seed = 0x9E3779B97F4A7C15ull) noexcept
        : window_width_ {window_width} {

        sz_assert_(window_width_ > 1 && "Window width must be > 1");
        for (size_t i = 0; i < 256; ++i) table_[i] = split_mix64(seed);
    }

    constexpr size_t window_width() const noexcept { return window_width_; }

    constexpr state_t push(state_t state, byte_t new_char) const noexcept {
        return rotl(state, 1) ^ table_[new_char & 0xFFu];
    }

    constexpr state_t roll(state_t state, byte_t old_char, byte_t new_char) const noexcept {
        constexpr unsigned bits_k = sizeof(state_t) * 8u;
        state_t const rolled = rotl(state, 1);
        state_t const remove_term = rotl(table_[old_char & 0xFFu], window_width_ & (bits_k - 1u));
        return rolled ^ remove_term ^ table_[new_char & 0xFFu];
    }

    constexpr hash_t digest(state_t state) const noexcept { return state; }

  private:
    static constexpr state_t rotl(state_t v, unsigned r) noexcept {
        constexpr unsigned bits_k = sizeof(state_t) * 8u;
        return (v << r) | (v >> (bits_k - r));
    }

    static constexpr u64_t split_mix64(u64_t &state) noexcept {
        state += 0x9E3779B97F4A7C15ull;
        u64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    size_t window_width_;
    state_t table_[256];
};

/**
 *  @brief Helper function to pick the second co-prime "modulo" base for the Karp-Rabin rolling hashes.
 *  @retval 0 on failure, or a valid prime number otherwise.
 */
inline u64_t choose_coprime_modulo(u64_t multiplier, u64_t limit) noexcept {
    if (multiplier == 0 || multiplier >= limit || limit <= 1) return 0;

    // Upper bound guaranteeing no overflow in non-discarding `update` calls
    u64_t max_input = std::numeric_limits<byte_t>::max() + 1u;
    u64_t bound = (limit - (max_input + 1)) / multiplier + 1;

    if (!(bound & 1u)) --bound; // Make odd

    for (u64_t p = bound; p >= 3; p -= 2)
        if (std::gcd(p, multiplier) == 1) return p;

    return 0;
}

template <typename state_type_ = f32_t>
struct floating_rolling_hasher;

/**
 *  @brief Rabin-Karp-style Rolling hash function for single-precision floating-point numbers.
 *  @tparam state_type_ Type of the floating-point number, e.g., `f32_t`.
 *
 *  The IEEE 754 single-precision `f32_t` has a 24-bit significand (23 explicit bits + 1 implicit bit).
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
 *  ! So the `floating_rolling_hasher<f32_t>` should only be used for exploratory purposes & testing.
 *
 *  @sa `floating_rolling_hasher<f64_t>` for 52 bit variant.
 */
template <>
struct floating_rolling_hasher<f32_t> {
    using state_t = f32_t;
    using hash_t = u32_t;

    /** @brief The largest integer exactly representable as a float. */
    static constexpr state_t limit_k = 8'388'607.0f;

    /** @brief The typical size of the alphabet - the 256 possible values of a single byte. */
    static constexpr hash_t default_alphabet_size_k = 256u;

    /** @brief The largest prime, that multiplied by `default_alphabet_size_k` and added a term - stays `limit_k`. */
    static constexpr hash_t default_modulo_base_k = 8123u;

    explicit floating_rolling_hasher(                      //
        size_t const window_width,                         //
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
        for (size_t i = 0; i + 1 < window_width_; ++i)
            negative_discarding_multiplier_ = ::fmodf(negative_discarding_multiplier_ * multiplier_, modulo_);
        negative_discarding_multiplier_ = -negative_discarding_multiplier_;
    }

    SZ_INLINE size_t window_width() const noexcept { return window_width_; }

    SZ_INLINE state_t push(state_t state, byte_t new_char) const noexcept {
        state_t new_term = state_t(new_char) + 1.0f;
        return fma_mod(state, multiplier_, new_term);
    }

    SZ_INLINE state_t roll(state_t state, byte_t old_char, byte_t new_char) const noexcept {

        state_t old_term = state_t(old_char) + 1.0f;
        state_t new_term = state_t(new_char) + 1.0f;

        state_t without_old = fma_mod(negative_discarding_multiplier_, old_term, state);
        return fma_mod(without_old, multiplier_, new_term);
    }

    SZ_INLINE hash_t digest(state_t state) const noexcept { return static_cast<hash_t>(state); }

  private:
    SZ_INLINE state_t fma_mod(state_t a, state_t b, state_t c) const noexcept { return barrett_mod(a * b + c); }

    /**
     *  @brief Barrett-style `std::fmodf` alternative to avoid overflow.
     *  @see https://en.cppreference.com/w/cpp/numeric/math/fmod
     */
    SZ_INLINE state_t barrett_mod(state_t x) const noexcept {

        state_t q = std::floor(x * inverse_modulo_);
        state_t result = x - q * modulo_;

        // Clamp into the [0, modulo_) range.
        if (result >= modulo_) result -= modulo_;
        if (result < 0.0f) result += modulo_;

        sz_assert_(result >= 0 && "Intermediate x underflows the zero");
        sz_assert_(result < limit_k && "Intermediate x overflows the limit");
        sz_assert_(static_cast<u64_t>(::fmodf(x, modulo_) + (::fmodf(x, modulo_) < 0.0f ? modulo_ : 0.0f)) ==
                       static_cast<u64_t>(result) &&
                   "Floating point modulo was incorrect");

        return result;
    }

    size_t window_width_ = 0;
    state_t multiplier_ = 0.0f;
    state_t modulo_ = 0.0f;
    state_t inverse_modulo_ = 0.0f;
    state_t negative_discarding_multiplier_ = 0.0f;
};

inline f64_t absolute_fmod(f64_t x, f64_t y) noexcept {
    f64_t result = std::fmod(x, y);
    return result < 0.0 ? result + y : result;
}

inline u64_t absolute_umod(f64_t x, f64_t y) noexcept { return static_cast<u64_t>(absolute_fmod(x, y)); }

/**
 * @brief Constexpr-compatible `std::floor`-like function for C++17, based on IEEE 754 bit manipulation.
 * @param[in] x The double-precision floating-point number to floor.
 * @return The largest integer value not greater than x.
 */
inline constexpr f64_t constexpr_floor(f64_t x) noexcept {
    // Use a union to access the bit representation of the double
    union ieee754_double {
        f64_t value;
        u64_t bits;
    };

    ieee754_double number = {x};

    // Extract the exponent: bits 52-62, biased by 1023
    i32_t exponent = static_cast<i32_t>((number.bits >> 52) & 0x7FF) - 1023;

    // If exponent < 0, then |x| < 1
    if (exponent < 0) {
        // Return 0 for positive numbers, -1 for negative numbers with fractional part
        if (static_cast<i64_t>(number.bits) >= 0) { return 0.0; }             // Positive number less than 1
        else if ((number.bits & 0x7FFFFFFFFFFFFFFFULL) != 0) { return -1.0; } // Negative number with fractional part
        return x;                                                             // Exactly 0 or -0
    }

    // If exponent >= 52, all bits represent the integer part (no fractional bits)
    if (exponent >= 52) return x; // Already an integer (or infinity/NaN)

    // Calculate which bits represent the fractional part
    u64_t fractional_mask = 0x000FFFFFFFFFFFFFULL >> exponent;

    // If no fractional bits are set, x is already an integer
    if ((number.bits & fractional_mask) == 0) return x;

    // For negative numbers, add 1 to the integer part before truncating
    if (static_cast<i64_t>(number.bits) < 0) number.bits += (0x0010000000000000ULL >> exponent);

    // Clear the fractional bits
    number.bits &= ~fractional_mask;
    return number.value;
}

/**
 *  @brief Rabin-Karp-style Rolling hash function for f64_t-precision floating-point numbers.
 *  @tparam state_type_ Type of the floating-point number, e.g., `f32_t`.
 *
 *  The IEEE 754 f64_t-precision `f32_t` has a 53-bit significand (52 explicit bits + 1 implicit bit).
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
 *  @sa `rabin_karp_rolling_hasher<u32_t, u64_t>` integer implementation for small modulo variants.
 *  @sa `floating_rolling_hasher<f32_t>` for a lower-resolution hash.
 */
template <>
struct floating_rolling_hasher<f64_t> {
    using state_t = f64_t;
    using hash_t = u64_t;

    static constexpr state_t limit_k = 4503599627370495.0;
    static constexpr hash_t default_alphabet_size_k = 256u;
    static constexpr hash_t default_modulo_base_k = 4503599626977u;

    explicit floating_rolling_hasher(                       //
        size_t const window_width,                          //
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

        for (size_t i = 0; i + 1 < window_width_; ++i)
            negative_discarding_multiplier_ = std::fmod(negative_discarding_multiplier_ * multiplier_, modulo_);
        negative_discarding_multiplier_ = -negative_discarding_multiplier_;
    }

    constexpr floating_rolling_hasher() noexcept = default;
    constexpr floating_rolling_hasher(floating_rolling_hasher &&) noexcept = default;
    constexpr floating_rolling_hasher(floating_rolling_hasher const &) noexcept = default;
    constexpr floating_rolling_hasher &operator=(floating_rolling_hasher &&) noexcept = default;
    constexpr floating_rolling_hasher &operator=(floating_rolling_hasher const &) noexcept = default;

    constexpr size_t window_width() const noexcept { return window_width_; }

    constexpr state_t push(state_t state, byte_t new_char) const noexcept {
        state_t new_term = state_t(new_char) + 1.0;
        return fma_mod(state, multiplier_, new_term);
    }

    constexpr state_t roll(state_t state, byte_t old_char, byte_t new_char) const noexcept {
        state_t old_term = state_t(old_char) + 1.0;
        state_t new_term = state_t(new_char) + 1.0;
        state_t without_old = fma_mod(negative_discarding_multiplier_, old_term, state);
        return fma_mod(without_old, multiplier_, new_term);
    }

    constexpr hash_t digest(state_t state) const noexcept { return static_cast<hash_t>(state); }

    constexpr state_t multiplier() const noexcept { return multiplier_; }
    constexpr state_t modulo() const noexcept { return modulo_; }
    constexpr state_t inverse_modulo() const noexcept { return inverse_modulo_; }
    constexpr state_t negative_discarding_multiplier() const noexcept { return negative_discarding_multiplier_; }

  private:
    constexpr state_t fma_mod(state_t a, state_t b, state_t c) const noexcept { return barrett_mod(a * b + c); }

    /**
     *  @brief Barrett-style `std::fmod` alternative to avoid overflow.
     *  @see https://en.cppreference.com/w/cpp/numeric/math/fmod
     */
    constexpr state_t barrett_mod(state_t x) const noexcept {

        state_t q = constexpr_floor(x * inverse_modulo_);
        state_t result = x - q * modulo_;

        // Clamp into the [0, modulo_) range.
        if (result >= modulo_) result -= modulo_;
        if (result < 0.0) result += modulo_;

        // Skip debug assertions that call non-constexpr functions:
        // sz_assert_(static_cast<u64_t>(absolute_fmod(x, modulo_)) == static_cast<u64_t>(result) &&
        //            "Floating point modulo was incorrect");
        sz_assert_(result >= 0 && "Intermediate x underflows the zero");
        sz_assert_(result < limit_k && "Intermediate x overflows the limit");
        return result;
    }

    size_t window_width_ = 0;
    state_t multiplier_ = 0.0;
    state_t modulo_ = 0.0;
    state_t inverse_modulo_ = 0.0;
    state_t negative_discarding_multiplier_ = 0.0;
};

#pragma endregion - Baseline Rolling Hashers

#pragma region - Optimized Rolling MinHashers

template <size_t dimensions_ = SZ_SIZE_MAX, typename hash_type_ = u32_t, typename count_type_ = u32_t>
void merge_count_min_sketches(                                                                           //
    span<hash_type_ const, dimensions_> a_min_hashes, span<count_type_ const, dimensions_> a_min_counts, //
    span<hash_type_ const, dimensions_> b_min_hashes, span<count_type_ const, dimensions_> b_min_counts, //
    span<hash_type_, dimensions_> c_min_hashes, span<count_type_, dimensions_> c_min_counts) noexcept {

    sz_assert_(a_min_hashes.size() == b_min_hashes.size() && "Input sketches must have the same size");
    sz_assert_(a_min_counts.size() == b_min_counts.size() && "Input counts must have the same size");
    sz_assert_(c_min_hashes.size() == a_min_hashes.size() && "Output hashes must have the same size");
    sz_assert_(c_min_counts.size() == a_min_counts.size() && "Output counts must have the same size");

    for (size_t dim = 0; dim < c_min_hashes.size(); ++dim) {
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
template <                                                           //
    typename hasher_type_ = rabin_karp_rolling_hasher<u32_t, u64_t>, //
    typename min_hash_type_ = u32_t,                                 //
    typename min_count_type_ = u32_t,                                //
    typename allocator_type_ = std::allocator<hasher_type_>,         //
    sz_capability_t capability_ = sz_cap_serial_k                    //
    >
struct basic_rolling_hashers;

template <                    //
    typename hasher_type_,    //
    typename min_hash_type_,  //
    typename min_count_type_, //
    typename allocator_type_  //
    >
struct basic_rolling_hashers<hasher_type_, min_hash_type_, min_count_type_, allocator_type_, sz_cap_serial_k> {

    using hasher_t = hasher_type_;
    using rolling_state_t = typename hasher_t::state_t;
    using rolling_hash_t = typename hasher_t::hash_t;

    using min_hash_t = min_hash_type_;
    using min_count_t = min_count_type_;
    using allocator_t = allocator_type_;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;

    static constexpr rolling_state_t skipped_rolling_state_k = std::numeric_limits<rolling_state_t>::max();
    static constexpr rolling_hash_t skipped_rolling_hash_k = std::numeric_limits<rolling_hash_t>::max();
    static constexpr min_hash_t max_hash_k = std::numeric_limits<min_hash_t>::max();

  private:
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using hasher_allocator_t = typename allocator_traits_t::template rebind_alloc<hasher_t>;
    using rolling_states_allocator_t = typename allocator_traits_t::template rebind_alloc<rolling_state_t>;
    using rolling_hashes_allocator_t = typename allocator_traits_t::template rebind_alloc<rolling_hash_t>;
    using min_counts_allocator_t = typename allocator_traits_t::template rebind_alloc<min_count_t>;

    allocator_t allocator_;
    safe_vector<hasher_t, hasher_allocator_t> hashers_;
    size_t max_window_width_ = 0;

  public:
    basic_rolling_hashers(allocator_t allocator = {}) noexcept
        : allocator_(std::move(allocator)),
          hashers_(allocator_traits_t::select_on_container_copy_construction(allocator)) {}

    size_t dimensions() const noexcept { return hashers_.size(); }
    size_t max_window_width() const noexcept { return max_window_width_; }
    size_t window_width(size_t dim) const noexcept { return hashers_[dim].window_width(); }

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
     *  basic_rolling_hashers<rabin_karp_rolling_hasher<u32_t>> hashers;
     *  hashers.try_extend(3, 32); // 32 dims for 3-grams
     *  hashers.try_extend(5, 32); // 32 dims for 5-grams
     *  hashers.try_extend(7, 64); // 64 dims for 7-grams
     *  std::array<u32_t, 128> fingerprint; // 128 total dims
     *  hashers("some text", fingerprint);
     *  @endcode
     */
    SZ_NOINLINE status_t try_extend(size_t window_width, size_t new_dims, size_t alphabet_size = 256) noexcept {
        size_t const old_dims = hashers_.size();
        if (hashers_.try_reserve(old_dims + new_dims) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t new_dim = 0; new_dim < new_dims; ++new_dim) {
            size_t const dim = old_dims + new_dim;
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
    SZ_NOINLINE status_t try_append(hasher_t hasher) noexcept {
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
    SZ_NOINLINE status_t try_fingerprint(         //
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
    SZ_NOINLINE void fingerprint_chunk(                     //
        span<byte_t const> text_chunk,                      //
        span<rolling_state_t, dimensions_> last_states,     //
        span<rolling_hash_t, dimensions_> rolling_minimums, //
        span<min_hash_t, dimensions_> min_hashes,           //
        span<min_count_t, dimensions_> min_counts,          //
        size_t const passed_progress = 0) const noexcept {

        sz_assert_(dimensions() == last_states.size() && "Dimensions number & states number mismatch");
        sz_assert_(dimensions() == rolling_minimums.size() && "Dimensions number & minimums number mismatch");
        sz_assert_(dimensions() == min_hashes.size() && "Dimensions number & min-hashes number mismatch");
        sz_assert_(dimensions() == min_counts.size() && "Dimensions number & hash-counts number mismatch");

        // Until we reach the maximum window length, use a branching code version
        size_t const prefix_length = (std::min)(text_chunk.size(), max_window_width_);
        size_t new_char_offset = passed_progress;
        for (; new_char_offset < prefix_length; ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            for (size_t dim = 0; dim < last_states.size(); ++dim) {
                hasher_t const &hasher = hashers_[dim];
                rolling_state_t &last_state = last_states[dim];
                rolling_hash_t &rolling_minimum = rolling_minimums[dim];
                min_count_t &min_count = min_counts[dim];
                if (new_char_offset < hasher.window_width()) {
                    last_state = hasher.push(last_state, new_char);
                    if (hasher.window_width() == (new_char_offset + 1)) {
                        rolling_minimum = (std::min)(rolling_minimum, hasher.digest(last_state));
                        min_count = 1; // First occurrence of this hash
                    }
                    continue;
                }
                auto const old_char = text_chunk[new_char_offset - hasher.window_width()];
                last_state = hasher.roll(last_state, old_char, new_char);
                rolling_hash_t new_hash = hasher.digest(last_state);
                min_count *= new_hash >= rolling_minimum; // ? Discard `min_count` to 0 for new extremums
                min_count += new_hash <= rolling_minimum; // ? Increments by 1 for new & old minimums
                rolling_minimum = (std::min)(rolling_minimum, new_hash);
            }
        }

        // Now we can avoid a branch in the nested loop, as we are passed the longest window width
        for (; new_char_offset < text_chunk.size(); ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            for (size_t dim = 0; dim < last_states.size(); ++dim) {
                hasher_t const &hasher = hashers_[dim];
                rolling_state_t &last_state = last_states[dim];
                rolling_hash_t &rolling_minimum = rolling_minimums[dim];
                min_count_t &min_count = min_counts[dim];
                auto const old_char = text_chunk[new_char_offset - hasher.window_width()];
                last_state = hasher.roll(last_state, old_char, new_char);
                rolling_hash_t new_hash = hasher.digest(last_state);
                min_count *= new_hash >= rolling_minimum; // ? Discard `min_count` to 0 for new extremums
                min_count += new_hash <= rolling_minimum; // ? Increments by 1 for new & old minimums
                rolling_minimum = (std::min)(rolling_minimum, new_hash);
            }
        }

        // Finally, export the minimum hashes into the smaller representations
        if (min_hashes)
            for (size_t dim = 0; dim < min_hashes.size(); ++dim) {
                rolling_hash_t const &rolling_minimum = rolling_minimums[dim];
                min_hash_t &min_hash = min_hashes[dim];
                min_hash = rolling_minimum == skipped_rolling_hash_k
                               ? max_hash_k // If the rolling minimum is not set, use the maximum hash value
                               : static_cast<min_hash_t>(rolling_minimum & max_hash_k);
            }

        // We may be in a position, when `text_chunk.size()` is smaller than the shortest window width,
        // so we must output zeros for the `min_counts` for every case, where the rolling state is skipped.
        if (min_counts)
            for (size_t dim = 0; dim < min_counts.size(); ++dim) {
                rolling_hash_t const &rolling_minimum = rolling_minimums[dim];
                min_count_t &min_count = min_counts[dim];
                min_count = rolling_minimum == skipped_rolling_hash_k
                                ? 0 // If the rolling minimum is not set, reset to zeros
                                : min_count;
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
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    SZ_NOINLINE status_t operator()(                                                                      //
        texts_type_ const &texts,                                                                         //
        min_hashes_per_text_type_ &&min_hashes_per_text, min_counts_per_text_type_ &&min_counts_per_text, //
        executor_type_ &&executor = {}, cpu_specs_t specs = {}) const noexcept {

        // Depending on document sizes, choose the appropriate parallelization strategy
        // - Either split each text into chunks across threads
        // - Or split the texts themselves across threads
        size_t const text_size_threshold = executor.threads_count() * specs.l2_bytes;
        size_t const dims = dimensions();

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
        for (size_t text_index = 0; text_index < texts.size(); ++text_index) {

            auto const &text = texts[text_index];
            if (text.size() < text_size_threshold) continue;

            // Split the text into chunks of the maximum window width
            auto const text_view = to_bytes_view(text);
            size_t const chunk_size = round_up_to_multiple(                  //
                divide_round_up(text_view.size(), executor.threads_count()), //
                specs.cache_line_width);

            // Distribute overlapping chunks across threads
            executor.for_threads([&](size_t thread_index) noexcept {
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
            for (size_t dim = 0; dim < min_hashes.size(); ++dim) {
                rolling_hash_t min_hash = skipped_rolling_hash_k;
                min_count_t min_count = 0;
                for (size_t thread_index = 0; thread_index < executor.threads_count(); ++thread_index) {
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
#if SZ_HAS_CONCEPTS_
    requires executor_like<executor_type_>
#endif
SZ_NOINLINE status_t floating_rolling_hashers_in_parallel_(                                           //
    engine_type_ const &engine, texts_type_ const &texts,                                             //
    min_hashes_per_text_type_ &&min_hashes_per_text, min_counts_per_text_type_ &&min_counts_per_text, //
    executor_type_ &&executor = {}, cpu_specs_t specs = {}) noexcept {

    using engine_t = engine_type_;
    using rolling_state_t = typename engine_t::rolling_state_t;
    using min_count_t = typename engine_t::min_count_t;
    using min_hash_t = typename engine_t::min_hash_t;
    static constexpr auto dimensions_k = engine_t::dimensions_k;
    static constexpr auto skipped_rolling_hash_k = engine_t::skipped_rolling_hash_k;
    static constexpr auto max_hash_k = engine_t::max_hash_k;

    // Depending on document sizes, choose the appropriate parallelization strategy
    // - Either split each text into chunks across threads
    // - Or split the texts themselves across threads
    size_t const text_size_threshold = specs.l2_bytes * executor.threads_count();
    size_t const window_width = engine.window_width();

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
    for (size_t text_index = 0; text_index < texts.size(); ++text_index) {

        auto const &text = texts[text_index];
        if (text.size() < text_size_threshold) continue;

        // Split the text into chunks of the maximum window width
        auto text_view = to_bytes_view(text);
        size_t const chunk_size = round_up_to_multiple(                  //
            divide_round_up(text_view.size(), executor.threads_count()), //
            specs.cache_line_width);

        rolling_state_t rolling_minimums[dimensions_k];
        for (size_t dim = 0; dim < dimensions_k; ++dim) rolling_minimums[dim] = skipped_rolling_hash_k;

        // Distribute overlapping chunks across threads
        auto min_hashes = to_span(min_hashes_per_text[text_index]);
        auto min_counts = to_span(min_counts_per_text[text_index]);
        auto gather_mutex = executor.make_mutex();
        executor.for_threads([&](size_t thread_index) noexcept {
            auto text_start = text_view.data() + (std::min)(text_view.size(), thread_index * chunk_size);
            // ? This overlap will be different for different window widths, but assuming we are
            // ? computing the non-weighted Min-Hash, recomputing & comparing a few hashes for the
            // ? same slices isn't a big deal.
            auto overlapping_text_end = (std::min)(text_start + chunk_size + window_width - 1, text_view.end());
            auto thread_local_text = span<byte_t const>(text_start, overlapping_text_end);

            rolling_state_t thread_local_states[dimensions_k];
            rolling_state_t thread_local_minimums[dimensions_k];
            min_count_t thread_local_counts[dimensions_k];
            for (size_t dim = 0; dim < dimensions_k; ++dim)
                thread_local_states[dim] = 0, thread_local_minimums[dim] = skipped_rolling_hash_k;
            engine.fingerprint_chunk(thread_local_text, thread_local_states, thread_local_minimums, {},
                                     thread_local_counts);

            lock_guard lock(gather_mutex);
            for (size_t dim = 0; dim < dimensions_k; ++dim) {
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
        for (size_t dim = 0; dim < min_hashes.size(); ++dim) {
            rolling_state_t const &rolling_minimum = rolling_minimums[dim];
            min_hash_t &min_hash = min_hashes[dim];
            auto const rolling_minimum_as_uint = static_cast<u64_t>(rolling_minimum);
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
    size_t dimensions_ = 64,                       //
    typename enable_ = void                        //
    >
struct floating_rolling_hashers;

template <size_t dimensions_>
struct floating_rolling_hashers<sz_cap_serial_k, dimensions_, void> {

    using hasher_t = floating_rolling_hasher<f64_t>;
    using rolling_state_t = f64_t;
    using min_hash_t = u32_t;
    using min_count_t = u32_t;

    static constexpr size_t dimensions_k = dimensions_;
    static constexpr sz_capability_t capability_k = sz_cap_serial_k;
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
    size_t window_width_;

  public:
    constexpr size_t dimensions() const noexcept { return dimensions_k; }
    constexpr size_t window_width() const noexcept { return window_width_; }
    constexpr size_t window_width(size_t) const noexcept { return window_width_; }

    floating_rolling_hashers() noexcept {
        // Reset all variables to zeros
        for (auto &multiplier : multipliers_) multiplier = 0.0;
        for (auto &modulo : modulos_) modulo = 0.0;
        for (auto &inverse_modulo : inverse_modulos_) inverse_modulo = 0.0;
        for (auto &negative_discarding_multiplier : negative_discarding_multipliers_)
            negative_discarding_multiplier = 0.0;
        window_width_ = 0;
    }

    /**
     *  @brief Initializes several rolling hashers with different multipliers and modulos.
     *  @param[in] alphabet_size Size of the alphabet, typically 256 for UTF-8, 4 for DNA, or 20 for proteins.
     *  @param[in] first_dimension_offset The offset for the first dimension within a larger fingerprint, typically 0.
     */
    SZ_NOINLINE status_t try_seed(size_t window_width, size_t alphabet_size = 256,
                                  size_t first_dimension_offset = 0) noexcept {
        for (size_t dim = 0; dim < dimensions_k; ++dim) {
            hasher_t hasher(window_width, alphabet_size + first_dimension_offset + dim,
                            hasher_t::default_modulo_base_k);
            multipliers_[dim] = hasher.multiplier();
            modulos_[dim] = hasher.modulo();
            inverse_modulos_[dim] = hasher.inverse_modulo();
            negative_discarding_multipliers_[dim] = hasher.negative_discarding_multiplier();
        }
        window_width_ = window_width;
        return status_t::success_k;
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     */
    SZ_NOINLINE void fingerprint(span<byte_t const> text, min_hashes_span_t min_hashes,
                                 min_counts_span_t min_counts) const noexcept {

        if (text.size() < window_width_) {
            for (auto &min_hash : min_hashes) min_hash = max_hash_k;
            for (auto &min_count : min_counts) min_count = 0;
            return;
        }

        rolling_state_t rolling_states[dimensions_k];
        rolling_state_t rolling_minimums[dimensions_k];
        for (size_t dim = 0; dim < dimensions_k; ++dim)
            rolling_states[dim] = 0, rolling_minimums[dim] = skipped_rolling_hash_k;
        fingerprint_chunk(text, rolling_states, rolling_minimums, min_hashes, min_counts);
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     */
    SZ_NOINLINE status_t try_fingerprint(span<byte_t const> text, min_hashes_span_t min_hashes,
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
    SZ_NOINLINE void fingerprint_chunk(                       //
        span<byte_t const> text_chunk,                        //
        span<rolling_state_t, dimensions_k> last_states,      //
        span<rolling_state_t, dimensions_k> rolling_minimums, //
        span<min_hash_t, dimensions_k> min_hashes,            //
        span<min_count_t, dimensions_k> min_counts,           //
        size_t const passed_progress = 0) const noexcept {

        // Until we reach the maximum window length, use a branching code version
        size_t const prefix_length = (std::min)(text_chunk.size(), window_width_);
        size_t new_char_offset = passed_progress;
        for (; new_char_offset < prefix_length; ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            rolling_state_t const new_term = static_cast<rolling_state_t>(new_char) + 1.0;
            for (size_t dim = 0; dim < dimensions_k; ++dim) {
                rolling_state_t &last_state = last_states[dim];
                last_state = std::fma(last_state, multipliers_[dim], new_term); // Add head
                last_state = barrett_mod(last_state, dim);
            }
        }

        // We now have our first minimum hashes
        if (new_char_offset == window_width_)
            for (size_t dim = 0; dim < dimensions_k; ++dim)
                rolling_minimums[dim] = (std::min)(rolling_minimums[dim], last_states[dim]),
                min_counts[dim] = 1; // First occurrence of this hash

        // Now we can avoid a branch in the nested loop, as we are passed the longest window width
        for (; new_char_offset < text_chunk.size(); ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            byte_t const old_char = text_chunk[new_char_offset - window_width_];
            rolling_state_t const new_term = static_cast<rolling_state_t>(new_char) + 1.0;
            rolling_state_t const old_term = static_cast<rolling_state_t>(old_char) + 1.0;
            for (size_t dim = 0; dim < dimensions_k; ++dim) {
                rolling_state_t &last_state = last_states[dim];
                rolling_state_t &rolling_minimum = rolling_minimums[dim];
                min_count_t &min_count = min_counts[dim];

                last_state = std::fma(negative_discarding_multipliers_[dim], old_term, last_state); // Remove tail
                last_state = barrett_mod(last_state, dim);
                last_state = std::fma(last_state, multipliers_[dim], new_term); // Add head
                last_state = barrett_mod(last_state, dim);

                // In essence, we need to increment the `min_count` if the new hash is equal to the minimum,
                // or reset it to 1 if the new hash is smaller than the minimum.
                //
                //      if (rolling_minimum == last_state) { min_count++; }
                //      else if (last_state < rolling_minimum) { rolling_minimum = last_state, min_count = 1; }
                //
                // There's a branchless approach to achieve the same outcome:
                min_count *= last_state >= rolling_minimum; // ? Discard `min_count` to 0 for new extremums
                min_count += last_state <= rolling_minimum; // ? Increments by 1 for new & old minimums
                rolling_minimum = (std::min)(rolling_minimum, last_state);
            }
        }

        // Finally, export the minimum hashes into the smaller representations
        if (min_hashes)
            for (size_t dim = 0; dim < dimensions_k; ++dim) {
                rolling_state_t const &rolling_minimum = rolling_minimums[dim];
                min_hash_t &min_hash = min_hashes[dim];
                auto const rolling_minimum_as_uint = static_cast<u64_t>(rolling_minimum);
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
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    SZ_NOINLINE status_t operator()(texts_type_ const &texts, min_hashes_per_text_type_ &&min_hashes, //
                                    min_counts_per_text_type_ &&min_counts, executor_type_ &&executor = {},
                                    cpu_specs_t specs = {}) noexcept {
        return floating_rolling_hashers_in_parallel_(            //
            *this, texts,                                        //
            std::forward<min_hashes_per_text_type_>(min_hashes), //
            std::forward<min_counts_per_text_type_>(min_counts), //
            std::forward<executor_type_>(executor), specs);
    }

  private:
    inline rolling_state_t barrett_mod(rolling_state_t x, size_t dim) const noexcept {
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

/*  AVX2 implementation of the string hashing algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "fma")
#endif

SZ_INLINE __m256d _mm256_floor_magic_pd(__m256d x) noexcept {
    // Magic number rounding approach for fast floor
    __m256d magic = _mm256_set1_pd(6755399441055744.0); // 2^52 + 2^51
    __m256d rounded = _mm256_sub_pd(_mm256_add_pd(x, magic), magic);

    // Handle negative numbers: if result > x, subtract 1
    __m256d neg_mask_pd = _mm256_cmp_pd(rounded, x, _CMP_GT_OQ);
    return _mm256_sub_pd(rounded, _mm256_and_pd(neg_mask_pd, _mm256_set1_pd(1.0)));
}

/**
 *  @brief Optimized rolling Min-Hashers built around floating-point numbers.
 *  In a single YMM register we can store 4 `f64_t` values, so we can process 4 hashes per register.
 */
template <size_t dimensions_>
struct floating_rolling_hashers<sz_cap_haswell_k, dimensions_, void> {

    using hasher_t = floating_rolling_hasher<f64_t>;
    using rolling_state_t = f64_t;
    using min_hash_t = u32_t;
    using min_count_t = u32_t;

    static constexpr size_t dimensions_k = dimensions_;
    static constexpr sz_capability_t capability_k = sz_cap_haswell_k;
    static constexpr rolling_state_t skipped_rolling_hash_k = std::numeric_limits<rolling_state_t>::max();
    static constexpr min_hash_t max_hash_k = std::numeric_limits<min_hash_t>::max();

    using min_hashes_span_t = span<min_hash_t, dimensions_k>;
    using min_counts_span_t = span<min_count_t, dimensions_k>;

    static constexpr unsigned hashes_per_ymm_k = sizeof(sz_u256_vec_t) / sizeof(rolling_state_t);
    static constexpr bool has_incomplete_tail_group_k = (dimensions_k % hashes_per_ymm_k) != 0;
    static constexpr size_t aligned_dimensions_k =
        has_incomplete_tail_group_k ? (dimensions_k / hashes_per_ymm_k + 1) * hashes_per_ymm_k : (dimensions_k);
    static constexpr unsigned groups_count_k = aligned_dimensions_k / hashes_per_ymm_k;

    static_assert(dimensions_k <= 256, "Too many dimensions to keep on stack");

  private:
    rolling_state_t multipliers_[aligned_dimensions_k];
    rolling_state_t modulos_[aligned_dimensions_k];
    rolling_state_t inverse_modulos_[aligned_dimensions_k];
    rolling_state_t negative_discarding_multipliers_[aligned_dimensions_k];
    size_t window_width_;

  public:
    constexpr size_t dimensions() const noexcept { return dimensions_k; }
    constexpr size_t window_width() const noexcept { return window_width_; }
    constexpr size_t window_width(size_t) const noexcept { return window_width_; }

    floating_rolling_hashers() noexcept {
        // Reset all variables to zeros
        for (auto &multiplier : multipliers_) multiplier = 0.0;
        for (auto &modulo : modulos_) modulo = 0.0;
        for (auto &inverse_modulo : inverse_modulos_) inverse_modulo = 0.0;
        for (auto &negative_discarding_multiplier : negative_discarding_multipliers_)
            negative_discarding_multiplier = 0.0;
        window_width_ = 0;
    }

    /**
     *  @brief Initializes several rolling hashers with different multipliers and modulos.
     *  @param[in] alphabet_size Size of the alphabet, typically 256 for UTF-8, 4 for DNA, or 20 for proteins.
     *  @param[in] first_dimension_offset The offset for the first dimension within a larger fingerprint, typically 0.
     */
    SZ_NOINLINE status_t try_seed(size_t window_width, size_t alphabet_size = 256,
                                  size_t first_dimension_offset = 0) noexcept {
        for (size_t dim = 0; dim < dimensions_k; ++dim) {
            hasher_t hasher(window_width, alphabet_size + first_dimension_offset + dim,
                            hasher_t::default_modulo_base_k);
            multipliers_[dim] = hasher.multiplier();
            modulos_[dim] = hasher.modulo();
            inverse_modulos_[dim] = hasher.inverse_modulo();
            negative_discarding_multipliers_[dim] = hasher.negative_discarding_multiplier();
        }
        window_width_ = window_width;
        return status_t::success_k;
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     */
    SZ_NOINLINE void fingerprint(span<byte_t const> text, min_hashes_span_t min_hashes,
                                 min_counts_span_t min_counts) const noexcept {

        if (text.size() < window_width_) {
            for (auto &min_hash : min_hashes) min_hash = max_hash_k;
            for (auto &min_count : min_counts) min_count = 0;
            return;
        }

        rolling_state_t rolling_states[dimensions_k];
        rolling_state_t rolling_minimums[dimensions_k];
        for (size_t dim = 0; dim < dimensions_k; ++dim)
            rolling_states[dim] = 0, rolling_minimums[dim] = skipped_rolling_hash_k;
        fingerprint_chunk(text, &rolling_states[0], &rolling_minimums[0], min_hashes, min_counts);
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     */
    SZ_NOINLINE status_t try_fingerprint(span<byte_t const> text, min_hashes_span_t min_hashes,
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
    SZ_NOINLINE void fingerprint_chunk(                       //
        span<byte_t const> text_chunk,                        //
        span<rolling_state_t, dimensions_k> last_states,      //
        span<rolling_state_t, dimensions_k> rolling_minimums, //
        min_hashes_span_t min_hashes,                         //
        min_counts_span_t min_counts,                         //
        size_t passed_progress = 0                            //
    ) const noexcept {

        for (unsigned group_index = 0; group_index < groups_count_k; ++group_index)
            roll_group(text_chunk, group_index, last_states, rolling_minimums, min_counts, passed_progress);

        // Finally, export the minimum hashes into the smaller representations
        if (min_hashes)
            for (size_t dim = 0; dim < dimensions_k; ++dim) {
                rolling_state_t const &rolling_minimum = rolling_minimums[dim];
                min_hash_t &min_hash = min_hashes[dim];
                auto const rolling_minimum_as_uint = static_cast<u64_t>(rolling_minimum);
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
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    SZ_NOINLINE status_t operator()(texts_type_ const &texts, min_hashes_per_text_type_ &&min_hashes_per_text, //
                                    min_counts_per_text_type_ &&min_counts_per_text, executor_type_ &&executor = {},
                                    cpu_specs_t specs = {}) noexcept {
        return floating_rolling_hashers_in_parallel_(                     //
            *this, texts,                                                 //
            std::forward<min_hashes_per_text_type_>(min_hashes_per_text), //
            std::forward<min_counts_per_text_type_>(min_counts_per_text), //
            std::forward<executor_type_>(executor), specs);
    }

  private:
    SZ_INLINE __m256d barrett_mod(__m256d xs, __m256d modulos, __m256d inverse_modulos) const noexcept {
        __m256d qs = _mm256_floor_magic_pd(_mm256_mul_pd(xs, inverse_modulos));
        __m256d results = _mm256_fnmadd_pd(qs, modulos, xs);

        // Clamp into the [0, modulo) range.
        __m256d overflow_mask_pd = _mm256_cmp_pd(results, modulos, _CMP_GE_OQ);
        results = _mm256_sub_pd(results, _mm256_and_pd(overflow_mask_pd, modulos));
        __m256d negative_mask_pd = _mm256_cmp_pd(results, _mm256_setzero_pd(), _CMP_LT_OQ);
        results = _mm256_add_pd(results, _mm256_and_pd(negative_mask_pd, modulos));

        return results;
    }

    void roll_group(                                               //
        span<byte_t const> text_chunk, unsigned const group_index, //
        span<rolling_state_t, dimensions_k> last_states,           //
        span<rolling_state_t, dimensions_k> rolling_minimums,      //
        span<min_count_t, dimensions_k> rolling_counts,            //
        size_t const passed_progress = 0) const noexcept {

        unsigned const first_dim = group_index * hashes_per_ymm_k;

        // Register space for in-out variables
        sz_u256_vec_t last_states_vec;
        sz_u256_vec_t rolling_minimums_vec;
        sz_u256_vec_t rolling_counts_vec; // ? Unlike other cases, use larger 64-bit counters simplify masking

        // Use scalar loads for the incomplete tail group
        if (has_incomplete_tail_group_k && group_index + 1 == groups_count_k) {
            for (size_t word_index = 0; word_index < (dimensions_k - first_dim); ++word_index) {
                last_states_vec.f64s[word_index] = last_states[first_dim + word_index];
                rolling_minimums_vec.f64s[word_index] = rolling_minimums[first_dim + word_index];
                rolling_counts_vec.u64s[word_index] = rolling_counts[first_dim + word_index];
            }
        }
        // Otherwise, everything is easy
        else {
            last_states_vec.ymm_pd = _mm256_loadu_pd(&last_states[first_dim]);
            rolling_minimums_vec.ymm_pd = _mm256_loadu_pd(&rolling_minimums[first_dim]);
            rolling_counts_vec.ymm =
                _mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<__m128i const *>(&rolling_counts[first_dim])));
        }

        // Temporary variables for the rolling state
        sz_u256_vec_t multipliers_vec, negative_discarding_multipliers_vec, modulos_vec, inverse_modulos_vec;
        multipliers_vec.ymm_pd = _mm256_loadu_pd(&multipliers_[first_dim]);
        negative_discarding_multipliers_vec.ymm_pd = _mm256_loadu_pd(&negative_discarding_multipliers_[first_dim]);
        modulos_vec.ymm_pd = _mm256_loadu_pd(&modulos_[first_dim]);
        inverse_modulos_vec.ymm_pd = _mm256_loadu_pd(&inverse_modulos_[first_dim]);

        // Until we reach the `window_width_`, we don't need to discard any symbols and can keep the code simpler
        size_t const prefix_length = (std::min)(text_chunk.size(), window_width_);
        size_t new_char_offset = passed_progress;
        for (; new_char_offset < prefix_length; ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            rolling_state_t const new_term = static_cast<rolling_state_t>(new_char) + 1.0;
            __m256d new_term_ymm = _mm256_set1_pd(new_term);

            last_states_vec.ymm_pd = _mm256_fmadd_pd(last_states_vec.ymm_pd, multipliers_vec.ymm_pd, new_term_ymm);
            last_states_vec.ymm_pd = barrett_mod( //
                last_states_vec.ymm_pd,           //
                modulos_vec.ymm_pd,               //
                inverse_modulos_vec.ymm_pd);
        }

        // We now have our first minimum hashes
        __m256i const ones_ymm = _mm256_set1_epi64x(1);
        if (new_char_offset == window_width_ && passed_progress < prefix_length)
            rolling_minimums_vec.ymm_pd = last_states_vec.ymm_pd, rolling_counts_vec.ymm = ones_ymm;

        // Now we can avoid a branch in the nested loop, as we are passed the longest window width
        for (; new_char_offset < text_chunk.size(); ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            byte_t const old_char = text_chunk[new_char_offset - window_width_];
            rolling_state_t const new_term = static_cast<rolling_state_t>(new_char) + 1.0;
            rolling_state_t const old_term = static_cast<rolling_state_t>(old_char) + 1.0;
            __m256d new_term_ymm = _mm256_set1_pd(new_term);
            __m256d old_term_ymm = _mm256_set1_pd(old_term);

            // Discard the old term
            last_states_vec.ymm_pd =
                _mm256_fmadd_pd(negative_discarding_multipliers_vec.ymm_pd, old_term_ymm, last_states_vec.ymm_pd);
            last_states_vec.ymm_pd = barrett_mod( //
                last_states_vec.ymm_pd,           //
                modulos_vec.ymm_pd,               //
                inverse_modulos_vec.ymm_pd);

            // Add the new term
            last_states_vec.ymm_pd = _mm256_fmadd_pd(last_states_vec.ymm_pd, multipliers_vec.ymm_pd, new_term_ymm);
            last_states_vec.ymm_pd = barrett_mod( //
                last_states_vec.ymm_pd,           //
                modulos_vec.ymm_pd,               //
                inverse_modulos_vec.ymm_pd);

            // To keep the right comparison mask, check out: https://stackoverflow.com/q/16988199
            __m256d found_ymm = _mm256_cmp_pd(last_states_vec.ymm_pd, rolling_minimums_vec.ymm_pd, _CMP_LE_OQ);
            __m256d discard_ymm = _mm256_cmp_pd(last_states_vec.ymm_pd, rolling_minimums_vec.ymm_pd, _CMP_GE_OQ);
            rolling_minimums_vec.ymm_pd =
                _mm256_blendv_pd(rolling_minimums_vec.ymm_pd, last_states_vec.ymm_pd, found_ymm);

            // A branchless way to update the counts
            // 1. Discard "min counts" to 0, if a new minimum is found
            // 2. Increment the counts for new & existing minimums
            rolling_counts_vec.ymm_pd = _mm256_blendv_pd(_mm256_setzero_pd(), rolling_counts_vec.ymm_pd, discard_ymm);
            rolling_counts_vec.ymm_pd =
                _mm256_blendv_pd(rolling_counts_vec.ymm_pd,
                                 _mm256_castsi256_pd(_mm256_add_epi64(rolling_counts_vec.ymm, ones_ymm)), found_ymm);
        }

        // Dump back the results from registers into our spans
        if (has_incomplete_tail_group_k && group_index + 1 == groups_count_k) {
            for (size_t word_index = 0; word_index < (dimensions_k - first_dim); ++word_index) {
                last_states[first_dim + word_index] = last_states_vec.f64s[word_index];
                rolling_minimums[first_dim + word_index] = rolling_minimums_vec.f64s[word_index];
                rolling_counts[first_dim + word_index] = static_cast<min_count_t>(rolling_counts_vec.u64s[word_index]);
            }
        }
        else {
            _mm256_storeu_pd(&last_states[first_dim], last_states_vec.ymm_pd);
            _mm256_storeu_pd(&rolling_minimums[first_dim], rolling_minimums_vec.ymm_pd);
            // AVX2-compatible replacement for `_mm256_cvtepi64_epi32`
            __m256i shuffled = _mm256_shuffle_epi32(rolling_counts_vec.ymm, _MM_SHUFFLE(2, 0, 2, 0));
            __m128i lo = _mm256_extracti128_si256(shuffled, 0);
            __m128i hi = _mm256_extracti128_si256(shuffled, 1);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(&rolling_counts[first_dim]), _mm_unpacklo_epi64(lo, hi));
        }
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#pragma endregion Haswell Implementation

/*  AVX512 implementation of the string hashing algorithms for Skylake and newer CPUs.
 *  Includes extensions: F, CD, ER, PF, VL, DQ, BW.
 *
 *  This is the "starting level" for the advanced algorithms using K-mask registers on x86.
 */
#pragma region Skylake Implementation
#if SZ_USE_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512dq,avx512bw,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512dq", "avx512bw", "bmi", "bmi2")
#endif

/**
 *  @brief Alternative to `_mm512_roundscale_pd` and `std::floor`.
 *  Using `_mm512_roundscale_pd` drops throughput to 1/10th of `std::floor`,
 *  while this approach is about 2x faster than `std::floor`.
 */
SZ_INLINE __m512d _mm512_floor_magic_pd(__m512d x) noexcept {
    // Add magic number to force rounding, then subtract it back
    __m512d magic = _mm512_set1_pd(6755399441055744.0); // 2^52 + 2^51
    __m512d rounded = _mm512_sub_pd(_mm512_add_pd(x, magic), magic);

    // Handle negative numbers: if result > x, subtract 1
    __mmask8 neg_mask = _mm512_cmp_pd_mask(rounded, x, _CMP_GT_OQ);
    return _mm512_mask_sub_pd(rounded, neg_mask, rounded, _mm512_set1_pd(1.0));
}

/**
 *  @brief Optimized rolling Min-Hashers built around floating-point numbers.
 *  In a single ZMM register we can store 8 `f64_t` values, so we can process 8 hashes per register.
 */
template <size_t dimensions_>
struct floating_rolling_hashers<sz_cap_skylake_k, dimensions_, void> {

    using hasher_t = floating_rolling_hasher<f64_t>;
    using rolling_state_t = f64_t;
    using min_hash_t = u32_t;
    using min_count_t = u32_t;

    static constexpr size_t dimensions_k = dimensions_;
    static constexpr sz_capability_t capability_k = sz_cap_skylake_k;
    static constexpr rolling_state_t skipped_rolling_hash_k = std::numeric_limits<rolling_state_t>::max();
    static constexpr min_hash_t max_hash_k = std::numeric_limits<min_hash_t>::max();

    using min_hashes_span_t = span<min_hash_t, dimensions_k>;
    using min_counts_span_t = span<min_count_t, dimensions_k>;

    static constexpr unsigned hashes_per_zmm_k = sizeof(sz_u512_vec_t) / sizeof(rolling_state_t);
    static constexpr bool has_incomplete_tail_group_k = (dimensions_k % hashes_per_zmm_k) != 0;
    static constexpr size_t aligned_dimensions_k =
        has_incomplete_tail_group_k ? (dimensions_k / hashes_per_zmm_k + 1) * hashes_per_zmm_k : (dimensions_k);
    static constexpr unsigned groups_count_k = aligned_dimensions_k / hashes_per_zmm_k;

    static_assert(dimensions_k <= 256, "Too many dimensions to keep on stack");

  private:
    rolling_state_t multipliers_[aligned_dimensions_k];
    rolling_state_t modulos_[aligned_dimensions_k];
    rolling_state_t inverse_modulos_[aligned_dimensions_k];
    rolling_state_t negative_discarding_multipliers_[aligned_dimensions_k];
    size_t window_width_ = 0;

  public:
    constexpr size_t dimensions() const noexcept { return dimensions_k; }
    constexpr size_t window_width() const noexcept { return window_width_; }
    constexpr size_t window_width(size_t) const noexcept { return window_width_; }

    floating_rolling_hashers() noexcept {
        // Reset all variables to zeros
        for (auto &multiplier : multipliers_) multiplier = 0.0;
        for (auto &modulo : modulos_) modulo = 0.0;
        for (auto &inverse_modulo : inverse_modulos_) inverse_modulo = 0.0;
        for (auto &negative_discarding_multiplier : negative_discarding_multipliers_)
            negative_discarding_multiplier = 0.0;
        window_width_ = 0;
    }

    /**
     *  @brief Initializes several rolling hashers with different multipliers and modulos.
     *  @param[in] alphabet_size Size of the alphabet, typically 256 for UTF-8, 4 for DNA, or 20 for proteins.
     *  @param[in] first_dimension_offset The offset for the first dimension within a larger fingerprint, typically 0.
     */
    SZ_NOINLINE status_t try_seed(size_t window_width, size_t alphabet_size = 256,
                                  size_t first_dimension_offset = 0) noexcept {
        for (size_t dim = 0; dim < dimensions_k; ++dim) {
            hasher_t hasher(window_width, alphabet_size + first_dimension_offset + dim,
                            hasher_t::default_modulo_base_k);
            multipliers_[dim] = hasher.multiplier();
            modulos_[dim] = hasher.modulo();
            inverse_modulos_[dim] = hasher.inverse_modulo();
            negative_discarding_multipliers_[dim] = hasher.negative_discarding_multiplier();
        }
        window_width_ = window_width;
        return status_t::success_k;
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     */
    SZ_NOINLINE void fingerprint(span<byte_t const> text, min_hashes_span_t min_hashes,
                                 min_counts_span_t min_counts) const noexcept {

        if (text.size() < window_width_) {
            for (auto &min_hash : min_hashes) min_hash = max_hash_k;
            for (auto &min_count : min_counts) min_count = 0;
            return;
        }

        rolling_state_t rolling_states[dimensions_k];
        rolling_state_t rolling_minimums[dimensions_k];
        for (size_t dim = 0; dim < dimensions_k; ++dim)
            rolling_states[dim] = 0, rolling_minimums[dim] = skipped_rolling_hash_k;
        fingerprint_chunk(text, &rolling_states[0], &rolling_minimums[0], min_hashes, min_counts);
    }

    /**
     *  @brief Computes the fingerprint of a single @p text on the current thread.
     *  @param[in] text The input text to hash, typically a UTF-8 encoded string.
     *  @param[out] min_hashes The output fingerprint, a vector of minimum hashes.
     *  @param[out] min_counts The output frequencies of @p `min_hashes` hashes.
     */
    SZ_NOINLINE status_t try_fingerprint(span<byte_t const> text, min_hashes_span_t min_hashes,
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
    SZ_NOINLINE void fingerprint_chunk(                       //
        span<byte_t const> text_chunk,                        //
        span<rolling_state_t, dimensions_k> last_states,      //
        span<rolling_state_t, dimensions_k> rolling_minimums, //
        min_hashes_span_t min_hashes,                         //
        min_counts_span_t min_counts,                         //
        size_t const passed_progress = 0) const noexcept {

        for (unsigned group_index = 0; group_index < groups_count_k; ++group_index)
            roll_group(text_chunk, group_index, last_states, rolling_minimums, min_counts, passed_progress);

        // Finally, export the minimum hashes into the smaller representations
        if (min_hashes)
            for (size_t dim = 0; dim < dimensions_k; ++dim) {
                rolling_state_t const &rolling_minimum = rolling_minimums[dim];
                min_hash_t &min_hash = min_hashes[dim];
                auto const rolling_minimum_as_uint = static_cast<u64_t>(rolling_minimum);
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
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    SZ_NOINLINE status_t operator()(texts_type_ const &texts, min_hashes_per_text_type_ &&min_hashes_per_text, //
                                    min_counts_per_text_type_ &&min_counts_per_text, executor_type_ &&executor = {},
                                    cpu_specs_t specs = {}) noexcept {
        return floating_rolling_hashers_in_parallel_(                     //
            *this, texts,                                                 //
            std::forward<min_hashes_per_text_type_>(min_hashes_per_text), //
            std::forward<min_counts_per_text_type_>(min_counts_per_text), //
            std::forward<executor_type_>(executor), specs);
    }

  private:
    SZ_INLINE __m512d barrett_mod(__m512d xs, __m512d modulos, __m512d inverse_modulos) const noexcept {

        // Use rounding SIMD arithmetic
        __m512d qs = _mm512_floor_magic_pd(_mm512_mul_pd(xs, inverse_modulos));
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

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((no_sanitize("address")))
#endif
    void
    roll_group(                                                    //
        span<byte_t const> text_chunk, unsigned const group_index, //
        span<rolling_state_t, dimensions_k> last_states,           //
        span<rolling_state_t, dimensions_k> rolling_minimums,      //
        span<min_count_t, dimensions_k> rolling_counts,            //
        size_t const passed_progress = 0) const noexcept {

        unsigned const first_dim = group_index * hashes_per_zmm_k;

        // Register space for in-out variables
        sz_u512_vec_t last_states_vec;
        sz_u512_vec_t rolling_minimums_vec;
        sz_u256_vec_t rolling_counts_vec;

        // Use masked loads for the incomplete tail group
        if (has_incomplete_tail_group_k && group_index + 1 == groups_count_k) {
            __mmask8 const load_mask =
                dimensions_k > first_dim ? sz_u8_mask_until_(dimensions_k - first_dim) : (__mmask8)0;
            last_states_vec.zmm_pd = _mm512_maskz_loadu_pd(load_mask, &last_states[first_dim]);
            rolling_minimums_vec.zmm_pd = _mm512_maskz_loadu_pd(load_mask, &rolling_minimums[first_dim]);
            rolling_counts_vec.ymm = _mm256_maskz_loadu_epi32(load_mask, &rolling_counts[first_dim]);
        }
        // Otherwise, everything is easy
        else {
            last_states_vec.zmm_pd = _mm512_loadu_pd(&last_states[first_dim]);
            rolling_minimums_vec.zmm_pd = _mm512_loadu_pd(&rolling_minimums[first_dim]);
            rolling_counts_vec.ymm = _mm256_loadu_si256(reinterpret_cast<__m256i const *>(&rolling_counts[first_dim]));
        }

        // Temporary variables for the rolling state
        sz_u512_vec_t multipliers_vec, negative_discarding_multipliers_vec, modulos_vec, inverse_modulos_vec;
        multipliers_vec.zmm_pd = _mm512_loadu_pd(&multipliers_[first_dim]);
        negative_discarding_multipliers_vec.zmm_pd = _mm512_loadu_pd(&negative_discarding_multipliers_[first_dim]);
        modulos_vec.zmm_pd = _mm512_loadu_pd(&modulos_[first_dim]);
        inverse_modulos_vec.zmm_pd = _mm512_loadu_pd(&inverse_modulos_[first_dim]);

        // Until we reach the `window_width_`, we don't need to discard any symbols and can keep the code simpler
        size_t const prefix_length = (std::min)(text_chunk.size(), window_width_);
        size_t new_char_offset = passed_progress;
        for (; new_char_offset < prefix_length; ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            rolling_state_t const new_term = static_cast<rolling_state_t>(new_char) + 1.0;
            __m512d new_term_zmm = _mm512_set1_pd(new_term);

            last_states_vec.zmm_pd = _mm512_fmadd_pd(last_states_vec.zmm_pd, multipliers_vec.zmm_pd, new_term_zmm);
            last_states_vec.zmm_pd = barrett_mod( //
                last_states_vec.zmm_pd,           //
                modulos_vec.zmm_pd,               //
                inverse_modulos_vec.zmm_pd);
        }

        // We now have our first minimum hashes
        __m256i const ones_ymm = _mm256_set1_epi32(1);
        if (new_char_offset == window_width_ && passed_progress < prefix_length)
            rolling_minimums_vec.zmm_pd = last_states_vec.zmm_pd, rolling_counts_vec.ymm = ones_ymm;

        // Now we can avoid a branch in the nested loop, as we are passed the longest window width
        for (; new_char_offset < text_chunk.size(); ++new_char_offset) {
            byte_t const new_char = text_chunk[new_char_offset];
            byte_t const old_char = text_chunk[new_char_offset - window_width_];
            rolling_state_t const new_term = static_cast<rolling_state_t>(new_char) + 1.0;
            rolling_state_t const old_term = static_cast<rolling_state_t>(old_char) + 1.0;
            __m512d new_term_zmm = _mm512_set1_pd(new_term);
            __m512d old_term_zmm = _mm512_set1_pd(old_term);

            // Discard the old term
            last_states_vec.zmm_pd =
                _mm512_fmadd_pd(negative_discarding_multipliers_vec.zmm_pd, old_term_zmm, last_states_vec.zmm_pd);
            last_states_vec.zmm_pd = barrett_mod( //
                last_states_vec.zmm_pd,           //
                modulos_vec.zmm_pd,               //
                inverse_modulos_vec.zmm_pd);

            // Add the new term
            last_states_vec.zmm_pd = _mm512_fmadd_pd(last_states_vec.zmm_pd, multipliers_vec.zmm_pd, new_term_zmm);
            last_states_vec.zmm_pd = barrett_mod( //
                last_states_vec.zmm_pd,           //
                modulos_vec.zmm_pd,               //
                inverse_modulos_vec.zmm_pd);

            // To keep the right comparison mask, check out: https://stackoverflow.com/q/16988199
            __mmask8 found_mask = _mm512_cmp_pd_mask(last_states_vec.zmm_pd, rolling_minimums_vec.zmm_pd, _CMP_LE_OQ);
            __mmask8 discard_mask = _mm512_cmp_pd_mask(last_states_vec.zmm_pd, rolling_minimums_vec.zmm_pd, _CMP_GE_OQ);
            rolling_minimums_vec.zmm_pd =
                _mm512_mask_mov_pd(rolling_minimums_vec.zmm_pd, found_mask, last_states_vec.zmm_pd);

            // A branchless way to update the counts
            // 1. Discard "min counts" to 0, if a new minimum is found
            // 2. Increment the counts for new & existing minimums
            rolling_counts_vec.ymm = _mm256_maskz_mov_epi32(discard_mask, rolling_counts_vec.ymm);
            rolling_counts_vec.ymm =
                _mm256_mask_add_epi32(rolling_counts_vec.ymm, found_mask, rolling_counts_vec.ymm, ones_ymm);
        }

        // Dump back the results from registers into our spans
        if (has_incomplete_tail_group_k && group_index + 1 == groups_count_k) {
            __mmask8 const store_mask =
                dimensions_k > first_dim ? sz_u8_mask_until_(dimensions_k - first_dim) : (__mmask8)0;
            _mm512_mask_storeu_pd(&last_states[first_dim], store_mask, last_states_vec.zmm_pd);
            _mm512_mask_storeu_pd(&rolling_minimums[first_dim], store_mask, rolling_minimums_vec.zmm_pd);
            _mm256_mask_storeu_epi32(&rolling_counts[first_dim], store_mask, rolling_counts_vec.ymm);
        }
        else {
            _mm512_storeu_pd(&last_states[first_dim], last_states_vec.zmm_pd);
            _mm512_storeu_pd(&rolling_minimums[first_dim], rolling_minimums_vec.zmm_pd);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(&rolling_counts[first_dim]), rolling_counts_vec.ymm);
        }
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SKYLAKE

#pragma endregion Skylake Implementation

#pragma endregion - Optimized Rolling MinHashers

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_FINGERPRINTS_HPP_
