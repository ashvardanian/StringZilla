/**
 *  @brief ISA-agnostic rolling-hash fingerprint template core (serial backend).
 *  @file include/stringzillas/fingerprints/serial.hpp
 *  @author Ash Vardanian
 *
 *  Contains the shared rolling-hash template core for all fingerprint backends -
 *  the baseline rolling hashers and the optimized rolling MinHashers - plus the serial aliases.
 *
 *  Every backend specialization header (`haswell.hpp`, `skylake.hpp`, `cuda.cuh`, ...) must
 *  include this file first, so the primary templates are visible before any specialization.
 */
#ifndef STRINGZILLAS_FINGERPRINTS_SERIAL_HPP_
#define STRINGZILLAS_FINGERPRINTS_SERIAL_HPP_

#include "stringzilla/types.hpp"  // `sz::error_cost_t`
#include "stringzillas/types.hpp" // `sz::executor_like`

#include <cstddef>
#include <limits>   // `std::numeric_limits` for numeric types
#include <iterator> // `std::iterator_traits` for iterators
#include <cmath>    // `std::fabsf` for `f32_rolling_hasher`
#include <numeric>  // `std::gcd` for `choose_coprime_modulo`

namespace ashvardanian {
namespace stringzillas {

#pragma region Baseline Rolling Hashers

/**
 *  @brief The default reproducibility seed for fingerprint engines.
 *
 *  Every seed - including this default - derives per-dimension multipliers @b and per-dimension moduli from a
 *  `splitmix64` stream of `seed + dim`, which is what gives the MinHashes their statistical independence. The
 *  default simply yields one deterministic, well-distributed set of per-dimension parameters.
 */
static constexpr u64_t default_seed_k = 0;

/**
 *  @brief SplitMix64 finalizer - a fast, well-distributed integer mixer used to derive per-dimension parameters.
 *  @param[in] state The input state to mix, typically `seed + dim` for a per-dimension stream.
 *  @return A thoroughly mixed 64-bit value.
 *  @sa https://prng.di.unimi.it/splitmix64.c
 */
inline constexpr u64_t splitmix64(u64_t state) noexcept {
    state += 0x9E3779B97F4A7C15ull;
    u64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

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

    /**
     *  @brief Seeds dimension @p dim of a fingerprint from a reproducibility @p seed.
     *
     *  Draws a per-dimension multiplier from a `splitmix64` stream of `seed + dim`, so neighbouring dimensions
     *  never share parameters. This hasher carries no modulo, relying on the integer type's natural overflow.
     */
    explicit multiplying_rolling_hasher(size_t window_width, size_t alphabet_size, size_t dim, u64_t seed) noexcept
        : multiplying_rolling_hasher(window_width,
                                     static_cast<hash_t>(alphabet_size + 1u + (splitmix64(seed + dim) % 256u))) {}

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

inline u64_t choose_coprime_modulo(u64_t multiplier, u64_t limit) noexcept; // ? Defined just below the hasher

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

    /**
     *  @brief Seeds dimension @p dim of a fingerprint from a reproducibility @p seed.
     *
     *  Draws a per-dimension multiplier from a `splitmix64` stream of `seed + dim`, then pairs it with a
     *  per-dimension co-prime modulo, so neighbouring dimensions share neither parameter - the property that
     *  keeps the resulting MinHashes statistically independent.
     */
    explicit rabin_karp_rolling_hasher(size_t window_width, size_t alphabet_size, size_t dim, u64_t seed) noexcept
        : rabin_karp_rolling_hasher(window_width, seeded_multiplier(alphabet_size, dim, seed),
                                    seeded_modulo(alphabet_size, dim, seed)) {}

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
    static hash_t seeded_multiplier(size_t alphabet_size, size_t dim, u64_t seed) noexcept {
        return static_cast<hash_t>(alphabet_size + (splitmix64(seed + dim) % 256u));
    }
    static hash_t seeded_modulo(size_t alphabet_size, size_t dim, u64_t seed) noexcept {
        u64_t const modulo = choose_coprime_modulo(seeded_multiplier(alphabet_size, dim, seed), default_modulo_base_k);
        return static_cast<hash_t>(modulo ? modulo : static_cast<u64_t>(default_modulo_base_k));
    }

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

    constexpr buz_rolling_hasher() noexcept : window_width_ {0}, table_ {} {}

    explicit buz_rolling_hasher(size_t window_width, u64_t seed = 0x9E3779B97F4A7C15ull) noexcept
        : window_width_ {window_width}, table_ {} {

        sz_assert_(window_width_ > 1 && "Window width must be > 1");
        for (size_t i = 0; i < 256; ++i) table_[i] = static_cast<state_t>(splitmix64(seed + i));
    }

    /**
     *  @brief Seeds dimension @p dim of a fingerprint from a reproducibility @p seed.
     *
     *  Folds `seed + dim` through `splitmix64` into the table seed, so neighbouring dimensions get independent
     *  substitution tables. The alphabet size is unused - BuzHash always covers the full 256-byte table.
     */
    explicit buz_rolling_hasher(size_t window_width, [[maybe_unused]] size_t alphabet_size, size_t dim,
                                u64_t seed) noexcept
        : buz_rolling_hasher(window_width, splitmix64(seed + dim)) {}

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

    size_t window_width_;
    state_t table_[256];
};

/**
 *  @brief Helper function to pick the second co-prime "modulo" base for the Karp-Rabin rolling hashes.
 *  @retval 0 on failure, or a valid prime number otherwise.
 */
inline u64_t choose_coprime_modulo(u64_t multiplier, u64_t limit) noexcept {
    u64_t max_input = std::numeric_limits<byte_t>::max() + 1u;
    // Reject `multiplier`/`limit` combinations that would underflow `limit - (max_input + 1)` below into an
    // enormous unsigned `bound` and spin the search loop effectively forever.
    if (multiplier == 0 || multiplier >= limit || limit <= max_input + 1) return 0;

    // Upper bound guaranteeing no overflow in non-discarding `update` calls
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

    /**
     *  @brief Seeds dimension @p dim of a fingerprint from a reproducibility @p seed.
     *
     *  Pulls the per-dimension multiplier and modulo from independent `splitmix64` streams of `seed + dim`, so
     *  neighbouring dimensions share neither parameter. Both ranges stay within the narrow `f32` overflow headroom.
     */
    explicit floating_rolling_hasher(size_t window_width, size_t alphabet_size, size_t dim, u64_t seed) noexcept
        : floating_rolling_hasher(window_width, static_cast<hash_t>(seeded_multiplier(alphabet_size, dim, seed)),
                                  static_cast<hash_t>(seeded_modulo(alphabet_size, dim, seed))) {}

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
    static state_t seeded_multiplier([[maybe_unused]] size_t alphabet_size, size_t dim, u64_t seed) noexcept {
        // The multiplier range already spans a full 256-symbol alphabet.
        return static_cast<state_t>(default_alphabet_size_k + (splitmix64(seed + dim) % 256ull)); // Range [256, 512)
    }
    static state_t seeded_modulo([[maybe_unused]] size_t alphabet_size, size_t dim, u64_t seed) noexcept {
        // A second, independent stream nudges the per-dimension modulo just below the validated base.
        return static_cast<state_t>(default_modulo_base_k - (splitmix64(splitmix64(seed + dim)) % 256ull));
    }

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
          negative_discarding_multiplier_ {1.0}, discarding_multiplier_ {0.0} {

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

        // The single-reduction `roll` fuses both updates into one Barrett pass. It folds the multiplier into the
        // discarding coefficient and uses its non-negative complement, so the combined intermediate never goes
        // negative before reduction: `state * multiplier + new_term + discarding_multiplier * old_term`.
        discarding_multiplier_ = std::fmod(negative_discarding_multiplier_ * multiplier_, modulo_) + modulo_;
        if (discarding_multiplier_ >= modulo_) discarding_multiplier_ -= modulo_;

        // The fused intermediate adds a second `largest_normalized_state * input_term` summand on top of the
        // classic one, so it requires a slightly tighter headroom than the two-reduction `push`/two-step `roll`.
        state_t const largest_fused_intermediary = largest_intermediary + largest_normalized_state * largest_input_term;
        sz_assert_(largest_fused_intermediary < limit_k && "Fused intermediate state overflows the limit");
    }

    /**
     *  @brief Seeds dimension @p dim of a fingerprint from a reproducibility @p seed.
     *
     *  Pulls the per-dimension multiplier and modulo from independent `splitmix64` streams of `seed + dim`. Varying
     *  the modulo per dimension - rather than only the multiplier over one shared modulo - is what makes the
     *  resulting MinHashes statistically independent, while both ranges stay within the Barrett overflow headroom.
     */
    explicit floating_rolling_hasher(size_t window_width, size_t alphabet_size, size_t dim, u64_t seed) noexcept
        : floating_rolling_hasher(window_width, seeded_multiplier(alphabet_size, dim, seed),
                                  seeded_modulo(alphabet_size, dim, seed)) {}

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

        // A single Barrett reduction handles both the discarded head symbol and the incoming tail symbol.
        // `discarding_multiplier_` already folds in the `multiplier_` and uses the non-negative complement,
        // so the intermediate stays within `[0, limit_k)` without underflowing zero before the reduction.
        state_t fused = state * multiplier_ + new_term;
        fused = fused + discarding_multiplier_ * old_term;
        return barrett_mod(fused);
    }

    constexpr hash_t digest(state_t state) const noexcept { return static_cast<hash_t>(state); }

    constexpr state_t multiplier() const noexcept { return multiplier_; }
    constexpr state_t modulo() const noexcept { return modulo_; }
    constexpr state_t inverse_modulo() const noexcept { return inverse_modulo_; }
    constexpr state_t negative_discarding_multiplier() const noexcept { return negative_discarding_multiplier_; }
    constexpr state_t discarding_multiplier() const noexcept { return discarding_multiplier_; }

  private:
    static state_t seeded_multiplier([[maybe_unused]] size_t alphabet_size, size_t dim, u64_t seed) noexcept {
        // The multiplier range already spans a full 256-symbol alphabet.
        return static_cast<state_t>(256ull + (splitmix64(seed + dim) % 384ull)); // Range [256, 640)
    }
    static state_t seeded_modulo([[maybe_unused]] size_t alphabet_size, size_t dim, u64_t seed) noexcept {
        // A second, independent stream keeps the per-dimension modulo just below the validated base, preserving
        // the Barrett-reduction overflow headroom.
        u64_t const modulo_drop = splitmix64(splitmix64(seed + dim)) % (1ull << 20);
        return static_cast<state_t>(static_cast<u64_t>(default_modulo_base_k) - modulo_drop);
    }

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
    state_t discarding_multiplier_ = 0.0;
};

#pragma endregion Baseline Rolling Hashers

#pragma region Optimized Rolling MinHashers

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
    SZ_NOINLINE status_t try_extend(size_t window_width, size_t new_dims, size_t alphabet_size = 256,
                                    u64_t seed = default_seed_k) noexcept {
        size_t const old_dims = hashers_.size();
        if (hashers_.try_reserve(old_dims + new_dims) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t new_dim = 0; new_dim < new_dims; ++new_dim) {
            size_t const dim = old_dims + new_dim;
            // Seeding goes through the hasher's own constructor, so every backend - AoS or SoA - derives identical
            // per-dimension parameters from `seed + dim`.
            status_t status = try_append(hasher_t(window_width, alphabet_size, dim, seed));
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
        auto rolling_states = span<rolling_state_t, dimensions_>(rolling_states_buffer.data(),
                                                                 rolling_states_buffer.size());
        auto rolling_minimums = span<rolling_hash_t, dimensions_>(rolling_minimums_buffer.data(),
                                                                  rolling_minimums_buffer.size());
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
                auto overlapping_text_end = (std::min)(text_start + chunk_size + max_window_width_ - 1,
                                                       text_view.end());
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
     *  @param[in] seed Reproducibility seed; every value derives independent per-dimension multipliers and moduli.
     */
    SZ_NOINLINE status_t try_seed(size_t window_width, size_t alphabet_size = 256, size_t first_dimension_offset = 0,
                                  u64_t seed = default_seed_k) noexcept {
        for (size_t dim = 0; dim < dimensions_k; ++dim) {
            hasher_t hasher(window_width, alphabet_size, first_dimension_offset + dim, seed);
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

#pragma endregion Optimized Rolling MinHashers

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_FINGERPRINTS_SERIAL_HPP_
