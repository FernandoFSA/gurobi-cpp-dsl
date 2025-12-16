#pragma once
/*
===============================================================================
NAMING SYSTEM — Human-readable symbolic names for the C++ Gurobi DSL
===============================================================================

OVERVIEW
--------
Provides compile-time and runtime name generation for Gurobi model elements.
Supports both debug (human-readable) and release (zero-overhead) modes with
consistent naming conventions across variables, constraints, and expressions.

KEY COMPONENTS
--------------
• make_name:: Debug-aware naming (returns empty strings in release builds)
• force_name:: Always-on naming for export, logging, and diagnostics
• naming_detail:: Internal implementation details and concepts
• Concepts: Integral, Streamable, IndexRange for compile-time type safety

DESIGN PHILOSOPHY
-----------------
• Zero runtime overhead in release builds by default
• Compile-time type checking via C++20 concepts
• Consistent conventions: index style (x_i_j) vs math style (x[i,j])
• Full support for variadic indices, ranges, containers, and std::format
• Seamless integration with DSL indexing and variable systems

USAGE EXAMPLES
--------------
    // Debug mode: "x_3", Release mode: ""
    auto var_name = make_name::index("x", 3);

    // Always produces name: "cap[1,2]"
    auto constr_name = force_name::math("cap", 1, 2);

    // std::format integration
    auto formatted = make_name::format("var_{:03d}_{}", 5, "suffix");

DEPENDENCIES
------------
• <string>, <string_view>, <sstream>, <format> - String handling
• <concepts>, <type_traits> - Compile-time type checking
• <stdexcept> - Exception handling for validation errors

PERFORMANCE NOTES
-----------------
• make_name:: functions: O(1) when naming_disabled() (return empty string)
• force_name:: functions: O(n) where n = total characters in result
• No dynamic allocation for empty results in release mode
• String building optimized with std::format where applicable

THREAD SAFETY
-------------
• All functions are thread-safe for concurrent calls
• No shared mutable state between invocations
• String streams use internal buffers; no thread-local storage required

EXCEPTION SAFETY
----------------
• make_name::: No-throw guarantee when naming_disabled()
• force_name::: Strong exception safety (complete or no operation)
• Base name validation: Throws std::invalid_argument on empty base with indices
• std::format errors: Propagate std::format_error exceptions

===============================================================================
*/

#include <string>
#include <string_view>
#include <sstream>
#include <stdexcept>
#include <iterator>
#include <type_traits>
#include <concepts>
#include <ostream>
#include <utility>
#include <algorithm>
#include <format>
#include <array>
#include <ranges>
#include <vector>

// ============================================================================
// BUILD CONFIGURATION
// ============================================================================
#if defined(DSL_DEBUG) || defined(_DEBUG)
    /**
     * @brief Global compile-time switch for debug naming mode
     *
     * @value true when DSL_DEBUG or _DEBUG is defined
     * @value false in release builds
     *
     * @note Controlled by build system; not runtime configurable
     * @see naming_enabled()
     */
inline constexpr bool DEBUG_NAMES = true;
#else
inline constexpr bool DEBUG_NAMES = false;
#endif

// ============================================================================
// PUBLIC INTERFACE DECLARATIONS
// ============================================================================

/**
 * @brief Returns true if debug-style naming is enabled
 *
 * @return true in debug builds (DSL_DEBUG or _DEBUG defined)
 * @return false in release builds
 *
 * @complexity O(1)
 * @noexcept
 *
 * @note This is a compile-time constant in practice, but exposed as
 *       a function for consistency with the DSL naming conventions.
 */
[[nodiscard]] constexpr bool naming_enabled() noexcept;

/**
 * @brief Returns true if debug-style naming is disabled
 *
 * @return true in release builds
 * @return false in debug builds
 *
 * @complexity O(1)
 * @noexcept
 *
 * @note Convenience function equivalent to !naming_enabled()
 */
[[nodiscard]] constexpr bool naming_disabled() noexcept;

// ============================================================================
// IMPLEMENTATION
// ============================================================================

// ----------------------------------------------------------------------------
// Public function implementations
// ----------------------------------------------------------------------------

[[nodiscard]] constexpr bool naming_enabled() noexcept {
    return DEBUG_NAMES;
}

[[nodiscard]] constexpr bool naming_disabled() noexcept {
    return !DEBUG_NAMES;
}

// ----------------------------------------------------------------------------
// Internal implementation details
// ----------------------------------------------------------------------------
namespace naming_detail {

    // ========================================================================
    // CONCEPTS
    // ========================================================================

    /**
     * @concept Integral
     * @brief True for integral types used as indices
     *
     * @tparam T Type to test
     *
     * @requirements
     * - std::is_integral_v<T> must be true
     * - Includes all signed and unsigned integer types
     * - Excludes floating-point, pointers, and class types
     *
     * @note char, bool, and enum types are considered integral
     */
    template<typename T>
    concept Integral = std::is_integral_v<std::remove_cvref_t<T>>;

    /**
     * @concept Streamable
     * @brief True if type can be written to std::ostream via operator<<
     *
     * @tparam T Type to test
     *
     * @requirements
     * - Must have valid operator<<(std::ostream&, T) overload
     * - Return type must be std::ostream& (for chaining)
     *
     * @note Used for concat() and format() functions
     */
    template<typename T>
    concept Streamable = requires(std::ostream & os, T && value) {
        { os << std::forward<T>(value) } -> std::same_as<std::ostream&>;
    };

    /**
     * @concept IndexIterator
     * @brief Iterator whose value_type is integral
     *
     * @tparam It Iterator type to test
     *
     * @requirements
     * - Must satisfy std::input_iterator
     * - iterator_traits<It>::value_type must satisfy Integral
     *
     * @note Supports raw pointers, vector::iterator, etc.
     */
    template<typename It>
    concept IndexIterator =
        std::input_iterator<It> &&
        Integral<typename std::iterator_traits<It>::value_type>;

    /**
     * @concept IndexRange
     * @brief Range whose elements are integral values
     *
     * @tparam R Range type to test
     *
     * @requirements
     * - Must have begin() and end() methods
     * - Dereferenced begin() must satisfy Integral
     * - Supports std::ranges, containers, and custom ranges
     */
    template<typename R>
    concept IndexRange =
        requires(R r) {
        std::ranges::begin(r);
        std::ranges::end(r);
    }&&
        Integral<std::remove_cvref_t<decltype(*std::ranges::begin(std::declval<R&>()))>>;

    // ========================================================================
    // Compile-time validation helpers
    // ========================================================================

    template<IndexIterator It>
    constexpr void validate_iterator() {
        using V = typename std::iterator_traits<It>::value_type;
        static_assert(Integral<V>,
            "naming: iterator dereference type must be integral");
    }

    template<typename... Ts>
    constexpr void validate_variadic() {
        static_assert((Integral<Ts> && ...),
            "naming: variadic index arguments must all be integral");
    }

    template<IndexRange R>
    constexpr void validate_range() {
        using V = std::remove_cvref_t<decltype(*std::ranges::begin(std::declval<R&>()))>;
        static_assert(Integral<V>,
            "naming: range-based index arguments must produce integral values");
    }

    template<typename... Args>
    constexpr void validate_concat() {
        static_assert((Streamable<Args> && ...),
            "naming: all concat() arguments must be streamable via operator<<");
    }

    // ========================================================================
    // Runtime validation
    // ========================================================================

    inline void check_base_name(std::string_view base, bool has_indices) {
        if constexpr (DEBUG_NAMES) {
            if (has_indices && base.empty()) {
                throw std::invalid_argument(
                    "naming: base name cannot be empty when indices are present");
            }
        }
        (void)base;
        (void)has_indices;
    }

    // ========================================================================
    // Core implementation functions
    // ========================================================================

    template<typename... Args>
    inline std::string concat_impl(Args&&... parts) {
        validate_concat<Args...>();
        std::ostringstream oss;
        ((oss << std::forward<Args>(parts)), ...);
        return oss.str();
    }

    template<Integral... Indices>
    inline std::string index_impl(std::string_view base, Indices... idx) {
        constexpr std::size_t N = sizeof...(idx);
        check_base_name(base, N > 0);

        if constexpr (N == 0) {
            return std::string(base);
        }

        std::string result;
        result.reserve(base.size() + (N * 6));

        result.append(base);
        ((result.append("_").append(std::to_string(static_cast<long long>(idx)))), ...);

        return result;
    }

    template<IndexIterator It>
    inline std::string index_impl(std::string_view base, It first, It last) {
        bool has_indices = (first != last);
        check_base_name(base, has_indices);

        if (!has_indices) {
            return std::string(base);
        }

        std::string result;
        result.reserve(base.size() + std::distance(first, last) * 6);

        result.append(base);
        for (; first != last; ++first) {
            result.append("_").append(std::to_string(static_cast<long long>(*first)));
        }

        return result;
    }

    template<Integral... Indices>
    inline std::string math_impl(std::string_view base, Indices... idx) {
        constexpr std::size_t N = sizeof...(idx);
        check_base_name(base, N > 0);

        if constexpr (N == 0) {
            return std::string(base);
        }

        std::string result;
        result.reserve(base.size() + (N * 6) + 2);

        result.append(base).append("[");

        bool first = true;
        ((result.append(first ? (first = false, "") : ",")
            .append(std::to_string(static_cast<long long>(idx)))), ...);

        result.append("]");
        return result;
    }

    template<IndexIterator It>
    inline std::string math_impl(std::string_view base, It first, It last) {
        bool has_indices = (first != last);
        check_base_name(base, has_indices);

        if (!has_indices) {
            return std::string(base);
        }

        std::string result;
        result.reserve(base.size() + std::distance(first, last) * 6 + 2);

        result.append(base).append("[");

        bool first_flag = true;
        for (; first != last; ++first) {
            if (!first_flag) {
                result.append(",");
            }
            first_flag = false;
            result.append(std::to_string(static_cast<long long>(*first)));
        }

        result.append("]");
        return result;
    }

    template<typename... Args>
    inline std::string format_impl(std::format_string<Args...> fmt, Args&&... args) {
        return std::format(fmt, std::forward<Args>(args)...);
    }

} // namespace naming_detail

// ============================================================================
// make_name IMPLEMENTATION
// ============================================================================
namespace make_name {

    template<typename... Args>
    inline std::string concat(Args&&... parts) {
        if (naming_disabled()) {
            return {};
        }
        return naming_detail::concat_impl(std::forward<Args>(parts)...);
    }

    template<typename... Indices>
        requires (naming_detail::Integral<Indices> && ...)
    inline std::string index(std::string_view base, Indices... idx) {
        if (naming_disabled()) {
            return {};
        }
        return naming_detail::index_impl(base, idx...);
    }

    template<naming_detail::IndexIterator It>
    inline std::string index(std::string_view base, It first, It last) {
        if (naming_disabled()) {
            return {};
        }
        return naming_detail::index_impl(base, first, last);
    }

    template<naming_detail::IndexRange R>
    inline std::string index(std::string_view base, const R& indices) {
        if (naming_disabled()) {
            return {};
        }
        return naming_detail::index_impl(base, std::ranges::begin(indices), std::ranges::end(indices));
    }

    template<typename... Indices>
        requires (naming_detail::Integral<Indices> && ...)
    inline std::string math(std::string_view base, Indices... idx) {
        if (naming_disabled()) {
            return {};
        }
        return naming_detail::math_impl(base, idx...);
    }

    template<naming_detail::IndexIterator It>
    inline std::string math(std::string_view base, It first, It last) {
        if (naming_disabled()) {
            return {};
        }
        return naming_detail::math_impl(base, first, last);
    }

    template<naming_detail::IndexRange R>
    inline std::string math(std::string_view base, const R& indices) {
        if (naming_disabled()) {
            return {};
        }
        return naming_detail::math_impl(base, std::ranges::begin(indices), std::ranges::end(indices));
    }

    template<typename... Args>
    inline std::string format(std::format_string<Args...> fmt, Args&&... args) {
        if (naming_disabled()) {
            return {};
        }
        return naming_detail::format_impl(fmt, std::forward<Args>(args)...);
    }

} // namespace make_name

// ============================================================================
// force_name IMPLEMENTATION
// ============================================================================
namespace force_name {

    template<typename... Args>
    inline std::string concat(Args&&... parts) {
        return naming_detail::concat_impl(std::forward<Args>(parts)...);
    }

    template<typename... Indices>
        requires (naming_detail::Integral<Indices> && ...)
    inline std::string index(std::string_view base, Indices... idx) {
        return naming_detail::index_impl(base, idx...);
    }

    template<naming_detail::IndexIterator It>
    inline std::string index(std::string_view base, It first, It last) {
        return naming_detail::index_impl(base, first, last);
    }

    template<naming_detail::IndexRange R>
    inline std::string index(std::string_view base, const R& indices) {
        return naming_detail::index_impl(base, std::ranges::begin(indices), std::ranges::end(indices));
    }

    template<typename... Indices>
        requires (naming_detail::Integral<Indices> && ...)
    inline std::string math(std::string_view base, Indices... idx) {
        return naming_detail::math_impl(base, idx...);
    }

    template<naming_detail::IndexIterator It>
    inline std::string math(std::string_view base, It first, It last) {
        return naming_detail::math_impl(base, first, last);
    }

    template<naming_detail::IndexRange R>
    inline std::string math(std::string_view base, const R& indices) {
        return naming_detail::math_impl(base, std::ranges::begin(indices), std::ranges::end(indices));
    }

    template<typename... Args>
    inline std::string format(std::format_string<Args...> fmt, Args&&... args) {
        return naming_detail::format_impl(fmt, std::forward<Args>(args)...);
    }

} // namespace force_name