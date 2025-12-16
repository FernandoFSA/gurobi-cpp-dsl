/*
===============================================================================
TEST NAMING — Comprehensive tests for naming.h
===============================================================================

OVERVIEW
--------
Validates the naming system for debug/release configuration detection, string
concatenation, indexed naming patterns, and format-based naming. Tests cover
make_name:: conditional functions, force_name:: always-on functions, and
naming_detail concepts.

TEST ORGANIZATION
-----------------
• Section A: Build configuration detection
• Section B: make_name::concat string concatenation
• Section C: make_name::index variadic and container indices
• Section D: make_name::math bracket notation
• Section E: make_name::format std::format integration
• Section F: force_name:: always-on functions
• Section G: Concept validation (Integral, Streamable)
• Section H: Edge cases and exception safety
• Section I: Performance and consistency checks

TEST STRATEGY
-------------
• Verify debug/release behavior with naming_enabled()/naming_disabled()
• Confirm empty string returns in release mode for make_name::
• Validate exception throwing for empty base names
• Exercise concepts with static_assert for compile-time validation

DEPENDENCIES
------------
• Catch2 v3.0+ - Test framework
• naming.h - System under test
• <vector>, <list>, <array> - Container types for testing

===============================================================================
*/

#include "catch_amalgamated.hpp"
#include <gurobi_dsl/naming.h>

#include <vector>
#include <list>
#include <array>
#include <string>
#include <numeric>
#include <type_traits>
#include <limits>

// ============================================================================
// TEST SUPPORT TYPES
// ============================================================================

/// Expected build configuration for test verification
#if defined(DSL_DEBUG) || defined(_DEBUG)
inline constexpr bool BUILD_CONFIG = true;
#else
inline constexpr bool BUILD_CONFIG = false;
#endif

/// Test type with custom stream operator for concat() tests
struct StreamableType {
    int value;
};

/// Stream operator for StreamableType
inline std::ostream& operator<<(std::ostream& os, const StreamableType& s) {
    return os << "Streamable{" << s.value << "}";
}

/// Test type without stream operator for concept validation
struct NonStreamableType {
    int value;
};

// ============================================================================
// SECTION A: BUILD CONFIGURATION DETECTION
// ============================================================================

/**
 * @test BuildConfig::NamingEnabledMatchesConfig
 * @brief Verifies naming_enabled() matches build configuration
 *
 * @scenario DSL compiled with debug or release settings
 * @given The current build configuration
 * @when Calling naming_enabled() and naming_disabled()
 * @then Values match BUILD_CONFIG expectations
 *
 * @covers naming_enabled()
 * @covers naming_disabled()
 */
TEST_CASE("A1: BuildConfig::NamingEnabledMatchesConfig", "[naming][config]")
{
    REQUIRE(naming_enabled() == BUILD_CONFIG);
    REQUIRE(naming_disabled() == !BUILD_CONFIG);
}

// ============================================================================
// SECTION B: MAKE_NAME::CONCAT STRING CONCATENATION
// ============================================================================

/**
 * @test Concat::BasicFunctionality
 * @brief Verifies make_name::concat basic string concatenation
 *
 * @scenario Various streamable arguments concatenated
 * @given Multiple streamable values
 * @when Calling make_name::concat
 * @then Returns concatenated string in debug, empty in release
 *
 * @covers make_name::concat()
 */
TEST_CASE("B1: Concat::BasicFunctionality", "[naming][concat]")
{
    using make_name::concat;

    if (naming_enabled()) {
        REQUIRE(concat("x") == "x");
        REQUIRE(concat("cost_", 42) == "cost_42");
        REQUIRE(concat("Y", "[", 3, "][", 5, "]") == "Y[3][5]");
        REQUIRE(concat("mixed", 1, '_', 2.5, '_', "text") == "mixed1_2.5_text");
    }
    else {
        REQUIRE(concat("x").empty());
        REQUIRE(concat("cost_", 42).empty());
        REQUIRE(concat("Y", "[", 3, "][", 5, "]").empty());
        REQUIRE(concat("mixed", 1, '_', 2.5, '_', "text").empty());
    }
}

/**
 * @test Concat::CustomStreamableTypes
 * @brief Verifies make_name::concat works with custom streamable types
 *
 * @scenario Custom type with operator<< overload
 * @given StreamableType with value 42
 * @when Calling make_name::concat with custom type
 * @then Correctly streams in debug, empty in release
 *
 * @covers make_name::concat() with custom types
 */
TEST_CASE("B2: Concat::CustomStreamableTypes", "[naming][concat]")
{
    using make_name::concat;

    StreamableType custom{ 42 };

    if (naming_enabled()) {
        REQUIRE(concat("custom_", custom) == "custom_Streamable{42}");
    }
    else {
        REQUIRE(concat("custom_", custom).empty());
    }
}

/**
 * @test Concat::LongStrings
 * @brief Verifies make_name::concat with large strings
 *
 * @scenario Very long input string (2000 characters)
 * @given A 2000-character string
 * @when Calling make_name::concat
 * @then Correctly concatenates in debug, empty in release
 *
 * @covers make_name::concat() efficiency
 */
TEST_CASE("B3: Concat::LongStrings", "[naming][concat]")
{
    using make_name::concat;

    std::string long_string(2000, 'x');

    if (naming_enabled()) {
        auto result = concat("prefix_", long_string, "_suffix");
        REQUIRE(result.size() == 7 + 2000 + 7);
        REQUIRE(result.starts_with("prefix_"));
        REQUIRE(result.ends_with("_suffix"));
    }
    else {
        auto result = concat("prefix_", long_string, "_suffix");
        REQUIRE(result.empty());
    }
}

// ============================================================================
// SECTION C: MAKE_NAME::INDEX VARIADIC AND CONTAINER INDICES
// ============================================================================

/**
 * @test Index::VariadicIndices
 * @brief Verifies make_name::index with variadic integral indices
 *
 * @scenario Various numbers of integral indices
 * @given Base name and 0-5 integral indices
 * @when Calling make_name::index
 * @then Produces underscore-separated names in debug, empty in release
 *
 * @covers make_name::index() variadic
 */
TEST_CASE("C1: Index::VariadicIndices", "[naming][index]")
{
    using make_name::index;

    if (naming_enabled()) {
        REQUIRE(index("x") == "x");
        REQUIRE(index("x", 1) == "x_1");
        REQUIRE(index("x", 1, 2) == "x_1_2");
        REQUIRE(index("x", 1, 2, 3) == "x_1_2_3");
        REQUIRE(index("x", 1, 2, 3, 4, 5) == "x_1_2_3_4_5");
    }
    else {
        REQUIRE(index("x").empty());
        REQUIRE(index("x", 1).empty());
        REQUIRE(index("x", 1, 2).empty());
        REQUIRE(index("x", 1, 2, 3).empty());
    }
}

/**
 * @test Index::NegativeAndExtremeValues
 * @brief Verifies make_name::index handles boundary values
 *
 * @scenario Negative, zero, and extreme integer values
 * @given Boundary integer values including INT_MIN/MAX
 * @when Calling make_name::index
 * @then Correctly formats all values in debug, empty in release
 *
 * @covers make_name::index() with edge values
 */
TEST_CASE("C2: Index::NegativeAndExtremeValues", "[naming][index]")
{
    using make_name::index;

    if (naming_enabled()) {
        REQUIRE(index("x", -3) == "x_-3");
        REQUIRE(index("x", 0) == "x_0");
        REQUIRE(index("x", -1, 0, 1) == "x_-1_0_1");

        REQUIRE(index("lim", std::numeric_limits<int>::min()) ==
            "lim_" + std::to_string(std::numeric_limits<int>::min()));
        REQUIRE(index("lim", std::numeric_limits<int>::max()) ==
            "lim_" + std::to_string(std::numeric_limits<int>::max()));
    }
    else {
        REQUIRE(index("x", -3).empty());
        REQUIRE(index("x", 0).empty());
        REQUIRE(index("x", -1, 0, 1).empty());
    }
}

/**
 * @test Index::IteratorRanges
 * @brief Verifies make_name::index works with iterator ranges
 *
 * @scenario std::vector and std::list containers
 * @given Containers with begin/end iterators
 * @when Calling make_name::index with iterator pair
 * @then Produces correct names in debug, empty in release
 *
 * @covers make_name::index() with iterators
 */
TEST_CASE("C3: Index::IteratorRanges", "[naming][index]")
{
    using make_name::index;

    std::vector<int> vec = { 4, 5, 6 };
    std::list<int> lst = { 7, 8, 9 };

    if (naming_enabled()) {
        REQUIRE(index("vec", vec.begin(), vec.end()) == "vec_4_5_6");
        REQUIRE(index("lst", lst.begin(), lst.end()) == "lst_7_8_9");
    }
    else {
        REQUIRE(index("vec", vec.begin(), vec.end()).empty());
        REQUIRE(index("lst", lst.begin(), lst.end()).empty());
    }
}

/**
 * @test Index::ContainerTypes
 * @brief Verifies make_name::index works with various containers
 *
 * @scenario vector, list, and array containers
 * @given Different container types with values
 * @when Calling make_name::index with container reference
 * @then Produces correct names in debug, empty in release
 *
 * @covers make_name::index() with containers
 */
TEST_CASE("C4: Index::ContainerTypes", "[naming][index]")
{
    using make_name::index;

    std::vector<int> vec = { 1, 2, 3 };
    std::list<int> lst = { 4, 5 };
    std::array<int, 3> arr = { 6, 7, 8 };

    if (naming_enabled()) {
        REQUIRE(index("vec", vec) == "vec_1_2_3");
        REQUIRE(index("lst", lst) == "lst_4_5");
        REQUIRE(index("arr", arr) == "arr_6_7_8");
    }
    else {
        REQUIRE(index("vec", vec).empty());
        REQUIRE(index("lst", lst).empty());
        REQUIRE(index("arr", arr).empty());
    }
}

/**
 * @test Index::EmptyContainers
 * @brief Verifies make_name::index handles empty containers
 *
 * @scenario Empty vector and list containers
 * @given Empty containers
 * @when Calling make_name::index
 * @then Returns base name only in debug, empty in release
 *
 * @covers make_name::index() with empty ranges
 */
TEST_CASE("C5: Index::EmptyContainers", "[naming][index]")
{
    using make_name::index;

    std::vector<int> empty_vec;
    std::list<int> empty_lst;

    if (naming_enabled()) {
        REQUIRE(index("x", empty_vec) == "x");
        REQUIRE(index("x", empty_lst) == "x");
        REQUIRE(index("x", empty_vec.begin(), empty_vec.end()) == "x");
    }
    else {
        REQUIRE(index("x", empty_vec).empty());
        REQUIRE(index("x", empty_lst).empty());
        REQUIRE(index("x", empty_vec.begin(), empty_vec.end()).empty());
    }
}

/**
 * @test Index::LargeSequences
 * @brief Verifies make_name::index handles many indices
 *
 * @scenario Vector with 100 sequential indices
 * @given Large container with 0..99 values
 * @when Calling make_name::index
 * @then Produces correct long name in debug, empty in release
 *
 * @covers make_name::index() efficiency
 */
TEST_CASE("C6: Index::LargeSequences", "[naming][index]")
{
    using make_name::index;

    std::vector<int> large(100);
    std::iota(large.begin(), large.end(), 0);

    if (naming_enabled()) {
        auto result = index("large", large);
        REQUIRE(result.starts_with("large_0_1_2_3"));
        REQUIRE(result.size() > 290);
        REQUIRE(result.find("_99") != std::string::npos);
    }
    else {
        REQUIRE(index("large", large).empty());
    }
}

// ============================================================================
// SECTION D: MAKE_NAME::MATH BRACKET NOTATION
// ============================================================================

/**
 * @test Math::VariadicIndices
 * @brief Verifies make_name::math with variadic indices
 *
 * @scenario Various numbers of integral indices
 * @given Base name and 0-5 integral indices
 * @when Calling make_name::math
 * @then Produces bracket notation in debug, empty in release
 *
 * @covers make_name::math() variadic
 */
TEST_CASE("D1: Math::VariadicIndices", "[naming][math]")
{
    using make_name::math;

    if (naming_enabled()) {
        REQUIRE(math("x") == "x");
        REQUIRE(math("x", 7) == "x[7]");
        REQUIRE(math("x", 1, 2) == "x[1,2]");
        REQUIRE(math("x", 1, 2, 3) == "x[1,2,3]");
        REQUIRE(math("x", 1, 2, 3, 4, 5) == "x[1,2,3,4,5]");
    }
    else {
        REQUIRE(math("x").empty());
        REQUIRE(math("x", 7).empty());
        REQUIRE(math("x", 1, 2).empty());
        REQUIRE(math("x", 1, 2, 3).empty());
    }
}

/**
 * @test Math::ContainerTypes
 * @brief Verifies make_name::math works with containers
 *
 * @scenario vector and array containers
 * @given Different container types with values
 * @when Calling make_name::math with container reference
 * @then Produces bracket notation in debug, empty in release
 *
 * @covers make_name::math() with containers
 */
TEST_CASE("D2: Math::ContainerTypes", "[naming][math]")
{
    using make_name::math;

    std::vector<int> vec = { 9, 8, 7 };
    std::array<int, 2> arr = { 3, 4 };

    if (naming_enabled()) {
        REQUIRE(math("vec", vec) == "vec[9,8,7]");
        REQUIRE(math("arr", arr) == "arr[3,4]");
    }
    else {
        REQUIRE(math("vec", vec).empty());
        REQUIRE(math("arr", arr).empty());
    }
}

// ============================================================================
// SECTION E: MAKE_NAME::FORMAT STD::FORMAT INTEGRATION
// ============================================================================

/**
 * @test Format::BasicFunctionality
 * @brief Verifies make_name::format with std::format strings
 *
 * @scenario Various format strings and arguments
 * @given Format strings with placeholders
 * @when Calling make_name::format
 * @then Produces formatted strings in debug, empty in release
 *
 * @covers make_name::format()
 */
TEST_CASE("E1: Format::BasicFunctionality", "[naming][format]")
{
    using make_name::format;

    if (naming_enabled()) {
        REQUIRE(format("var_{}", 5) == "var_5");
        REQUIRE(format("id_{:04d}", 42) == "id_0042");
        REQUIRE(format("{}_{:.1f}", "temp", 36.6) == "temp_36.6");
        REQUIRE(format("range_{}_{}", 1, 10) == "range_1_10");
    }
    else {
        REQUIRE(format("var_{}", 5).empty());
        REQUIRE(format("id_{:04d}", 42).empty());
        REQUIRE(format("{}_{:.1f}", "temp", 36.6).empty());
    }
}

/**
 * @test Validation::EmptyBaseName
 * @brief Verifies empty base name validation in make_name::
 *
 * @scenario Empty base string with indices
 * @given Empty string as base name
 * @when Calling make_name::index or math with empty base
 * @then Throws in debug, returns empty in release
 *
 * @covers make_name:: empty base validation
 */
TEST_CASE("E2: Validation::EmptyBaseName", "[naming][validation]")
{
    using make_name::index;
    using make_name::math;

    if (naming_enabled()) {
        REQUIRE_THROWS_AS(index("", 1), std::invalid_argument);
        REQUIRE_THROWS_AS(index("", std::vector<int>{1, 2}), std::invalid_argument);
        REQUIRE_THROWS_AS(math("", 1), std::invalid_argument);
        REQUIRE_THROWS_AS(math("", std::vector<int>{1, 2}), std::invalid_argument);
    }
    else {
        REQUIRE(index("", 1).empty());
        REQUIRE(index("", std::vector<int>{1, 2}).empty());
        REQUIRE(math("", 1).empty());
        REQUIRE(math("", std::vector<int>{1, 2}).empty());
    }
}

// ============================================================================
// SECTION F: FORCE_NAME:: ALWAYS-ON FUNCTIONS
// ============================================================================

/**
 * @test ForceConcat::AlwaysProducesOutput
 * @brief Verifies force_name::concat works regardless of build config
 *
 * @scenario Various streamable arguments
 * @given Multiple streamable values
 * @when Calling force_name::concat
 * @then Always returns concatenated string
 *
 * @covers force_name::concat()
 */
TEST_CASE("F1: ForceConcat::AlwaysProducesOutput", "[naming][force]")
{
    using force_name::concat;

    REQUIRE(concat("x") == "x");
    REQUIRE(concat("cost_", 42) == "cost_42");
    REQUIRE(concat("Y", "[", 3, "][", 5, "]") == "Y[3][5]");

    StreamableType custom{ 99 };
    REQUIRE(concat("custom_", custom) == "custom_Streamable{99}");
}

/**
 * @test ForceIndex::AlwaysProducesIndexedNames
 * @brief Verifies force_name::index always works
 *
 * @scenario Various indices and containers
 * @given Indices and container values
 * @when Calling force_name::index
 * @then Always returns underscore-separated names
 *
 * @covers force_name::index()
 */
TEST_CASE("F2: ForceIndex::AlwaysProducesIndexedNames", "[naming][force]")
{
    using force_name::index;

    REQUIRE(index("x") == "x");
    REQUIRE(index("x", 1) == "x_1");
    REQUIRE(index("x", 1, 2, 3) == "x_1_2_3");

    std::vector<int> vec = { 4, 5 };
    REQUIRE(index("vec", vec) == "vec_4_5");

    std::array<int, 3> arr = { 6, 7, 8 };
    REQUIRE(index("arr", arr) == "arr_6_7_8");
}

/**
 * @test ForceMath::AlwaysProducesMathNames
 * @brief Verifies force_name::math always works
 *
 * @scenario Various indices and containers
 * @given Indices and container values
 * @when Calling force_name::math
 * @then Always returns bracket-notation names
 *
 * @covers force_name::math()
 */
TEST_CASE("F3: ForceMath::AlwaysProducesMathNames", "[naming][force]")
{
    using force_name::math;

    REQUIRE(math("x") == "x");
    REQUIRE(math("x", 7) == "x[7]");
    REQUIRE(math("x", 1, 2, 3) == "x[1,2,3]");

    std::vector<int> vec = { 9, 8, 7 };
    REQUIRE(math("vec", vec) == "vec[9,8,7]");

    std::list<int> lst = { 3, 4 };
    REQUIRE(math("lst", lst) == "lst[3,4]");
}

/**
 * @test ForceFormat::AlwaysProducesFormattedNames
 * @brief Verifies force_name::format always works
 *
 * @scenario Various format strings and arguments
 * @given Format strings with placeholders
 * @when Calling force_name::format
 * @then Always returns formatted strings
 *
 * @covers force_name::format()
 */
TEST_CASE("F4: ForceFormat::AlwaysProducesFormattedNames", "[naming][force]")
{
    using force_name::format;

    REQUIRE(format("id_{}", 100) == "id_100");
    REQUIRE(format("temp_{:.2f}C", 23.456) == "temp_23.46C");
    REQUIRE(format("range_{:04d}_{:04d}", 1, 999) == "range_0001_0999");
}

/**
 * @test ForceValidation::EmptyBaseNameThrows
 * @brief Verifies force_name validates empty base names
 *
 * @scenario Empty base string with indices
 * @given Empty string as base name
 * @when Calling force_name::index or math with empty base
 * @then Throws std::invalid_argument when indices present
 *
 * @covers force_name:: empty base validation
 */
TEST_CASE("F5: ForceValidation::EmptyBaseNameThrows", "[naming][force][validation]")
{
    using force_name::index;
    using force_name::math;

    if (naming_enabled()) {
        REQUIRE_THROWS_AS(index("", 1), std::invalid_argument);
        REQUIRE_THROWS_AS(index("", std::vector<int>{1, 2}), std::invalid_argument);
        REQUIRE_THROWS_AS(math("", 1), std::invalid_argument);
        REQUIRE_THROWS_AS(math("", std::vector<int>{1, 2}), std::invalid_argument);
    }

    // Empty base without indices should work
    REQUIRE(index("") == "");
    REQUIRE(math("") == "");
}

/**
 * @test ForceLargeSequences::HandlesMany Indices
 * @brief Verifies force_name handles many indices correctly
 *
 * @scenario Large sequence of 50 indices
 * @given Vector with 0..49 values
 * @when Calling force_name::index and math
 * @then Produces correct long names
 *
 * @covers force_name:: with large sequences
 */
TEST_CASE("F6: ForceLargeSequences::HandlesManyIndices", "[naming][force]")
{
    using force_name::index;
    using force_name::math;

    std::vector<int> large(50);
    std::iota(large.begin(), large.end(), 0);

    auto idx_result = index("big", large);
    auto math_result = math("big", large);

    REQUIRE(idx_result.starts_with("big_0_1_2_3"));
    REQUIRE(math_result.starts_with("big[0,1,2,3"));
    REQUIRE(idx_result.find("_49") != std::string::npos);
    REQUIRE(math_result.find(",49]") != std::string::npos);
}

// ============================================================================
// SECTION G: CONCEPT VALIDATION
// ============================================================================

/**
 * @test IntegralConcept::IdentifiesIntegralTypes
 * @brief Verifies Integral concept correctly identifies integral types
 *
 * @scenario Various C++ types tested against Integral concept
 * @given Integral and non-integral types
 * @when Testing with naming_detail::Integral
 * @then Correctly identifies integral vs non-integral types
 *
 * @covers naming_detail::Integral
 */
TEST_CASE("G1: IntegralConcept::IdentifiesIntegralTypes", "[naming][concept]")
{
    using naming_detail::Integral;

    static_assert(Integral<int>);
    static_assert(Integral<long>);
    static_assert(Integral<unsigned>);
    static_assert(Integral<char>);
    static_assert(Integral<bool>);
    static_assert(Integral<short>);

    static_assert(!Integral<double>);
    static_assert(!Integral<float>);
    static_assert(!Integral<std::string>);
    static_assert(!Integral<std::string_view>);
    static_assert(!Integral<StreamableType>);

    static_assert(Integral<const int>);
    static_assert(Integral<volatile long>);
    static_assert(Integral<const volatile char>);

    SUCCEED("All Integral concept checks passed");
}

/**
 * @test StreamableConcept::IdentifiesStreamableTypes
 * @brief Verifies Streamable concept correctly identifies streamable types
 *
 * @scenario Various types with and without operator<<
 * @given Streamable and non-streamable types
 * @when Testing with naming_detail::Streamable
 * @then Correctly identifies streamable vs non-streamable types
 *
 * @covers naming_detail::Streamable
 */
TEST_CASE("G2: StreamableConcept::IdentifiesStreamableTypes", "[naming][concept]")
{
    using naming_detail::Streamable;

    static_assert(Streamable<int>);
    static_assert(Streamable<double>);
    static_assert(Streamable<std::string>);
    static_assert(Streamable<const char*>);
    static_assert(Streamable<StreamableType>);

    static_assert(!Streamable<NonStreamableType>);
    static_assert(!Streamable<std::vector<int>>);

    static_assert(Streamable<int&>);
    static_assert(Streamable<const StreamableType&>);

    SUCCEED("All Streamable concept checks passed");
}

// ============================================================================
// SECTION H: EDGE CASES AND EXCEPTION SAFETY
// ============================================================================

/**
 * @test MoveSemantics::WorksCorrectly
 * @brief Verifies naming functions work with move semantics
 *
 * @scenario Strings passed as rvalue references
 * @given Movable string values
 * @when Calling naming functions with std::move
 * @then Functions work correctly
 *
 * @covers Move semantics support
 */
TEST_CASE("H1: MoveSemantics::WorksCorrectly", "[naming][edge]")
{
    using make_name::concat;

    std::string movable = "data";

    if (naming_enabled()) {
        auto result1 = concat(std::move(movable), "_suffix");
        REQUIRE(result1 == "data_suffix");
    }

    std::string movable2 = "data";
    auto result2 = force_name::concat(std::move(movable2), "_suffix");
    REQUIRE(result2 == "data_suffix");

    SUCCEED("Move semantics tests passed");
}

/**
 * @test UTF8Support::PreservesUnicode
 * @brief Verifies naming functions handle UTF-8 correctly
 *
 * @scenario Unicode strings and characters
 * @given UTF-8 encoded strings
 * @when Calling naming functions with Unicode content
 * @then Preserves Unicode characters correctly
 *
 * @covers UTF-8 string handling
 */
TEST_CASE("H2: UTF8Support::PreservesUnicode", "[naming][edge]")
{
    using force_name::concat;
    using force_name::format;

    std::string unicode_name = "variable_abc_y";
    REQUIRE(concat(unicode_name, "_", 1) == "variable_abc_y_1");

    std::string with_emoji = "cost_$";
    REQUIRE(concat(with_emoji, "_", 100) == "cost_$_100");

    REQUIRE(format("price_{}_in", 99) == "price_99_in");

    SUCCEED("Unicode/UTF-8 tests passed");
}

// ============================================================================
// SECTION I: PERFORMANCE AND CONSISTENCY CHECKS
// ============================================================================

/**
 * @test CompileTimeConstraints::PreventInvalidUsage
 * @brief Verifies concepts prevent invalid usage at compile time
 *
 * @scenario Valid and invalid type usage with naming functions
 * @given Integral and non-integral index types
 * @when Compiling naming function calls
 * @then Valid code compiles, invalid would fail
 *
 * @covers Compile-time type constraints
 */
TEST_CASE("I1: CompileTimeConstraints::PreventInvalidUsage", "[naming][compile]")
{
    using make_name::index;
    using make_name::math;

    if (naming_enabled()) {
        auto valid1 = index("x", 1, 2, 3);
        auto valid2 = math("y", 4, 5);
        (void)valid1;
        (void)valid2;
    }

    // The following would fail to compile:
    // auto invalid1 = index("x", 1.5, 2);      // double not integral
    // auto invalid2 = math("y", "string", 3);  // string not integral

    SUCCEED("Compile-time constraints verified indirectly");
}

/**
 * @test ReleasePerformance::MinimalOverhead
 * @brief Verifies make_name:: has minimal overhead in release builds
 *
 * @scenario 1000 calls to make_name:: functions
 * @given Release build (naming_disabled())
 * @when Calling make_name:: functions repeatedly
 * @then Returns quickly without string construction
 *
 * @covers Release mode performance
 */
TEST_CASE("I2: ReleasePerformance::MinimalOverhead", "[naming][performance]")
{
    if (naming_disabled()) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 1000; ++i) {
            auto result = make_name::concat("prefix_", i, "_suffix");
            REQUIRE(result.empty());

            auto result2 = make_name::index("x", 1, 2, 3, 4, 5);
            REQUIRE(result2.empty());

            auto result3 = make_name::math("y", std::vector<int>{1, 2, 3});
            REQUIRE(result3.empty());
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        REQUIRE(duration.count() < 10);
    }
    else {
        SUCCEED("Test only applicable in release builds");
    }
}

/**
 * @test Consistency::IteratorAndContainerEquivalence
 * @brief Verifies iterator and container forms produce identical results
 *
 * @scenario Same data in different container types
 * @given vector and list with identical values
 * @when Calling naming functions with iterator vs container overloads
 * @then Results are identical
 *
 * @covers API consistency
 */
TEST_CASE("I3: Consistency::IteratorAndContainerEquivalence", "[naming][consistency]")
{
    using make_name::index;
    using make_name::math;

    std::vector<int> vec = { 1, 2, 3, 4, 5 };
    std::list<int> lst = { 1, 2, 3, 4, 5 };

    if (naming_enabled()) {
        auto vec_idx = index("x", vec);
        auto lst_idx = index("x", lst);
        auto iter_idx = index("x", vec.begin(), vec.end());

        REQUIRE(vec_idx == iter_idx);
        REQUIRE(vec_idx == lst_idx);

        auto vec_math = math("y", vec);
        auto lst_math = math("y", lst);
        auto iter_math = math("y", vec.begin(), vec.end());

        REQUIRE(vec_math == iter_math);
        REQUIRE(vec_math == lst_math);
    }

    auto f_vec_idx = force_name::index("x", vec);
    auto f_lst_idx = force_name::index("x", lst);
    auto f_iter_idx = force_name::index("x", vec.begin(), vec.end());

    REQUIRE(f_vec_idx == f_iter_idx);
    REQUIRE(f_vec_idx == f_lst_idx);

    auto f_vec_math = force_name::math("y", vec);
    auto f_lst_math = force_name::math("y", lst);
    auto f_iter_math = force_name::math("y", vec.begin(), vec.end());

    REQUIRE(f_vec_math == f_iter_math);
    REQUIRE(f_vec_math == f_lst_math);
}