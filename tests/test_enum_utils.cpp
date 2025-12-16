/*
===============================================================================
TEST ENUM_UTILS — Comprehensive tests for enum_utils.h
===============================================================================

OVERVIEW
--------
Validates the enum utilities for macro expansion, type-safe arrays, compile-time
traits, bounds checking, and safe conversions. Tests cover DECLARE_ENUM_WITH_COUNT
macro behavior, EnumArray template, enum_size trait, is_valid_enum_value bounds
checking, and enum_from_value safe conversions.

TEST ORGANIZATION
-----------------
• Section A: DECLARE_ENUM_WITH_COUNT macro behavior
• Section B: EnumArray type-safe indexing
• Section C: enum_size trait functionality
• Section D: is_valid_enum_value bounds checking
• Section E: enum_from_value safe conversions
• Section F: Integration and real-world usage
• Section G: Constexpr verification

TEST STRATEGY
-------------
• Verify macro expansion produces correct enum class definitions
• Confirm type safety with std::is_enum_v and underlying_type checks
• Validate compile-time evaluation with static_assert
• Exercise boundary conditions for validation functions

DEPENDENCIES
------------
• Catch2 v3.0+ - Test framework
• enum_utils.h - System under test
• <array>, <type_traits>, <cstdint> - Standard utilities

===============================================================================
*/

#include "catch_amalgamated.hpp"
#include <gurobi_dsl/enum_utils.h>

#include <array>
#include <type_traits>
#include <cstdint>

// ============================================================================
// TEST SUPPORT TYPES
// ============================================================================

/// Custom enum not using DECLARE_ENUM_WITH_COUNT for specialization tests
enum class CustomEnum { First, Second, Third };

/// enum_size specialization for CustomEnum (must be at namespace scope)
template<>
struct enum_size<CustomEnum> {
    static constexpr std::size_t value = 3;
};

/// constexpr factorial helper for compile-time computation tests
constexpr std::size_t factorial(std::size_t n) noexcept {
    std::size_t result = 1;
    for (std::size_t i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// ============================================================================
// SECTION A: DECLARE_ENUM_WITH_COUNT MACRO BEHAVIOR
// ============================================================================

/**
 * @test MacroExpansion::CreatesValidEnumClass
 * @brief Verifies DECLARE_ENUM_WITH_COUNT produces a valid strongly-typed enum
 *
 * @scenario Macro declares an enum with three values
 * @given A DECLARE_ENUM_WITH_COUNT(TestEnum, A, B, C) declaration
 * @when Inspecting type traits, values, and COUNT sentinel
 * @then Enum is strongly-typed with sequential values 0..2 and COUNT=3
 *
 * @covers DECLARE_ENUM_WITH_COUNT macro
 */
TEST_CASE("A1: MacroExpansion::CreatesValidEnumClass", "[enum_utils][macro]")
{
    DECLARE_ENUM_WITH_COUNT(TestEnum, A, B, C);

    SECTION("Enum type is properly defined")
    {
        REQUIRE(std::is_enum_v<TestEnum>);
        REQUIRE_FALSE(std::is_convertible_v<TestEnum, int>);
        REQUIRE(std::is_same_v<std::underlying_type_t<TestEnum>, int>);
    }

    SECTION("Enumerators have correct sequential values")
    {
        REQUIRE(static_cast<int>(TestEnum::A) == 0);
        REQUIRE(static_cast<int>(TestEnum::B) == 1);
        REQUIRE(static_cast<int>(TestEnum::C) == 2);
        REQUIRE(static_cast<int>(TestEnum::COUNT) == 3);
    }

    SECTION("Size constant is correctly defined")
    {
        REQUIRE(TestEnum_COUNT == 3);
        REQUIRE(TestEnum_COUNT == static_cast<std::size_t>(TestEnum::COUNT));
        static_assert(TestEnum_COUNT == 3, "Compile-time constant check");
    }
}

/**
 * @test MacroExpansion::SingleEnumerator
 * @brief Verifies single-enumerator declaration works correctly
 *
 * @scenario Macro declares an enum with one value
 * @given A DECLARE_ENUM_WITH_COUNT(SingleEnum, OnlyValue) declaration
 * @when Inspecting values and COUNT
 * @then OnlyValue is 0 and COUNT equals 1
 *
 * @covers DECLARE_ENUM_WITH_COUNT with single enumerator
 */
TEST_CASE("A2: MacroExpansion::SingleEnumerator", "[enum_utils][macro]")
{
    DECLARE_ENUM_WITH_COUNT(SingleEnum, OnlyValue);

    REQUIRE(static_cast<int>(SingleEnum::OnlyValue) == 0);
    REQUIRE(static_cast<int>(SingleEnum::COUNT) == 1);
    REQUIRE(SingleEnum_COUNT == 1);

    SingleEnum e = SingleEnum::OnlyValue;
    REQUIRE(e == SingleEnum::OnlyValue);
}

/**
 * @test MacroExpansion::ManyEnumerators
 * @brief Verifies large enumerator list maintains sequential values
 *
 * @scenario Macro declares an enum with 20 values
 * @given A DECLARE_ENUM_WITH_COUNT with E00..E19 enumerators
 * @when Checking sequential values and COUNT
 * @then Values are 0..19 and COUNT equals 20
 *
 * @covers DECLARE_ENUM_WITH_COUNT with many enumerators
 */
TEST_CASE("A3: MacroExpansion::ManyEnumerators", "[enum_utils][macro]")
{
    DECLARE_ENUM_WITH_COUNT(LargeEnum,
        E00, E01, E02, E03, E04, E05, E06, E07, E08, E09,
        E10, E11, E12, E13, E14, E15, E16, E17, E18, E19
    );

    REQUIRE(LargeEnum_COUNT == 20);
    REQUIRE(static_cast<int>(LargeEnum::COUNT) == 20);
    REQUIRE(static_cast<int>(LargeEnum::E00) == 0);
    REQUIRE(static_cast<int>(LargeEnum::E19) == 19);
}

// ============================================================================
// SECTION B: ENUMARRAY TYPE-SAFE INDEXING
// ============================================================================

/**
 * @test EnumArray::TypeSafeIndexing
 * @brief Verifies EnumArray provides correct size and element access
 *
 * @scenario EnumArray indexed by Status enum stores string messages
 * @given An enum with three values and corresponding array
 * @when Accessing elements via enum ordinal positions
 * @then All elements are accessible with correct array size
 *
 * @covers EnumArray template
 */
TEST_CASE("B1: EnumArray::TypeSafeIndexing", "[enum_utils][array]")
{
    DECLARE_ENUM_WITH_COUNT(Status, Ok, Warning, Error);

    EnumArray<Status, std::string> messages = {
        "Success",
        "Proceed with caution",
        "Failure"
    };

    SECTION("Array has correct size")
    {
        REQUIRE(std::size(messages) == Status_COUNT);
        REQUIRE(std::size(messages) == 3);
    }

    SECTION("Elements accessible via enum ordinal")
    {
        REQUIRE(messages[static_cast<std::size_t>(Status::Ok)] == "Success");
        REQUIRE(messages[static_cast<std::size_t>(Status::Warning)] == "Proceed with caution");
        REQUIRE(messages[static_cast<std::size_t>(Status::Error)] == "Failure");
    }

    SECTION("Array iteration works correctly")
    {
        int count = 0;
        for (const auto& msg : messages) {
            ++count;
            REQUIRE(!msg.empty());
        }
        REQUIRE(count == Status_COUNT);
    }
}

/**
 * @test EnumArray::DifferentElementTypes
 * @brief Verifies EnumArray supports various element types
 *
 * @scenario EnumArray with integers, doubles, and custom structs
 * @given An enum indexing arrays of different types
 * @when Initializing and accessing elements
 * @then Values match expectations across all types
 *
 * @covers EnumArray with integral, floating-point, and struct types
 */
TEST_CASE("B2: EnumArray::DifferentElementTypes", "[enum_utils][array]")
{
    DECLARE_ENUM_WITH_COUNT(Dimensions, X, Y, Z);

    SECTION("With integers")
    {
        EnumArray<Dimensions, int> coords = { 10, 20, 30 };
        REQUIRE(coords[static_cast<std::size_t>(Dimensions::X)] == 10);
        REQUIRE(coords[static_cast<std::size_t>(Dimensions::Y)] == 20);
        REQUIRE(coords[static_cast<std::size_t>(Dimensions::Z)] == 30);
    }

    SECTION("With doubles")
    {
        EnumArray<Dimensions, double> weights = { 1.5, 2.5, 3.5 };
        REQUIRE(weights[static_cast<std::size_t>(Dimensions::X)] == 1.5);
    }

    SECTION("With custom types")
    {
        struct Point { int x, y; };
        EnumArray<Dimensions, Point> points = {
            Point{1, 2},
            Point{3, 4},
            Point{5, 6}
        };
        REQUIRE(points[static_cast<std::size_t>(Dimensions::Y)].x == 3);
    }
}

// ============================================================================
// SECTION C: ENUM_SIZE TRAIT FUNCTIONALITY
// ============================================================================

/**
 * @test EnumSizeTrait::UniformSizeAccess
 * @brief Verifies enum_size trait provides consistent size access
 *
 * @scenario enum_size<T>::value queried for macro-declared enum
 * @given An enum declared with DECLARE_ENUM_WITH_COUNT
 * @when Querying enum_size<Color>::value
 * @then Value equals the number of user enumerators (excludes COUNT)
 *
 * @covers enum_size<T>::value
 */
TEST_CASE("C1: EnumSizeTrait::UniformSizeAccess", "[enum_utils][trait]")
{
    DECLARE_ENUM_WITH_COUNT(Color, Red, Green, Blue);

    SECTION("Primary template works with macro-declared enums")
    {
        REQUIRE(enum_size<Color>::value == 3);
        static_assert(enum_size<Color>::value == 3, "Compile-time check");
        REQUIRE(enum_size<Color>::value == Color_COUNT);
    }

    SECTION("Can be used in constexpr contexts")
    {
        constexpr std::size_t size = enum_size<Color>::value;
        REQUIRE(size == 3);

        std::array<int, enum_size<Color>::value> arr{};
        REQUIRE(arr.size() == 3);
    }
}

/**
 * @test EnumSizeTrait::Specialization
 * @brief Verifies explicit specialization works for custom enums
 *
 * @scenario CustomEnum with manual enum_size specialization
 * @given A custom enum and its enum_size specialization
 * @when Querying enum_size<CustomEnum>::value
 * @then Value matches the specialization (3)
 *
 * @covers enum_size specialization
 */
TEST_CASE("C2: EnumSizeTrait::Specialization", "[enum_utils][trait]")
{
    REQUIRE(enum_size<CustomEnum>::value == 3);
    static_assert(enum_size<CustomEnum>::value == 3);
}

// ============================================================================
// SECTION D: IS_VALID_ENUM_VALUE BOUNDS CHECKING
// ============================================================================

/**
 * @test BoundsCheck::ValidAndInvalidValues
 * @brief Verifies is_valid_enum_value correctly identifies valid enumerators
 *
 * @scenario Bounds checks performed on in-range, COUNT, and out-of-range values
 * @given An enum with three values and trailing COUNT
 * @when Validating various values with is_valid_enum_value()
 * @then In-range returns true; COUNT and out-of-range return false
 *
 * @covers is_valid_enum_value<T>()
 */
TEST_CASE("D1: BoundsCheck::ValidAndInvalidValues", "[enum_utils][validation]")
{
    DECLARE_ENUM_WITH_COUNT(Level, Low, Medium, High);

    SECTION("Valid values return true")
    {
        REQUIRE(is_valid_enum_value(Level::Low));
        REQUIRE(is_valid_enum_value(Level::Medium));
        REQUIRE(is_valid_enum_value(Level::High));

        REQUIRE(is_valid_enum_value(static_cast<Level>(0)));
        REQUIRE(is_valid_enum_value(static_cast<Level>(1)));
        REQUIRE(is_valid_enum_value(static_cast<Level>(2)));
    }

    SECTION("COUNT sentinel returns false")
    {
        REQUIRE_FALSE(is_valid_enum_value(Level::COUNT));
    }

    SECTION("Out of bounds values return false")
    {
        REQUIRE_FALSE(is_valid_enum_value(static_cast<Level>(3)));
        REQUIRE_FALSE(is_valid_enum_value(static_cast<Level>(100)));
    }

    SECTION("Function is constexpr and noexcept")
    {
        static_assert(noexcept(is_valid_enum_value(Level::Low)));

        constexpr bool valid = is_valid_enum_value(Level::Medium);
        static_assert(valid);

        constexpr bool invalid = is_valid_enum_value(Level::COUNT);
        static_assert(!invalid);
    }
}

/**
 * @test BoundsCheck::SingleValueEnum
 * @brief Verifies single-value enum validation
 *
 * @scenario Bounds checks on enum with single enumerator
 * @given An enum with one value (Only) and COUNT
 * @when Validating Only, COUNT, and out-of-range
 * @then Only is valid, COUNT and out-of-range are invalid
 *
 * @covers is_valid_enum_value with single-element enum
 */
TEST_CASE("D2: BoundsCheck::SingleValueEnum", "[enum_utils][validation]")
{
    DECLARE_ENUM_WITH_COUNT(Singleton, Only);

    REQUIRE(is_valid_enum_value(Singleton::Only));
    REQUIRE_FALSE(is_valid_enum_value(Singleton::COUNT));
    REQUIRE_FALSE(is_valid_enum_value(static_cast<Singleton>(1)));
}

// ============================================================================
// SECTION E: ENUM_FROM_VALUE SAFE CONVERSIONS
// ============================================================================

/**
 * @test SafeConversion::IntegerToEnum
 * @brief Verifies enum_from_value safely converts integers to enums
 *
 * @scenario Integers 0..2 converted to Priority enum values
 * @given An enum with three priority levels
 * @when Converting integers with enum_from_value<Priority>()
 * @then Conversion yields corresponding enumerators
 *
 * @covers enum_from_value<T>()
 */
TEST_CASE("E1: SafeConversion::IntegerToEnum", "[enum_utils][conversion]")
{
    DECLARE_ENUM_WITH_COUNT(Priority, Low, Normal, High);

    SECTION("Valid conversions")
    {
        REQUIRE(enum_from_value<Priority>(0) == Priority::Low);
        REQUIRE(enum_from_value<Priority>(1) == Priority::Normal);
        REQUIRE(enum_from_value<Priority>(2) == Priority::High);
    }

    SECTION("Function is constexpr and noexcept")
    {
        static_assert(noexcept(enum_from_value<Priority>(0)));

        constexpr Priority p = enum_from_value<Priority>(1);
        static_assert(p == Priority::Normal);
    }

    SECTION("Can be used in array initialization")
    {
        constexpr std::array<Priority, 3> priorities = {
            enum_from_value<Priority>(0),
            enum_from_value<Priority>(1),
            enum_from_value<Priority>(2)
        };

        REQUIRE(priorities[0] == Priority::Low);
        REQUIRE(priorities[1] == Priority::Normal);
        REQUIRE(priorities[2] == Priority::High);
    }
}

// ============================================================================
// SECTION F: INTEGRATION AND REAL-WORLD USAGE
// ============================================================================

/**
 * @test Integration::EnumIndexedConfiguration
 * @brief Verifies enum-indexed configuration tables work correctly
 *
 * @scenario Configuration struct array indexed by Algorithm enum
 * @given An enum with three algorithm types and config struct
 * @when Accessing and aggregating configuration values
 * @then Data is consistent and safely accessible via enum ordinals
 *
 * @covers EnumArray with struct elements
 */
TEST_CASE("F1: Integration::EnumIndexedConfiguration", "[enum_utils][integration]")
{
    DECLARE_ENUM_WITH_COUNT(Algorithm, Greedy, Dynamic, Backtrack);

    struct Config {
        int max_iterations;
        double tolerance;
        bool verbose;
    };

    EnumArray<Algorithm, Config> configurations = {
        Config{1000, 1e-6, false},
        Config{500,  1e-8, true},
        Config{100,  1e-4, false}
    };

    SECTION("Access configurations safely")
    {
        const auto& greedy_cfg = configurations[static_cast<std::size_t>(Algorithm::Greedy)];
        REQUIRE(greedy_cfg.max_iterations == 1000);
        REQUIRE(greedy_cfg.tolerance == 1e-6);
        REQUIRE_FALSE(greedy_cfg.verbose);

        const auto& dynamic_cfg = configurations[static_cast<std::size_t>(Algorithm::Dynamic)];
        REQUIRE(dynamic_cfg.verbose);
    }

    SECTION("Iterate over all configurations")
    {
        int total_iterations = 0;
        for (std::size_t i = 0; i < Algorithm_COUNT; ++i) {
            total_iterations += configurations[i].max_iterations;
        }
        REQUIRE(total_iterations == 1600);
    }
}

/**
 * @test Integration::StateMachine
 * @brief Verifies state machine lookup tables work correctly
 *
 * @scenario State names and handlers indexed by State enum
 * @given An enum with four states and corresponding arrays
 * @when Simulating state transitions
 * @then Names and validation behave as expected
 *
 * @covers EnumArray with function pointers and strings
 */
TEST_CASE("F2: Integration::StateMachine", "[enum_utils][integration]")
{
    DECLARE_ENUM_WITH_COUNT(State, Idle, Processing, Completed, Error);

    using TransitionHandler = void(*)();

    EnumArray<State, TransitionHandler> on_enter = {
        []() { /* Idle enter */ },
        []() { /* Processing enter */ },
        []() { /* Completed enter */ },
        []() { /* Error enter */ }
    };

    EnumArray<State, std::string> state_names = {
        "Idle", "Processing", "Completed", "Error"
    };

    SECTION("State machine operations")
    {
        State current = State::Idle;
        REQUIRE(state_names[static_cast<std::size_t>(current)] == "Idle");

        current = State::Processing;
        REQUIRE(state_names[static_cast<std::size_t>(current)] == "Processing");
        REQUIRE(is_valid_enum_value(current));
    }
}

/**
 * @test Integration::CompileTimeArrays
 * @brief Verifies compile-time array sizes and validation logic
 *
 * @scenario Constexpr arrays and static_assert checks
 * @given An enum with three flag values
 * @when Using COUNT and enum_size in constexpr contexts
 * @then Arrays size correctly and checks pass at compile-time
 *
 * @covers Compile-time enum utilities
 */
TEST_CASE("F3: Integration::CompileTimeArrays", "[enum_utils][integration]")
{
    DECLARE_ENUM_WITH_COUNT(Flags, Read, Write, Execute);

    SECTION("Compile-time array sizes")
    {
        constexpr std::array<int, Flags_COUNT> flag_masks = { 0x1, 0x2, 0x4 };
        static_assert(flag_masks.size() == 3);

        constexpr std::array<const char*, enum_size<Flags>::value> names = {
            "READ", "WRITE", "EXECUTE"
        };
        static_assert(names.size() == 3);
    }

    SECTION("Compile-time validation")
    {
        static_assert(Flags_COUNT == 3);
        static_assert(enum_size<Flags>::value == 3);
        static_assert(is_valid_enum_value(Flags::Read));
        static_assert(!is_valid_enum_value(Flags::COUNT));
        static_assert(static_cast<int>(Flags::COUNT) == 3);
    }
}

/**
 * @test Integration::MultipleIndependentEnums
 * @brief Verifies macro re-use across multiple enums
 *
 * @scenario Several independent enum declarations used concurrently
 * @given Multiple DECLARE_ENUM_WITH_COUNT declarations
 * @when Using them simultaneously
 * @then Constants and values remain independent
 *
 * @covers Multiple enum declarations
 */
TEST_CASE("F4: Integration::MultipleIndependentEnums", "[enum_utils][integration]")
{
    DECLARE_ENUM_WITH_COUNT(Enum1, A, B);
    DECLARE_ENUM_WITH_COUNT(Enum2, X, Y, Z);
    DECLARE_ENUM_WITH_COUNT(Enum3, Only);

    SECTION("Each enum has independent constants")
    {
        REQUIRE(Enum1_COUNT == 2);
        REQUIRE(Enum2_COUNT == 3);
        REQUIRE(Enum3_COUNT == 1);

        REQUIRE(static_cast<int>(Enum1::COUNT) == 2);
        REQUIRE(static_cast<int>(Enum2::COUNT) == 3);
        REQUIRE(static_cast<int>(Enum3::COUNT) == 1);
    }

    SECTION("Can use all enums simultaneously")
    {
        EnumArray<Enum1, int> arr1 = { 1, 2 };
        EnumArray<Enum2, int> arr2 = { 10, 20, 30 };
        EnumArray<Enum3, int> arr3 = { 100 };

        REQUIRE(arr1[0] + arr2[1] + arr3[0] == 121);
    }
}

/**
 * @test Integration::EdgeCases
 * @brief Verifies edge cases and robustness
 *
 * @scenario Empty enums, reserved keyword-like names, large counts
 * @given Various edge-case enum declarations
 * @when Using the enum utilities
 * @then All edge cases are handled correctly
 *
 * @covers Edge cases and boundary conditions
 */
TEST_CASE("F5: Integration::EdgeCases", "[enum_utils][edge]")
{
    SECTION("Empty enumerator list")
    {
        DECLARE_EMPTY_ENUM_WITH_COUNT(EmptyEnum);

        REQUIRE(EmptyEnum_COUNT == 0);
        REQUIRE(static_cast<int>(EmptyEnum::COUNT) == 0);
        REQUIRE_FALSE(is_valid_enum_value(EmptyEnum::COUNT));
    }

    SECTION("Enum with reserved keyword-like names")
    {
        DECLARE_ENUM_WITH_COUNT(Keywords, class_, struct_, namespace_);

        REQUIRE(Keywords_COUNT == 3);
        REQUIRE(is_valid_enum_value(Keywords::class_));
        REQUIRE(is_valid_enum_value(Keywords::struct_));
        REQUIRE(is_valid_enum_value(Keywords::namespace_));
    }

    SECTION("Large enumerator count")
    {
        DECLARE_ENUM_WITH_COUNT(BigEnum,
            E00, E01, E02, E03, E04, E05, E06, E07, E08, E09,
            E10, E11, E12, E13, E14, E15, E16, E17, E18, E19,
            E20, E21, E22, E23, E24, E25, E26, E27, E28, E29
        );

        REQUIRE(BigEnum_COUNT == 30);

        for (int i = 0; i < 30; ++i) {
            auto value = static_cast<BigEnum>(i);
            REQUIRE(is_valid_enum_value(value));
            REQUIRE(static_cast<int>(value) == i);
        }

        REQUIRE_FALSE(is_valid_enum_value(BigEnum::COUNT));
    }
}

// ============================================================================
// SECTION G: CONSTEXPR VERIFICATION
// ============================================================================

/**
 * @test ConstexprVerification::AllUtilitiesConstexpr
 * @brief Verifies all utilities are constexpr where possible
 *
 * @scenario Constants, traits, and functions used in constexpr contexts
 * @given An enum with three values
 * @when Using COUNT, enum_size, and functions in static_assert
 * @then All applicable expressions compile and evaluate correctly
 *
 * @covers Constexpr behavior of enum utilities
 */
TEST_CASE("G1: ConstexprVerification::AllUtilitiesConstexpr", "[enum_utils][constexpr]")
{
    DECLARE_ENUM_WITH_COUNT(ConstExprTest, X, Y, Z);

    SECTION("Constants are compile-time evaluable")
    {
        constexpr std::size_t count = ConstExprTest_COUNT;
        static_assert(count == 3);

        constexpr auto size = enum_size<ConstExprTest>::value;
        static_assert(size == 3);

        constexpr std::array<int, ConstExprTest_COUNT> arr = { 1, 2, 3 };
        static_assert(arr.size() == 3);
        static_assert(arr[0] == 1);
    }

    SECTION("Functions are constexpr")
    {
        constexpr bool valid = is_valid_enum_value(ConstExprTest::Y);
        static_assert(valid);

        constexpr bool invalid = is_valid_enum_value(ConstExprTest::COUNT);
        static_assert(!invalid);

        constexpr auto from_val = enum_from_value<ConstExprTest>(1);
        static_assert(from_val == ConstExprTest::Y);
    }

    SECTION("Complex compile-time computations")
    {
        constexpr std::size_t fact = factorial(ConstExprTest_COUNT);
        static_assert(fact == 6);
        REQUIRE(fact == 6);
    }
}

/**
 * @test TypeTraits::EnumInteraction
 * @brief Verifies type trait interactions with enum utilities
 *
 * @scenario underlying_type and compile-time constant verification
 * @given An enum with three values
 * @when Using type traits and compile-time checks
 * @then All traits behave correctly and are constexpr-friendly
 *
 * @covers Type trait compatibility
 */
TEST_CASE("G2: TypeTraits::EnumInteraction", "[enum_utils][traits]")
{
    DECLARE_ENUM_WITH_COUNT(TestTraits, Alpha, Beta, Gamma);

    SECTION("Underlying type consistency")
    {
        using underlying = std::underlying_type_t<TestTraits>;
        REQUIRE(std::is_same_v<underlying, int>);

        static_assert(static_cast<underlying>(TestTraits::Gamma) == 2);
        static_assert(static_cast<underlying>(TestTraits::COUNT) == 3);
    }

    SECTION("Size constant as template argument")
    {
        constexpr std::size_t size = TestTraits_COUNT;
        static_assert(size == 3);

        std::array<int, TestTraits_COUNT> arr{};
        REQUIRE(arr.size() == 3);
        REQUIRE(arr.max_size() == 3);
    }

    SECTION("Enum values are distinct")
    {
        REQUIRE(TestTraits::Alpha != TestTraits::Beta);
        REQUIRE(TestTraits::Beta != TestTraits::Gamma);
        REQUIRE(TestTraits::Alpha != TestTraits::Gamma);
        REQUIRE(TestTraits::COUNT != TestTraits::Alpha);
    }
}

/**
 * @test AlgorithmIntegration::EnumArrayWithAlgorithms
 * @brief Verifies EnumArray works with standard algorithm patterns
 *
 * @scenario Iteration, searching, and transformation over EnumArray
 * @given An enum-indexed array of integers
 * @when Performing accumulate, find, and transform operations
 * @then Results match expectations
 *
 * @covers EnumArray with algorithmic patterns
 */
TEST_CASE("G3: AlgorithmIntegration::EnumArrayWithAlgorithms", "[enum_utils][algorithm]")
{
    DECLARE_ENUM_WITH_COUNT(Numbers, One, Two, Three, Four, Five);

    EnumArray<Numbers, int> values = { 1, 2, 3, 4, 5 };

    SECTION("Accumulation")
    {
        int sum = 0;
        for (std::size_t i = 0; i < Numbers_COUNT; ++i) {
            sum += values[i];
        }
        REQUIRE(sum == 15);
    }

    SECTION("Find by value")
    {
        std::size_t index = Numbers_COUNT;
        for (std::size_t i = 0; i < Numbers_COUNT; ++i) {
            if (values[i] == 3) {
                index = i;
                break;
            }
        }

        REQUIRE(index == static_cast<std::size_t>(Numbers::Three));
        REQUIRE(index == 2);
    }

    SECTION("Transform")
    {
        EnumArray<Numbers, int> doubled{};
        for (std::size_t i = 0; i < Numbers_COUNT; ++i) {
            doubled[i] = values[i] * 2;
        }

        REQUIRE(doubled[static_cast<std::size_t>(Numbers::One)] == 2);
        REQUIRE(doubled[static_cast<std::size_t>(Numbers::Five)] == 10);
    }
}

// ============================================================================
// SECTION H: ERROR SCENARIOS
// ============================================================================

/**
 * @test ErrorScenarios::InvalidUsagePatterns
 * @brief Verifies expected behavior for invalid usage patterns
 *
 * @scenario COUNT used as regular value, validation of sentinel
 * @given An enum with two values and COUNT sentinel
 * @when Using COUNT as a value or index
 * @then COUNT is distinct and invalid for domain use
 *
 * @covers Invalid usage detection
 */
TEST_CASE("H1: ErrorScenarios::InvalidUsagePatterns", "[enum_utils][error]")
{
    DECLARE_ENUM_WITH_COUNT(Test, A, B);

    SECTION("COUNT should not be used as regular value")
    {
        REQUIRE(Test::COUNT != Test::A);
        REQUIRE(Test::COUNT != Test::B);
        REQUIRE_FALSE(is_valid_enum_value(Test::COUNT));
    }

    SECTION("Proper enum declaration avoids conflicts")
    {
        // Let macro append COUNT, never define it manually
        DECLARE_ENUM_WITH_COUNT(GoodEnum, A, B);
        REQUIRE(GoodEnum_COUNT == 2);
    }
}