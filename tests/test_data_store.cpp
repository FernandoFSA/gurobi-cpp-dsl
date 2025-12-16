/*
===============================================================================
TEST DATA_STORE — Comprehensive tests for data_store.h
===============================================================================

OVERVIEW
--------
Validates the `Value` and `DataStore` components for type-safe storage,
access patterns, exception guarantees, and integration behavior. Tests cover
mutable/const retrieval, optional access, implicit conversion, lazy
computation, and strict type enforcement.

TEST ORGANIZATION
-----------------
• Section A: Type safety and exception behavior
• Section B: Lifecycle and type mutation
• Section C: Complex type storage (containers, strings)
• Section D: Optional access patterns via `try_get`
• Section E: Implicit conversion operator behavior
• Section F: Safe default access via `get_or`
• Section G: Flexible lazy computation via `getOrCompute`
• Section H: Strict type-enforced lazy computation via `getStrictOrCompute`
• Section I: DataStore integration tests (map semantics)

TEST STRATEGY
-------------
• Verify thrown exceptions with REQUIRE_THROWS and type checks with REQUIRE
• Confirm noexcept guarantees for safe access APIs
• Validate copy vs reference semantics using mutations and re-reads
• Exercise both empty and populated states across APIs

DEPENDENCIES
------------
• Catch2 v3.0+ - Test framework
• data_store.h - System under test
• <vector>, <string> - Container and string tests

BUILD CONFIGURATION NOTES
-------------------------
Tests do not depend on debug/release macros; behavior must be consistent across
build configurations.

===============================================================================
*/

#include "catch_amalgamated.hpp"
#include <gurobi_dsl/data_store.h>

#include <vector>
#include <string>

// ============================================================================
// SECTION A: TYPE SAFETY AND EXCEPTION BEHAVIOR
// ============================================================================

 /**
  * @test ValueAccess::ThrowsOnTypeMismatch
  * @brief Verifies that accessing stored values with incorrect types throws
  *
  * @scenario A Value stores an int, then attempts to access it as double/string
  * @given A DataStore with an integer value under key "x"
  * @when Accessing the value with get<double>() or get<std::string>()
  * @then std::bad_any_cast is thrown
  *
  * @covers Value::get<T>()
  * @covers Value::is<T>()
  */
TEST_CASE("A1: ValueAccess::ThrowsOnTypeMismatch", "[data_store][value][exception]")
{
    DataStore data;
    data["x"] = 10; // Store as int

    REQUIRE(data["x"].is<int>());

    SECTION("Accessing int as double throws bad_any_cast")
    {
        REQUIRE_THROWS_AS(data["x"].get<double>(), std::bad_any_cast);
    }

    SECTION("Accessing int as string throws bad_any_cast")
    {
        REQUIRE_THROWS_AS(data["x"].get<std::string>(), std::bad_any_cast);
    }

    SECTION("Type check passes for correct type")
    {
        REQUIRE_NOTHROW(data["x"].get<int>());
        REQUIRE(data["x"].get<int>() == 10);
    }
}

// ============================================================================
// SECTION B: VALUE LIFECYCLE MANAGEMENT
// ============================================================================

/**
 * @test ValueLifecycle::HasValueAndReset
 * @brief Verifies Value state management (empty, assigned, reset)
 *
 * @scenario A Value transitions through empty, assigned, and reset states
 * @given A default-constructed Value
 * @when Values are assigned and reset
 * @then has_value() and type() reflect correct states
 *
 * @covers Value::has_value()
 * @covers Value::type()
 * @covers Value::reset()
 */
TEST_CASE("B1: ValueLifecycle::HasValueAndReset", "[data_store][value][lifecycle]")
{
    Value v;

    SECTION("Default constructed Value is empty")
    {
        REQUIRE_FALSE(v.has_value());
        REQUIRE(v.type() == typeid(void));
    }

    SECTION("After assignment, has_value() returns true")
    {
        v = 5;
        REQUIRE(v.has_value());
        REQUIRE(v.is<int>());
        REQUIRE(v.type() == typeid(int));
    }

    SECTION("Reset clears stored value")
    {
        v = 3.14;
        REQUIRE(v.has_value());

        v.reset();
        REQUIRE_FALSE(v.has_value());
        REQUIRE(v.type() == typeid(void));
    }

    SECTION("Multiple assignments update type correctly")
    {
        v = 42;
        REQUIRE(v.is<int>());

        v = std::string{ "test" };
        REQUIRE(v.is<std::string>());
        REQUIRE_FALSE(v.is<int>());
    }
}

// ============================================================================
// SECTION C: VALUE TYPE MUTATION
// ============================================================================

/**
 * @test ValueMutation::OverwritingChangesType
 * @brief Verifies that assignment can change stored type
 *
 * @scenario A DataStore entry changes from int to double
 * @given A DataStore with an integer value
 * @when The value is reassigned as a double
 * @then The type changes and new value is accessible
 *
 * @covers Value::operator=
 * @covers Value::is<T>()
 */
TEST_CASE("C1: ValueMutation::OverwritingChangesType", "[data_store][value][mutation]")
{
    DataStore data;

    SECTION("Integer assignment sets correct type")
    {
        data["key"] = 42;
        REQUIRE(data["key"].is<int>());
        REQUIRE(data["key"].get<int>() == 42);
    }

    SECTION("Double assignment overwrites and changes type")
    {
        data["key"] = 42;
        data["key"] = 3.14;

        REQUIRE(data["key"].is<double>());
        REQUIRE_FALSE(data["key"].is<int>());
        REQUIRE(data["key"].get<double>() == Catch::Approx(3.14));
    }

    SECTION("Multiple type changes are supported")
    {
        data["key"] = 100;
        REQUIRE(data["key"].is<int>());

        data["key"] = true;
        REQUIRE(data["key"].is<bool>());

        data["key"] = std::vector<int>{ 1, 2, 3 };
        REQUIRE(data["key"].is<std::vector<int>>());
    }
}

// ============================================================================
// SECTION D: COMPLEX TYPE STORAGE
// ============================================================================

/**
 * @test ComplexTypes::StoringAndRetrievingContainers
 * @brief Verifies that complex types (vectors) can be stored and retrieved
 *
 * @scenario A std::vector<int> is stored in a Value
 * @given An empty Value
 * @when A vector is assigned and retrieved
 * @then The vector maintains its contents and can be modified
 *
 * @covers Value with complex types
 * @covers Value::get<T>() with references
 */
TEST_CASE("D1: ComplexTypes::StoringAndRetrievingContainers", "[data_store][value][complex_types]")
{
    DataStore data;

    SECTION("Vector can be stored and retrieved")
    {
        data["vec"] = std::vector<int>{ 1, 2, 3 };

        REQUIRE(data["vec"].is<std::vector<int>>());

        auto& v = data["vec"].get<std::vector<int>>();
        REQUIRE(v.size() == 3);
        REQUIRE(v[0] == 1);
        REQUIRE(v[1] == 2);
        REQUIRE(v[2] == 3);
    }

    SECTION("Retrieved reference allows modification")
    {
        data["vec"] = std::vector<int>{ 1, 2, 3 };
        auto& v = data["vec"].get<std::vector<int>>();

        v.push_back(4);
        REQUIRE(v.size() == 4);
        REQUIRE(v[3] == 4);

        // Verify modification persists
        auto& v2 = data["vec"].get<std::vector<int>>();
        REQUIRE(v2.size() == 4);
    }

    SECTION("Strings and nested containers work")
    {
        data["str"] = std::string{ "Hello, World!" };
        REQUIRE(data["str"].is<std::string>());
        REQUIRE(data["str"].get<std::string>() == "Hello, World!");
    }
}

// ============================================================================
// SECTION E: OPTIONAL ACCESS PATTERNS
// ============================================================================

/**
 * @test OptionalAccess::TryGetBehavior
 * @brief Verifies try_get() returns optional references correctly
 *
 * @scenario try_get() is used with matching and mismatching types
 * @given A DataStore with double and bool values
 * @when try_get() is called with various types
 * @then Returns optional with value on match, nullopt on mismatch
 *
 * @covers Value::try_get<T>()
 */
TEST_CASE("E1: OptionalAccess::TryGetBehavior", "[data_store][value][optional]")
{
    DataStore data;
    data["x"] = 3.14;
    data["flag"] = true;

    SECTION("try_get succeeds with matching types")
    {
        auto vx = data["x"].try_get<double>();
        REQUIRE(vx.has_value());
        REQUIRE(vx->get() == Catch::Approx(3.14));

        auto vflag = data["flag"].try_get<bool>();
        REQUIRE(vflag.has_value());
        REQUIRE(vflag->get() == true);
    }

    SECTION("try_get returns nullopt on type mismatch")
    {
        auto bad1 = data["x"].try_get<int>();
        REQUIRE_FALSE(bad1.has_value());

        auto bad2 = data["flag"].try_get<double>();
        REQUIRE_FALSE(bad2.has_value());
    }

    SECTION("try_get returns nullopt for missing keys (via default Value)")
    {
        // Note: operator[] creates default Value if key doesn't exist
        auto missing = data["nonexistent"].try_get<int>();
        REQUIRE_FALSE(missing.has_value());
    }
}

// ============================================================================
// SECTION F: IMPLICIT CONVERSION BEHAVIOR
// ============================================================================

/**
 * @test ImplicitConversion::OperatorTBehavior
 * @brief Verifies implicit conversion operator works correctly
 *
 * @scenario Values are implicitly converted to various types
 * @given A DataStore with double, bool, and string values
 * @when Implicit conversion is attempted
 * @then Converts on match, throws on mismatch
 *
 * @covers Value::operator T()
 */
TEST_CASE("F1: ImplicitConversion::OperatorTBehavior", "[data_store][value][conversion]")
{
    DataStore data;
    data["a"] = 1.5;
    data["b"] = true;
    data["s"] = std::string("abc");

    SECTION("Implicit conversions work with matching types")
    {
        double a = data["a"];   // operator double()
        bool b = data["b"];     // operator bool()
        std::string s = data["s"]; // operator std::string()

        REQUIRE(a == Catch::Approx(1.5));
        REQUIRE(b == true);
        REQUIRE(s == "abc");
    }

    SECTION("Implicit conversion throws on type mismatch")
    {
        REQUIRE_THROWS_AS(int(data["a"]), std::bad_any_cast);    // double -> int
        REQUIRE_THROWS_AS(bool(data["a"]), std::bad_any_cast);   // double -> bool
        REQUIRE_THROWS_AS(double(data["b"]), std::bad_any_cast); // bool -> double
        REQUIRE_THROWS_AS(int(data["s"]), std::bad_any_cast);    // string -> int
    }

    SECTION("Conversion creates copy, not reference")
    {
        double a = data["a"];
        a = 99.9;

        // Original value unchanged
        REQUIRE(data["a"].get<double>() == Catch::Approx(1.5));
    }
}

// ============================================================================
// SECTION G: SAFE DEFAULT ACCESS PATTERNS
// ============================================================================

/**
 * @test SafeAccess::GetOrBehavior
 * @brief Verifies get_or() provides safe access with defaults
 *
 * @scenario get_or() is used with matching, mismatching, and empty values
 * @given A DataStore with values and an empty Value
 * @when get_or() is called with various defaults
 * @then Returns stored value on match, default on mismatch or empty
 *
 * @covers Value::get_or<T>()
 */
TEST_CASE("G1: SafeAccess::GetOrBehavior", "[data_store][value][safe_access]")
{
    DataStore data;
    data["x"] = 3.14;
    data["flag"] = true;

    SECTION("get_or returns stored value when types match")
    {
        REQUIRE(data["x"].get_or<double>(0.0) == Catch::Approx(3.14));
        REQUIRE(data["flag"].get_or<bool>(false) == true);
    }

    SECTION("get_or returns default on type mismatch")
    {
        REQUIRE(data["x"].get_or<int>(99) == 99);
        REQUIRE(data["flag"].get_or<double>(123.4) == Catch::Approx(123.4));
    }

    SECTION("get_or returns default for empty Value")
    {
        Value v; // Empty
        REQUIRE(v.get_or<int>(7) == 7);
        REQUIRE(v.get_or<std::string>("default") == "default");
    }
}

// ============================================================================
// SECTION H: FLEXIBLE LAZY COMPUTATION
// ============================================================================

/**
 * @test LazyComputation::GetOrComputeFlexible
 * @brief Verifies flexible lazy computation with getOrCompute()
 *
 * @scenario getOrCompute() handles empty values, type matches, and mismatches
 * @given Values in various states (empty, int, double)
 * @when getOrCompute() is called with computation functions
 * @then Computes when needed, caches results, handles type changes
 *
 * @covers Value::getOrCompute<T>()
 */
TEST_CASE("H1: LazyComputation::GetOrComputeFlexible", "[data_store][value][lazy]")
{
    Value v;

    SECTION("Computes and stores when empty")
    {
        int callCount = 0;
        auto result = v.getOrCompute<double>([&] {
            callCount++;
            return 3.14;
            });

        REQUIRE(result == Catch::Approx(3.14));
        REQUIRE(callCount == 1);
        REQUIRE(v.is<double>());
        REQUIRE(v.get<double>() == Catch::Approx(3.14));
    }

    SECTION("Returns stored value on second call without recompute")
    {
        v = 2.0;

        int callCount = 0;
        auto result = v.getOrCompute<double>([&] {
            callCount++;
            return 99.9;
            });

        REQUIRE(result == Catch::Approx(2.0));
        REQUIRE(callCount == 0);
    }

    SECTION("Type mismatch triggers compute and stores new type")
    {
        v = 42; // Store as int

        auto result = v.getOrCompute<double>([&] {
            return 1.5;
            });

        REQUIRE(result == Catch::Approx(1.5));
        REQUIRE(v.is<double>());
        REQUIRE_FALSE(v.is<int>());
        REQUIRE(v.get<double>() == Catch::Approx(1.5));
    }

    SECTION("Exception in compute function propagates")
    {
        REQUIRE_THROWS_AS(
            v.getOrCompute<int>([&]() -> int {
                throw std::runtime_error("Compute failed");
                }),
            std::runtime_error
        );

        // Value remains empty after exception
        REQUIRE_FALSE(v.has_value());
    }
}

// ============================================================================
// SECTION I: STRICT TYPE-ENFORCED LAZY COMPUTATION
// ============================================================================

/**
 * @test StrictComputation::GetStrictOrCompute
 * @brief Verifies strict type-enforced lazy computation
 *
 * @scenario getStrictOrCompute() enforces type stability
 * @given Values in various states (empty, double, int)
 * @when getStrictOrCompute() is called
 * @then Computes only when empty, enforces type otherwise
 *
 * @covers Value::getStrictOrCompute<T>()
 */
TEST_CASE("I1: StrictComputation::GetStrictOrCompute", "[data_store][value][strict]")
{
    Value v;

    SECTION("Computes and stores when empty")
    {
        int callCount = 0;

        auto result = v.getStrictOrCompute<double>([&] {
            callCount++;
            return 7.7;
            });

        REQUIRE(result == Catch::Approx(7.7));
        REQUIRE(callCount == 1);
        REQUIRE(v.is<double>());
    }

    SECTION("Returns stored value when type matches")
    {
        v = 4.4;

        int callCount = 0;
        auto result = v.getStrictOrCompute<double>([&] {
            callCount++;
            return 999.9;
            });

        REQUIRE(result == Catch::Approx(4.4));
        REQUIRE(callCount == 0); // No computation
    }

    SECTION("Throws bad_any_cast when type mismatches")
    {
        v = 10; // Store as int

        REQUIRE_THROWS_AS(
            v.getStrictOrCompute<double>([&] {
                return 0.0;
                }),
            std::bad_any_cast
        );

        // Original value unchanged
        REQUIRE(v.is<int>());
        REQUIRE(v.get<int>() == 10);
    }

    SECTION("Exception in compute function propagates (empty case)")
    {
        REQUIRE_THROWS_AS(
            v.getStrictOrCompute<int>([&]() -> int {
                throw std::logic_error("Logic error");
                }),
            std::logic_error
        );

        // Value remains empty after exception
        REQUIRE_FALSE(v.has_value());
    }
}

// ============================================================================
// SECTION J: DATASTORE INTEGRATION TESTS
// ============================================================================

/**
 * @test DataStoreIntegration::MapBehavior
 * @brief Verifies DataStore (unordered_map) integration works correctly
 *
 * @scenario DataStore is used like a regular unordered_map with Values
 * @given An empty DataStore
 * @when Keys are inserted, accessed, and iterated
 * @then Behaves like std::unordered_map with Value values
 *
 * @covers DataStore type alias
 */
TEST_CASE("J1: DataStoreIntegration::MapBehavior", "[data_store][integration]")
{
    DataStore store;

    SECTION("Insert and retrieve values")
    {
        store["int"] = 42;
        store["double"] = 3.14;
        store["bool"] = true;
        store["string"] = std::string{ "test" };

        REQUIRE(store.size() == 4);
        REQUIRE(store["int"].get<int>() == 42);
        REQUIRE(store["double"].get<double>() == Catch::Approx(3.14));
        REQUIRE(store["bool"].get<bool>() == true);
        REQUIRE(store["string"].get<std::string>() == "test");
    }

    SECTION("Iteration works correctly")
    {
        store["a"] = 1;
        store["b"] = 2;
        store["c"] = 3;

        int sum = 0;
        for (const auto& [key, value] : store) {
            sum += value.get<int>();
        }

        REQUIRE(sum == 6);
        REQUIRE(store.size() == 3);
    }

    SECTION("Missing keys create default Values")
    {
        // operator[] creates default Value for missing keys
        auto& value = store["new_key"];
        REQUIRE_FALSE(value.has_value());

        value = 100;
        REQUIRE(store["new_key"].get<int>() == 100);
    }

    SECTION("find() and count() work as expected")
    {
        store["x"] = 10;

        auto it = store.find("x");
        REQUIRE(it != store.end());
        REQUIRE(it->second.get<int>() == 10);

        REQUIRE(store.count("x") == 1);
        REQUIRE(store.count("nonexistent") == 0);
    }
}