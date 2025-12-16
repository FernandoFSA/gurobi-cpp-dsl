/*
===============================================================================
TEST INDEXING — Comprehensive tests for indexing.h
===============================================================================

OVERVIEW
--------
Validates the DSL indexing system for index list construction, set operations,
range views, Cartesian products, filtered views, and printing utilities. Tests
cover IndexList construction, set operators (+, &, -, ^), RangeView iteration,
Cartesian product composition, and filter predicates.

TEST ORGANIZATION
-----------------
• Section A: IndexList construction and basic operations
• Section B: Set operations (union, intersection, difference, symmetric)
• Section C: RangeView construction and iteration
• Section D: Cartesian product basics and iteration order
• Section E: Filtered views (member .filter() and pipe syntax)
• Section F: Printing utilities
• Section G: Integration and complex scenarios
• Section H: Edge cases and error conditions
• Section I: Iterator properties and performance

TEST STRATEGY
-------------
• Verify construction from various sources (initializer_list, containers)
• Confirm set operation semantics and order preservation
• Validate lazy evaluation for RangeView and Cartesian products
• Exercise filter predicates with lambdas and captures

DEPENDENCIES
------------
• Catch2 v3.0+ - Test framework
• indexing.h - System under test
• <vector>, <list>, <set> - Container types for testing

===============================================================================
*/

#define CATCH_CONFIG_MAIN
#include "catch_amalgamated.hpp"
#include <gurobi_dsl/indexing.h>
#include <vector>
#include <list>
#include <set>
#include <sstream>
#include <algorithm>
#include <numeric>

using namespace dsl;

// ============================================================================
// SECTION A: INDEXLIST CONSTRUCTION AND BASIC OPERATIONS
// ============================================================================

/**
 * @test IndexListConstruction::DefaultAndInitializerList
 * @brief Verifies IndexList construction from various sources
 *
 * @scenario IndexList created via default, initializer_list, and containers
 * @given Various construction methods
 * @when Creating IndexList instances
 * @then Size, emptiness, and contents match expectations
 *
 * @covers IndexList constructors
 */
TEST_CASE("A1: IndexListConstruction::DefaultAndInitializerList", "[IndexList][construction]") {
    SECTION("Default constructor creates empty list") {
        IndexList empty;
        REQUIRE(empty.size() == 0);
        REQUIRE(empty.empty());
        REQUIRE(empty.begin() == empty.end());
    }

    SECTION("Initializer list constructor") {
        IndexList I{ 1, 3, 7 };
        REQUIRE(I.size() == 3);
        REQUIRE_FALSE(I.empty());

        std::vector<int> expected{ 1, 3, 7 };
        REQUIRE(std::equal(I.begin(), I.end(), expected.begin()));
    }

    SECTION("Duplicates are preserved") {
        IndexList I{ 1, 2, 2, 3, 1 };
        REQUIRE(I.size() == 5);
        std::vector<int> expected{ 1, 2, 2, 3, 1 };
        REQUIRE(std::equal(I.begin(), I.end(), expected.begin()));
    }

    SECTION("Construct from std::vector") {
        std::vector<int> v{ 4, 5, 6, 7 };
        IndexList I(v);
        REQUIRE(I.size() == 4);
        std::vector<int> expected{ 4, 5, 6, 7 };
        REQUIRE(std::equal(I.begin(), I.end(), expected.begin()));
    }

    SECTION("Construct from std::list") {
        std::list<int> l{ 10, 20, 30 };
        IndexList I(l);
        REQUIRE(I.size() == 3);
        std::vector<int> expected{ 10, 20, 30 };
        REQUIRE(std::equal(I.begin(), I.end(), expected.begin()));
    }

    SECTION("Construct from std::set") {
        std::set<int> s{ 5, 3, 8, 3 };
        IndexList I(s);
        REQUIRE(I.size() == 3);
        REQUIRE(I[0] == 3);
        REQUIRE(I[1] == 5);
        REQUIRE(I[2] == 8);
    }

    SECTION("Move constructor from vector") {
        std::vector<int> v{ 1, 2, 3 };
        IndexList I(std::move(v));
        REQUIRE(I.size() == 3);
        REQUIRE(I[0] == 1);
        REQUIRE(I[1] == 2);
        REQUIRE(I[2] == 3);
    }
}

/**
 * @test IndexListOperations::BasicMethods
 * @brief Verifies IndexList basic operations
 *
 * @scenario IndexList methods: size, empty, contains, operator[], iteration
 * @given An IndexList with values including duplicates
 * @when Calling various access methods
 * @then Methods return expected values
 *
 * @covers IndexList::size(), empty(), contains(), operator[]
 */
TEST_CASE("A2: IndexListOperations::BasicMethods", "[IndexList][operations]") {
    IndexList I{ 1, 3, 7, 3, 9 };

    SECTION("size() and empty()") {
        REQUIRE(I.size() == 5);
        REQUIRE_FALSE(I.empty());

        IndexList empty;
        REQUIRE(empty.size() == 0);
        REQUIRE(empty.empty());
    }

    SECTION("contains() method") {
        REQUIRE(I.contains(1));
        REQUIRE(I.contains(3));
        REQUIRE(I.contains(7));
        REQUIRE(I.contains(9));
        REQUIRE_FALSE(I.contains(0));
        REQUIRE_FALSE(I.contains(2));
        REQUIRE_FALSE(I.contains(10));

        IndexList with_dupes{ 1, 2, 2, 3 };
        REQUIRE(with_dupes.contains(2));
    }

    SECTION("Random access with operator[]") {
        REQUIRE(I[0] == 1);
        REQUIRE(I[1] == 3);
        REQUIRE(I[2] == 7);
        REQUIRE(I[3] == 3);
        REQUIRE(I[4] == 9);
    }

    SECTION("Iteration with range-based for") {
        std::vector<int> collected;
        for (int x : I) {
            collected.push_back(x);
        }
        std::vector<int> expected{ 1, 3, 7, 3, 9 };
        REQUIRE(collected == expected);
    }

    SECTION("push_back() and reserve()") {
        IndexList J;
        J.reserve(10);
        J.push_back(100);
        J.push_back(200);
        J.push_back(300);

        REQUIRE(J.size() == 3);
        REQUIRE(J[0] == 100);
        REQUIRE(J[1] == 200);
        REQUIRE(J[2] == 300);
    }

    SECTION("raw() method returns reference") {
        const auto& v = I.raw();
        REQUIRE(v.size() == 5);
        REQUIRE(v[0] == 1);
        REQUIRE(v[4] == 9);
        REQUIRE(&v == &I.raw());
    }
}

// ============================================================================
// SECTION B: SET OPERATIONS
// ============================================================================

/**
 * @test SetUnion::BasicAndOrderPreservation
 * @brief Verifies set union operator (+) behavior
 *
 * @scenario Union of two IndexLists with various configurations
 * @given Two IndexLists with overlapping and unique elements
 * @when Applying operator+
 * @then Result contains all unique elements, preserving A's order
 *
 * @covers operator+(IndexList, IndexList)
 */
TEST_CASE("B1: SetUnion::BasicAndOrderPreservation", "[IndexList][set][union]") {
    SECTION("Basic union") {
        IndexList A{ 1, 2, 3 };
        IndexList B{ 3, 4, 5 };

        auto U = A + B;
        REQUIRE(U.size() == 5);
        std::vector<int> expected{ 1, 2, 3, 4, 5 };
        REQUIRE(std::equal(U.begin(), U.end(), expected.begin()));
    }

    SECTION("Union preserves order of first operand") {
        IndexList A{ 3, 1, 2 };
        IndexList B{ 2, 4, 0 };

        auto U = A + B;
        // A's order preserved: 3, 1, 2
        // Then B's elements not in A: 4, 0 (in B's order)
        std::vector<int> expected{ 3, 1, 2, 4, 0 };
        REQUIRE(std::equal(U.begin(), U.end(), expected.begin()));
    }

    SECTION("Union with duplicates in first operand") {
        IndexList A{ 1, 2, 2, 3 };
        IndexList B{ 2, 3, 4 };

        auto U = A + B;
        // All duplicates from A preserved
        std::vector<int> expected{ 1, 2, 2, 3, 4 };
        REQUIRE(std::equal(U.begin(), U.end(), expected.begin()));
    }

    SECTION("Union with duplicates in second operand") {
        IndexList A{ 1, 2, 3 };
        IndexList B{ 3, 3, 4, 4 };

        auto U = A + B;
        // Only first occurrence of duplicates from B matter
        std::vector<int> expected{ 1, 2, 3, 4 };
        REQUIRE(std::equal(U.begin(), U.end(), expected.begin()));
    }

    SECTION("Union with empty sets") {
        IndexList A{ 1, 2, 3 };
        IndexList empty;

        REQUIRE((A + empty).size() == 3);
        REQUIRE((empty + A).size() == 3);
        REQUIRE((empty + empty).size() == 0);

        // Verify order preserved
        auto U1 = A + empty;
        std::vector<int> expected1{ 1, 2, 3 };
        REQUIRE(std::equal(U1.begin(), U1.end(), expected1.begin()));

        auto U2 = empty + A;
        std::vector<int> expected2{ 1, 2, 3 };
        REQUIRE(std::equal(U2.begin(), U2.end(), expected2.begin()));
    }

    SECTION("Union is commutative in terms of elements (but not order)") {
        IndexList A{ 1, 2, 3 };
        IndexList B{ 3, 4, 5 };

        auto U1 = A + B;
        auto U2 = B + A;

        // Same elements but different order
        std::sort(U1.begin(), U1.end());
        std::sort(U2.begin(), U2.end());
        REQUIRE(std::equal(U1.begin(), U1.end(), U2.begin()));
    }
}

/**
 * @test SetIntersection::BasicAndOrderPreservation
 * @brief Verifies set intersection operator (&) behavior
 *
 * @scenario Intersection of two IndexLists
 * @given Two IndexLists with common and unique elements
 * @when Applying operator&
 * @then Result contains only common elements, preserving A's order
 *
 * @covers operator&(IndexList, IndexList)
 */
TEST_CASE("B2: SetIntersection::BasicAndOrderPreservation", "[IndexList][set][intersection]") {
    SECTION("Basic intersection") {
        IndexList A{ 1, 2, 3, 4 };
        IndexList B{ 3, 4, 5, 6 };

        auto I = A & B;
        REQUIRE(I.size() == 2);
        std::vector<int> expected{ 3, 4 };
        REQUIRE(std::equal(I.begin(), I.end(), expected.begin()));
    }

    SECTION("Intersection preserves order of first operand") {
        IndexList A{ 4, 2, 3, 1 };
        IndexList B{ 1, 3, 5 };

        auto I = A & B;
        // Order from A: 4 (no), 2 (no), 3 (yes), 1 (yes)
        std::vector<int> expected{ 3, 1 };
        REQUIRE(std::equal(I.begin(), I.end(), expected.begin()));
    }

    SECTION("Intersection with duplicates") {
        IndexList A{ 1, 2, 2, 3, 2 };
        IndexList B{ 2, 3, 4 };

        auto I = A & B;
        // All occurrences of 2 and 3 from A that are in B
        std::vector<int> expected{ 2, 2, 3, 2 };
        REQUIRE(std::equal(I.begin(), I.end(), expected.begin()));
    }

    SECTION("Empty intersection") {
        IndexList A{ 1, 2, 3 };
        IndexList B{ 4, 5, 6 };

        auto I = A & B;
        REQUIRE(I.empty());
        REQUIRE(I.size() == 0);
    }

    SECTION("Intersection with empty set") {
        IndexList A{ 1, 2, 3 };
        IndexList empty;

        auto I1 = A & empty;
        REQUIRE(I1.empty());

        auto I2 = empty & A;
        REQUIRE(I2.empty());
    }

    SECTION("Self intersection") {
        IndexList A{ 1, 2, 3 };
        auto I = A & A;
        REQUIRE(I.size() == 3);
        REQUIRE(std::equal(A.begin(), A.end(), I.begin()));
    }
}

/**
 * @test SetDifference::BasicAndOrderPreservation
 * @brief Verifies set difference operator (-) behavior
 *
 * @scenario Difference of two IndexLists
 * @given Two IndexLists with overlapping elements
 * @when Applying operator-
 * @then Result contains A's elements not in B, preserving order
 *
 * @covers operator-(IndexList, IndexList)
 */
TEST_CASE("B3: SetDifference::BasicAndOrderPreservation", "[IndexList][set][difference]") {
    SECTION("Basic difference") {
        IndexList A{ 1, 2, 3, 4, 5 };
        IndexList B{ 2, 4, 6 };

        auto D = A - B;
        REQUIRE(D.size() == 3);
        std::vector<int> expected{ 1, 3, 5 };
        REQUIRE(std::equal(D.begin(), D.end(), expected.begin()));
    }

    SECTION("Difference preserves order of first operand") {
        IndexList A{ 5, 3, 1, 4, 2 };
        IndexList B{ 1, 4 };

        auto D = A - B;
        // Keep: 5 (not in B), 3 (not in B), 1 (in B - remove), 4 (in B - remove), 2 (not in B)
        std::vector<int> expected{ 5, 3, 2 };
        REQUIRE(std::equal(D.begin(), D.end(), expected.begin()));
    }

    SECTION("Difference with duplicates") {
        IndexList A{ 1, 2, 2, 3, 2, 4 };
        IndexList B{ 2, 4 };

        auto D = A - B;
        // Remove all occurrences of 2 and 4
        std::vector<int> expected{ 1, 3 };
        REQUIRE(std::equal(D.begin(), D.end(), expected.begin()));
    }

    SECTION("Complete difference (all elements removed)") {
        IndexList A{ 1, 2, 3 };
        IndexList B{ 1, 2, 3, 4 };  // B superset of A

        auto D = A - B;
        REQUIRE(D.empty());
    }

    SECTION("Empty difference (no elements removed)") {
        IndexList A{ 1, 2, 3 };
        IndexList B{ 4, 5, 6 };

        auto D = A - B;
        REQUIRE(D.size() == 3);
        REQUIRE(std::equal(A.begin(), A.end(), D.begin()));
    }

    SECTION("Difference with empty set") {
        IndexList A{ 1, 2, 3 };
        IndexList empty;

        auto D1 = A - empty;
        REQUIRE(D1.size() == 3);
        REQUIRE(std::equal(A.begin(), A.end(), D1.begin()));

        auto D2 = empty - A;
        REQUIRE(D2.empty());
    }
}

/**
 * @test SetSymmetricDifference::BasicAndCommutativity
 * @brief Verifies symmetric difference operator (^) behavior
 *
 * @scenario Symmetric difference of two IndexLists
 * @given Two IndexLists with overlapping elements
 * @when Applying operator^
 * @then Result contains elements in exactly one set
 *
 * @covers operator^(IndexList, IndexList)
 */
TEST_CASE("B4: SetSymmetricDifference::BasicAndCommutativity", "[IndexList][set][symmetric]") {
    SECTION("Basic symmetric difference") {
        IndexList A{ 1, 2, 3, 4 };
        IndexList B{ 3, 4, 5, 6 };

        auto S = A ^ B;
        REQUIRE(S.size() == 4);
        // Order: (A - B) = {1, 2} then (B - A) = {5, 6}
        std::vector<int> expected{ 1, 2, 5, 6 };
        REQUIRE(std::equal(S.begin(), S.end(), expected.begin()));
    }

    SECTION("Symmetric difference is commutative") {
        IndexList A{ 1, 2, 3 };
        IndexList B{ 3, 4, 5 };

        auto S1 = A ^ B;
        auto S2 = B ^ A;

        // Same elements but different order
        // A ^ B: (A-B)={1,2} + (B-A)={4,5} = {1,2,4,5}
        // B ^ A: (B-A)={4,5} + (A-B)={1,2} = {4,5,1,2}
        REQUIRE(S1.size() == 4);
        REQUIRE(S2.size() == 4);

        // Sort to compare element equality
        std::vector<int> v1(S1.begin(), S1.end());
        std::vector<int> v2(S2.begin(), S2.end());
        std::sort(v1.begin(), v1.end());
        std::sort(v2.begin(), v2.end());
        REQUIRE(v1 == v2);
    }

    SECTION("Symmetric difference with duplicates") {
        IndexList A{ 1, 2, 2, 3 };
        IndexList B{ 2, 3, 3, 4 };

        auto S = A ^ B;
        // A-B: remove all 2s and 3s from A -> {1}
        // B-A: remove all 2s and 3s from B -> {4}
        std::vector<int> expected{ 1, 4 };
        REQUIRE(std::equal(S.begin(), S.end(), expected.begin()));
    }

    SECTION("Symmetric difference with empty sets") {
        IndexList A{ 1, 2, 3 };
        IndexList empty;

        auto S1 = A ^ empty;
        REQUIRE(S1.size() == 3);
        REQUIRE(std::equal(A.begin(), A.end(), S1.begin()));

        auto S2 = empty ^ A;
        REQUIRE(S2.size() == 3);
        REQUIRE(std::equal(A.begin(), A.end(), S2.begin()));

        auto S3 = empty ^ empty;
        REQUIRE(S3.empty());
    }

    SECTION("Symmetric difference of identical sets") {
        IndexList A{ 1, 2, 3 };

        auto S = A ^ A;
        REQUIRE(S.empty());
    }
}

// ============================================================================
// SECTION C: RANGEVIEW CONSTRUCTION AND ITERATION
// ============================================================================

/**
 * @test RangeViewConstruction::BasicProperties
 * @brief Verifies RangeView construction and basic properties
 *
 * @scenario RangeView with begin, end, and optional step
 * @given Various RangeView configurations
 * @when Checking size, emptiness, and element access
 * @then Properties match expected values
 *
 * @covers RangeView constructors
 */
TEST_CASE("C1: RangeViewConstruction::BasicProperties", "[RangeView][construction]") {
    SECTION("Basic range [begin, end)") {
        RangeView R(0, 5);
        REQUIRE(R.size() == 5);
        REQUIRE_FALSE(R.empty());

        std::vector<int> expected{ 0, 1, 2, 3, 4 };
        REQUIRE(std::equal(R.begin(), R.end(), expected.begin()));
    }

    SECTION("Range with step") {
        RangeView R(0, 10, 3);
        REQUIRE(R.size() == 4);  // 0, 3, 6, 9
        REQUIRE_FALSE(R.empty());

        std::vector<int> expected{ 0, 3, 6, 9 };
        REQUIRE(std::equal(R.begin(), R.end(), expected.begin()));
    }

    SECTION("Empty ranges") {
        // end <= begin
        RangeView R1(5, 0);
        REQUIRE(R1.empty());
        REQUIRE(R1.size() == 0);

        // end == begin
        RangeView R2(5, 5);
        REQUIRE(R2.empty());
        REQUIRE(R2.size() == 0);

        // step <= 0
        RangeView R3(0, 10, 0);
        REQUIRE(R3.empty());
        REQUIRE(R3.size() == 0);

        RangeView R4(0, 10, -1);
        REQUIRE(R4.empty());
        REQUIRE(R4.size() == 0);
    }

    SECTION("Single element range") {
        RangeView R(7, 8);
        REQUIRE(R.size() == 1);
        REQUIRE_FALSE(R.empty());

        std::vector<int> expected{ 7 };
        REQUIRE(std::equal(R.begin(), R.end(), expected.begin()));
    }

    SECTION("Random access via operator[]") {
        RangeView R(10, 20, 2);  // 10, 12, 14, 16, 18
        REQUIRE(R.size() == 5);

        REQUIRE(R[0] == 10);
        REQUIRE(R[1] == 12);
        REQUIRE(R[2] == 14);
        REQUIRE(R[3] == 16);
        REQUIRE(R[4] == 18);
    }

    SECTION("Large step that exceeds range") {
        RangeView R(0, 5, 10);  // Only 0 fits
        REQUIRE(R.size() == 1);
        REQUIRE(R[0] == 0);

        std::vector<int> expected{ 0 };
        REQUIRE(std::equal(R.begin(), R.end(), expected.begin()));
    }

    SECTION("Default constructor") {
        RangeView R;
        REQUIRE(R.empty());
        REQUIRE(R.size() == 0);
        REQUIRE(R.begin() == R.end());
    }
}

/**
 * @test RangeViewIteration::ForwardIteration
 * @brief Verifies RangeView iteration behavior
 *
 * @scenario Iterating through RangeView elements
 * @given A RangeView with step
 * @when Using iterators and range-based for
 * @then Elements are accessed in correct order
 *
 * @covers RangeView::iterator
 */
TEST_CASE("C2: RangeViewIteration::ForwardIteration", "[RangeView][iteration]") {
    SECTION("Forward iteration") {
        RangeView R(2, 7, 2);  // 2, 4, 6
        auto it = R.begin();
        REQUIRE(*it == 2);
        ++it;
        REQUIRE(*it == 4);
        ++it;
        REQUIRE(*it == 6);
        ++it;
        REQUIRE(it == R.end());
    }

    SECTION("Range-based for loop") {
        RangeView R(5, 9);  // 5, 6, 7, 8
        std::vector<int> collected;
        for (int x : R) {
            collected.push_back(x);
        }
        std::vector<int> expected{ 5, 6, 7, 8 };
        REQUIRE(collected == expected);
    }

    SECTION("Iterator equality") {
        RangeView R(0, 3);
        auto it1 = R.begin();
        auto it2 = R.begin();
        REQUIRE(it1 == it2);
        ++it1;
        REQUIRE(it1 != it2);

        auto end1 = R.end();
        auto end2 = R.end();
        REQUIRE(end1 == end2);
    }

    SECTION("End iterator is one past last element") {
        RangeView R(0, 3);  // 0, 1, 2
        auto it = R.begin();
        ++it; ++it; ++it;
        REQUIRE(it == R.end());
        // Can't dereference end iterator
    }
}

/**
 * @test RangeViewHelper::HelperFunction
 * @brief Verifies range_view() helper function
 *
 * @scenario Creating RangeView via helper function
 * @given Various parameters for range_view()
 * @when Calling the helper function
 * @then Returns correctly configured RangeView
 *
 * @covers range_view() helper
 */
TEST_CASE("C3: RangeViewHelper::HelperFunction", "[RangeView][helper]") {
    SECTION("Default step = 1") {
        auto R = range_view(0, 5);
        REQUIRE(R.size() == 5);
        REQUIRE(R[0] == 0);
        REQUIRE(R[4] == 4);
    }

    SECTION("With explicit step") {
        auto R = range_view(0, 10, 3);
        REQUIRE(R.size() == 4);
        REQUIRE(R[0] == 0);
        REQUIRE(R[3] == 9);
    }

    SECTION("Empty range") {
        auto R = range_view(5, 0);
        REQUIRE(R.empty());
        REQUIRE(R.size() == 0);
    }
}

/**
 * @test RangeMaterialized::RangeHelper
 * @brief Verifies range() materialized range helper
 *
 * @scenario Creating IndexList via range() helper
 * @given Range parameters
 * @when Calling range() function
 * @then Returns IndexList with expected values
 *
 * @covers range() helper
 */
TEST_CASE("C4: RangeMaterialized::RangeHelper", "[range][helper]") {
    SECTION("Basic range") {
        auto I = range(2, 5);
        REQUIRE(I.size() == 3);
        REQUIRE(I[0] == 2);
        REQUIRE(I[1] == 3);
        REQUIRE(I[2] == 4);
    }

    SECTION("Empty range") {
        auto I = range(5, 2);  // end < begin
        REQUIRE(I.empty());
        REQUIRE(I.size() == 0);

        auto I2 = range(3, 3);  // end == begin
        REQUIRE(I2.empty());
        REQUIRE(I2.size() == 0);
    }

    SECTION("Single element range") {
        auto I = range(7, 8);
        REQUIRE(I.size() == 1);
        REQUIRE(I[0] == 7);
    }

    SECTION("Compare with RangeView") {
        auto materialized = range(0, 5);
        auto lazy = range_view(0, 5);

        REQUIRE(materialized.size() == lazy.size());
        for (std::size_t i = 0; i < materialized.size(); ++i) {
            REQUIRE(materialized[i] == lazy[i]);
        }
    }
}

// ============================================================================
// SECTION D: CARTESIAN PRODUCT BASICS AND ITERATION ORDER
// ============================================================================

/**
 * @test CartesianBasics::ProductConstruction
 * @brief Verifies Cartesian product basic functionality
 *
 * @scenario 2D and 3D Cartesian products of IndexLists
 * @given Two or three IndexLists
 * @when Applying operator*
 * @then Product contains all combinations with correct size
 *
 * @covers operator*(IndexList, IndexList)
 */
TEST_CASE("D1: CartesianBasics::ProductConstruction", "[Cartesian][construction]") {
    SECTION("2D Cartesian product") {
        IndexList A{ 1, 2 };
        IndexList B{ 10, 20 };

        auto P = A * B;
        REQUIRE(P.size() == 4);

        std::vector<std::pair<int, int>> expected{
            {1, 10}, {1, 20}, {2, 10}, {2, 20}
        };

        std::size_t i = 0;
        for (auto [a, b] : P) {
            REQUIRE(a == expected[i].first);
            REQUIRE(b == expected[i].second);
            ++i;
        }
        REQUIRE(i == 4);
    }

    SECTION("Cartesian product with RangeView") {
        auto R = range_view(0, 2);  // 0, 1
        IndexList B{ 10, 20 };

        auto P = R * B;
        REQUIRE(P.size() == 4);

        std::vector<std::pair<int, int>> collected;
        for (auto [r, b] : P) {
            collected.emplace_back(r, b);
        }

        std::vector<std::pair<int, int>> expected{
            {0, 10}, {0, 20}, {1, 10}, {1, 20}
        };
        REQUIRE(collected == expected);
    }

    SECTION("3D Cartesian product") {
        IndexList A{ 1, 2 };
        IndexList B{ 10, 20 };
        IndexList C{ 100 };

        auto P = A * B * C;
        REQUIRE(P.size() == 4);  // 2 * 2 * 1

        std::vector<std::tuple<int, int, int>> expected{
            {1, 10, 100}, {1, 20, 100}, {2, 10, 100}, {2, 20, 100}
        };

        std::size_t i = 0;
        for (auto [a, b, c] : P) {
            REQUIRE(a == std::get<0>(expected[i]));
            REQUIRE(b == std::get<1>(expected[i]));
            REQUIRE(c == std::get<2>(expected[i]));
            ++i;
        }
        REQUIRE(i == 4);
    }

    SECTION("Empty Cartesian product") {
        IndexList A{ 1, 2 };
        IndexList empty;

        auto P1 = A * empty;
        REQUIRE(P1.empty());
        REQUIRE(P1.size() == 0);
        for (auto x : P1) {
            (void)x;  // Should never execute
            REQUIRE(false);
        }

        auto P2 = empty * A;
        REQUIRE(P2.empty());
        REQUIRE(P2.size() == 0);

        auto P3 = empty * empty;
        REQUIRE(P3.empty());
        REQUIRE(P3.size() == 0);
    }

    SECTION("Single element sets") {
        IndexList A{ 1 };
        IndexList B{ 10 };

        auto P = A * B;
        REQUIRE(P.size() == 1);

        auto it = P.begin();
        auto [a, b] = *it;
        REQUIRE(a == 1);
        REQUIRE(b == 10);

        ++it;
        REQUIRE(it == P.end());
    }
}

/**
 * @test CartesianIteration::LexicographicOrder
 * @brief Verifies Cartesian product iteration order
 *
 * @scenario Iterating through Cartesian product
 * @given A Cartesian product of IndexLists
 * @when Collecting elements during iteration
 * @then Elements appear in lexicographic order
 *
 * @covers Cartesian::iterator
 */
TEST_CASE("D2: CartesianIteration::LexicographicOrder", "[Cartesian][iteration]") {
    SECTION("Lexicographic order (nested loops)") {
        IndexList A{ 1, 2, 3 };
        IndexList B{ 10, 20 };

        auto P = A * B;

        std::vector<std::pair<int, int>> collected;
        for (auto [a, b] : P) {
            collected.emplace_back(a, b);
        }

        // Expected order: a varies slowest, b varies fastest
        std::vector<std::pair<int, int>> expected{
            {1, 10}, {1, 20},
            {2, 10}, {2, 20},
            {3, 10}, {3, 20}
        };
        REQUIRE(collected == expected);
    }

    SECTION("3D lexicographic order") {
        IndexList A{ 1, 2 };
        IndexList B{ 10, 20 };
        IndexList C{ 100, 200 };

        auto P = A * B * C;

        std::vector<std::tuple<int, int, int>> collected;
        for (auto [a, b, c] : P) {
            collected.emplace_back(a, b, c);
        }

        std::vector<std::tuple<int, int, int>> expected{
            {1, 10, 100}, {1, 10, 200},
            {1, 20, 100}, {1, 20, 200},
            {2, 10, 100}, {2, 10, 200},
            {2, 20, 100}, {2, 20, 200}
        };
        REQUIRE(collected == expected);
    }

    SECTION("Iterator increment") {
        IndexList A{ 1, 2 };
        IndexList B{ 10, 20 };

        auto P = A * B;
        auto it = P.begin();

        REQUIRE(std::get<0>(*it) == 1);
        REQUIRE(std::get<1>(*it) == 10);

        ++it;
        REQUIRE(std::get<0>(*it) == 1);
        REQUIRE(std::get<1>(*it) == 20);

        ++it;
        REQUIRE(std::get<0>(*it) == 2);
        REQUIRE(std::get<1>(*it) == 10);

        ++it;
        REQUIRE(std::get<0>(*it) == 2);
        REQUIRE(std::get<1>(*it) == 20);

        ++it;
        REQUIRE(it == P.end());
    }

    SECTION("Begin and end equality for empty product") {
        IndexList empty1, empty2;
        auto P = empty1 * empty2;
        REQUIRE(P.begin() == P.end());
    }
}

/**
 * @test CartesianMixed::MixedTypes
 * @brief Verifies Cartesian product with mixed types
 *
 * @scenario Cartesian product of IndexList and RangeView
 * @given Mixed index source types
 * @when Applying operator*
 * @then Product works correctly with mixed sources
 *
 * @covers Cartesian with RangeView
 */
TEST_CASE("D3: CartesianMixed::MixedTypes", "[Cartesian][mixed]") {
    SECTION("IndexList * RangeView") {
        IndexList A{ 1, 2 };
        auto R = range_view(10, 12);  // 10, 11

        auto P = A * R;
        REQUIRE(P.size() == 4);

        std::vector<std::pair<int, int>> collected;
        for (auto [a, r] : P) {
            collected.emplace_back(a, r);
        }

        std::vector<std::pair<int, int>> expected{
            {1, 10}, {1, 11}, {2, 10}, {2, 11}
        };
        REQUIRE(collected == expected);
    }

    SECTION("RangeView * RangeView") {
        auto R1 = range_view(0, 2);  // 0, 1
        auto R2 = range_view(10, 12);  // 10, 11

        auto P = R1 * R2;
        REQUIRE(P.size() == 4);

        std::vector<std::pair<int, int>> collected;
        for (auto [r1, r2] : P) {
            collected.emplace_back(r1, r2);
        }

        std::vector<std::pair<int, int>> expected{
            {0, 10}, {0, 11}, {1, 10}, {1, 11}
        };
        REQUIRE(collected == expected);
    }

    SECTION("Chained products") {
        IndexList A{ 1 };
        auto R = range_view(10, 12);  // 10, 11
        IndexList C{ 100, 200 };

        auto P1 = A * R;
        auto P2 = P1 * C;  // Cartesian<IndexList, RangeView, IndexList>

        REQUIRE(P2.size() == 4);  // 1 * 2 * 2

        std::vector<std::tuple<int, int, int>> collected;
        for (auto [a, r, c] : P2) {
            collected.emplace_back(a, r, c);
        }

        std::vector<std::tuple<int, int, int>> expected{
            {1, 10, 100}, {1, 10, 200},
            {1, 11, 100}, {1, 11, 200}
        };
        REQUIRE(collected == expected);
    }

    SECTION("Commutative extension") {
        IndexList A{ 1, 2 };
        IndexList B{ 10, 20 };

        auto P1 = A * B;
        auto P2 = P1 * A;  // Extend with A on the right

        REQUIRE(P2.size() == 8);  // 2 * 2 * 2

        // Check first few elements
        auto it = P2.begin();
        auto [a1, b1, a2] = *it;
        REQUIRE(a1 == 1);
        REQUIRE(b1 == 10);
        REQUIRE(a2 == 1);

        ++it;
        auto [a3, b2, a4] = *it;
        REQUIRE(a3 == 1);
        REQUIRE(b2 == 10);
        REQUIRE(a4 == 2);
    }
}

// ============================================================================
// SECTION E: FILTERED VIEWS
// ============================================================================

/**
 * @test FilterIndexList::MemberFilter
 * @brief Verifies IndexList.filter() member function
 *
 * @scenario Filtering IndexList with predicates
 * @given An IndexList with values
 * @when Applying .filter() with lambdas
 * @then Only matching elements are iterated
 *
 * @covers IndexList::filter()
 */
TEST_CASE("E1: FilterIndexList::MemberFilter", "[filter][IndexList]") {
    SECTION("Single predicate") {
        IndexList I{ 1, 2, 3, 4, 5, 6 };

        auto even = I.filter([](int x) { return x % 2 == 0; });

        std::vector<int> collected;
        for (int x : even) {
            collected.push_back(x);
        }

        std::vector<int> expected{ 2, 4, 6 };
        REQUIRE(collected == expected);
    }

    SECTION("Multiple predicates") {
        IndexList I{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        auto filtered = I.filter(
            [](int x) { return x % 2 == 0; },   // even
            [](int x) { return x > 3; },        // > 3
            [](int x) { return x < 9; }         // < 9
        );

        std::vector<int> collected;
        for (int x : filtered) {
            collected.push_back(x);
        }

        std::vector<int> expected{ 4, 6, 8 };
        REQUIRE(collected == expected);
    }

    SECTION("Empty result") {
        IndexList I{ 1, 3, 5, 7 };

        auto even = I.filter([](int x) { return x % 2 == 0; });

        REQUIRE(even.begin() == even.end());
        std::vector<int> collected;
        for (int x : even) {
            collected.push_back(x);
        }
        REQUIRE(collected.empty());
    }

    SECTION("All elements pass") {
        IndexList I{ 2, 4, 6, 8 };

        auto even = I.filter([](int x) { return x % 2 == 0; });

        std::vector<int> collected;
        for (int x : even) {
            collected.push_back(x);
        }

        REQUIRE(collected.size() == 4);
        REQUIRE(std::equal(I.begin(), I.end(), collected.begin()));
    }

    SECTION("Predicate with capture") {
        int threshold = 4;
        IndexList I{ 1, 2, 3, 4, 5, 6 };

        auto filtered = I.filter([threshold](int x) { return x > threshold; });

        std::vector<int> collected;
        for (int x : filtered) {
            collected.push_back(x);
        }

        std::vector<int> expected{ 5, 6 };
        REQUIRE(collected == expected);
    }
}

/**
 * @test FilterRangeView::MemberFilter
 * @brief Verifies RangeView.filter() member function
 *
 * @scenario Filtering RangeView with predicates
 * @given A RangeView
 * @when Applying .filter() with lambdas
 * @then Only matching elements are iterated
 *
 * @covers RangeView::filter()
 */
TEST_CASE("E2: FilterRangeView::MemberFilter", "[filter][RangeView]") {
    SECTION("Filter range view") {
        auto R = range_view(0, 10);  // 0..9

        auto even = R.filter([](int x) { return x % 2 == 0; });

        std::vector<int> collected;
        for (int x : even) {
            collected.push_back(x);
        }

        std::vector<int> expected{ 0, 2, 4, 6, 8 };
        REQUIRE(collected == expected);
    }

    SECTION("Filter stepped range") {
        auto R = range_view(0, 20, 3);  // 0, 3, 6, 9, 12, 15, 18

        auto filtered = R.filter(
            [](int x) { return x % 2 == 0; },   // even
            [](int x) { return x > 5; }         // > 5
        );

        std::vector<int> collected;
        for (int x : filtered) {
            collected.push_back(x);
        }

        std::vector<int> expected{ 6, 12, 18 };
        REQUIRE(collected == expected);
    }

    SECTION("Filter empty range") {
        auto R = range_view(5, 0);  // empty

        auto filtered = R.filter([](int x) { return true; });  // always true

        REQUIRE(filtered.begin() == filtered.end());
        std::vector<int> collected;
        for (int x : filtered) {
            collected.push_back(x);
        }
        REQUIRE(collected.empty());
    }
}

/**
 * @test FilterCartesian::MemberFilter
 * @brief Verifies Cartesian.filter() member function
 *
 * @scenario Filtering Cartesian product with predicates
 * @given A Cartesian product
 * @when Applying .filter() with multi-argument lambdas
 * @then Only matching tuples are iterated
 *
 * @covers Cartesian::filter()
 */
TEST_CASE("E3: FilterCartesian::MemberFilter", "[filter][Cartesian]") {
    SECTION("Filter 2D Cartesian product") {
        IndexList I{ 1, 2, 3 };
        IndexList J{ 1, 2, 3 };

        auto P = I * J;
        auto filtered = P.filter(
            [](int i, int j) { return i < j; }  // only i < j
        );

        std::vector<std::pair<int, int>> collected;
        for (auto [i, j] : filtered) {
            collected.emplace_back(i, j);
        }

        std::vector<std::pair<int, int>> expected{
            {1, 2}, {1, 3}, {2, 3}
        };
        REQUIRE(collected == expected);
    }

    SECTION("Filter with multiple predicates") {
        IndexList I{ 1, 2, 3, 4 };
        IndexList J{ 1, 2, 3, 4 };

        auto P = I * J;
        auto filtered = P.filter(
            [](int i, int j) { return i != j; },   // not diagonal
            [](int i, int j) { return i + j <= 5; } // sum <= 5
        );

        std::vector<std::pair<int, int>> collected;
        for (auto [i, j] : filtered) {
            collected.emplace_back(i, j);
        }

        std::vector<std::pair<int, int>> expected{
            {1, 2}, {1, 3}, {1, 4},
            {2, 1}, {2, 3},
            {3, 1}, {3, 2},
            {4, 1}
        };
        REQUIRE(collected.size() == expected.size());
        // Order matters in Cartesian iteration
        REQUIRE(collected == expected);
    }

    SECTION("Filter 3D Cartesian product") {
        IndexList A{ 1, 2 };
        IndexList B{ 1, 2 };
        IndexList C{ 1, 2 };

        auto P = A * B * C;
        auto filtered = P.filter(
            [](int a, int b, int c) { return a + b + c == 4; }
        );

        std::vector<std::tuple<int, int, int>> collected;
        for (auto [a, b, c] : filtered) {
            collected.emplace_back(a, b, c);
        }

        std::vector<std::tuple<int, int, int>> expected{
            {1, 1, 2}, {1, 2, 1}, {2, 1, 1}
        };
        REQUIRE(collected == expected);
    }

    SECTION("Filter with capture") {
        int max_sum = 5;
        IndexList I{ 1, 2, 3 };
        IndexList J{ 1, 2, 3 };

        auto P = I * J;
        auto filtered = P.filter(
            [max_sum](int i, int j) { return i + j <= max_sum; }
        );

        std::vector<std::pair<int, int>> collected;
        for (auto [i, j] : filtered) {
            collected.emplace_back(i, j);
        }

        REQUIRE(collected.size() == 8);  // All pairs except (3,3): 9 - 1 = 8
        for (auto [i, j] : collected) {
            REQUIRE(i + j <= max_sum);
        }
    }

    SECTION("Filter empty Cartesian product") {
        IndexList empty;
        IndexList I{ 1, 2, 3 };

        auto P = empty * I;
        auto filtered = P.filter([](int i, int j) { return true; });

        REQUIRE(filtered.begin() == filtered.end());
        std::vector<std::pair<int, int>> collected;
        for (auto [i, j] : filtered) {
            collected.emplace_back(i, j);
        }
        REQUIRE(collected.empty());
    }
}

/**
 * @test FilterPipe::PipeSyntax
 * @brief Verifies pipe syntax with dsl::filter
 *
 * @scenario Filtering using operator| with dsl::filter
 * @given Various index sources
 * @when Applying | dsl::filter()
 * @then Pipe syntax works identically to member .filter()
 *
 * @covers operator| with dsl::filter
 */
TEST_CASE("E4: FilterPipe::PipeSyntax", "[filter][pipe]") {
    SECTION("IndexList with pipe") {
        IndexList I{ 1, 2, 3, 4, 5, 6 };

        auto even = I | dsl::filter([](int x) { return x % 2 == 0; });

        std::vector<int> collected;
        for (int x : even) {
            collected.push_back(x);
        }

        std::vector<int> expected{ 2, 4, 6 };
        REQUIRE(collected == expected);
    }

    SECTION("Multiple predicates with pipe") {
        IndexList I{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        auto filtered = I | dsl::filter(
            [](int x) { return x % 2 == 0; },
            [](int x) { return x > 3; },
            [](int x) { return x < 9; }
        );

        std::vector<int> collected;
        for (int x : filtered) {
            collected.push_back(x);
        }

        std::vector<int> expected{ 4, 6, 8 };
        REQUIRE(collected == expected);
    }

    SECTION("RangeView with pipe") {
        auto R = range_view(0, 10);

        auto filtered = R | dsl::filter(
            [](int x) { return x % 3 == 0; }  // multiples of 3
        );

        std::vector<int> collected;
        for (int x : filtered) {
            collected.push_back(x);
        }

        std::vector<int> expected{ 0, 3, 6, 9 };
        REQUIRE(collected == expected);
    }

    SECTION("Cartesian product with pipe") {
        IndexList I{ 1, 2, 3 };
        IndexList J{ 1, 2, 3 };

        auto filtered = (I * J) | dsl::filter(
            [](int i, int j) { return i < j; }
        );

        std::vector<std::pair<int, int>> collected;
        for (auto [i, j] : filtered) {
            collected.emplace_back(i, j);
        }

        std::vector<std::pair<int, int>> expected{
            {1, 2}, {1, 3}, {2, 3}
        };
        REQUIRE(collected == expected);
    }

    SECTION("Chained filtering") {
        IndexList I{ 1, 2, 3, 4, 5 };

        // Note: This would require filter to return something filterable again
        // The current implementation doesn't support chaining filter() calls
        // directly, but you can chain predicates
        auto filtered = I | dsl::filter(
            [](int x) { return x % 2 == 0; },
            [](int x) { return x > 2; }
        );

        std::vector<int> collected;
        for (int x : filtered) {
            collected.push_back(x);
        }

        std::vector<int> expected{ 4 };
        REQUIRE(collected == expected);
    }

    SECTION("Pipe with capture") {
        int min_val = 3;
        int max_val = 7;

        auto R = range_view(0, 10);
        auto filtered = R | dsl::filter(
            [min_val, max_val](int x) { return x >= min_val && x <= max_val; }
        );

        std::vector<int> collected;
        for (int x : filtered) {
            collected.push_back(x);
        }

        std::vector<int> expected{ 3, 4, 5, 6, 7 };
        REQUIRE(collected == expected);
    }
}

/**
 * @test FilterEquivalence::MemberVsPipe
 * @brief Verifies equivalence of member filter and pipe syntax
 *
 * @scenario Same predicates applied via both syntaxes
 * @given An IndexList and predicates
 * @when Filtering with .filter() vs | dsl::filter()
 * @then Results are identical
 *
 * @covers Filter syntax equivalence
 */
TEST_CASE("E5: FilterEquivalence::MemberVsPipe", "[filter][equivalence]") {
    IndexList I{ 1, 2, 3, 4, 5, 6 };

    auto filtered1 = I.filter(
        [](int x) { return x % 2 == 0; },
        [](int x) { return x > 2; }
    );

    auto filtered2 = I | dsl::filter(
        [](int x) { return x % 2 == 0; },
        [](int x) { return x > 2; }
    );

    std::vector<int> v1(filtered1.begin(), filtered1.end());
    std::vector<int> v2(filtered2.begin(), filtered2.end());

    REQUIRE(v1 == v2);
    REQUIRE(v1 == std::vector<int>{4, 6});
}

// ============================================================================
// SECTION F: PRINTING UTILITIES
// ============================================================================

/**
 * @test PrintIndexList::StreamOutput
 * @brief Verifies IndexList stream output
 *
 * @scenario Printing IndexList to ostream
 * @given Various IndexList configurations
 * @when Streaming to ostringstream
 * @then Output matches expected format
 *
 * @covers operator<<(ostream, IndexList)
 */
TEST_CASE("F1: PrintIndexList::StreamOutput", "[printing][IndexList]") {
    SECTION("Empty IndexList") {
        IndexList empty;
        std::ostringstream oss;
        oss << empty;
        REQUIRE(oss.str() == "{}");
    }

    SECTION("Single element") {
        IndexList I{ 42 };
        std::ostringstream oss;
        oss << I;
        REQUIRE(oss.str() == "{42}");
    }

    SECTION("Multiple elements") {
        IndexList I{ 1, 3, 7 };
        std::ostringstream oss;
        oss << I;
        REQUIRE(oss.str() == "{1, 3, 7}");
    }

    SECTION("With duplicates") {
        IndexList I{ 1, 2, 2, 3 };
        std::ostringstream oss;
        oss << I;
        REQUIRE(oss.str() == "{1, 2, 2, 3}");
    }

    SECTION("Negative numbers") {
        IndexList I{ -5, 0, 5 };
        std::ostringstream oss;
        oss << I;
        REQUIRE(oss.str() == "{-5, 0, 5}");
    }
}

/**
 * @test PrintRangeView::StreamOutput
 * @brief Verifies RangeView stream output
 *
 * @scenario Printing RangeView to ostream
 * @given Various RangeView configurations
 * @when Streaming to ostringstream
 * @then Output matches expected format
 *
 * @covers operator<<(ostream, RangeView)
 */
TEST_CASE("F2: PrintRangeView::StreamOutput", "[printing][RangeView]") {
    SECTION("Empty RangeView") {
        auto R = range_view(5, 0);
        std::ostringstream oss;
        oss << R;
        REQUIRE(oss.str() == "range_view(empty)");
    }

    SECTION("Simple range") {
        auto R = range_view(0, 5);
        std::ostringstream oss;
        oss << R;
        // Format: "range_view(0, 5, step=1) -> [0, 1, 2, 3, 4]"
        REQUIRE(oss.str().find("range_view(0, 5, step=1) -> [0, 1, 2, 3, 4]") != std::string::npos);
    }

    SECTION("Range with step") {
        auto R = range_view(0, 10, 2);
        std::ostringstream oss;
        oss << R;
        // Format: "range_view(0, 10, step=2) -> [0, 2, 4, 6, 8]"
        REQUIRE(oss.str().find("range_view(0, 10, step=2) -> [0, 2, 4, 6, 8]") != std::string::npos);
    }

    SECTION("Single element range") {
        auto R = range_view(7, 8);
        std::ostringstream oss;
        oss << R;
        REQUIRE(oss.str().find("range_view(7, 8, step=1) -> [7]") != std::string::npos);
    }

    SECTION("Large range (truncated)") {
        auto R = range_view(0, 100);
        std::ostringstream oss;
        oss << R;
        // Should show first 10 elements then "..."
        REQUIRE(oss.str().find("...") != std::string::npos);
        REQUIRE(oss.str().find("0, 1, 2, 3, 4, 5, 6, 7, 8, 9") != std::string::npos);
    }
}

/**
 * @test PrintCartesian::StreamOutput
 * @brief Verifies Cartesian product stream output
 *
 * @scenario Printing Cartesian product to ostream
 * @given Various Cartesian configurations
 * @when Streaming to ostringstream
 * @then Output matches expected format
 *
 * @covers operator<<(ostream, Cartesian)
 */
TEST_CASE("F3: PrintCartesian::StreamOutput", "[printing][Cartesian]") {
SECTION("2D Cartesian product") {
    IndexList A{ 1, 2 };
    IndexList B{ 10, 20 };

    auto P = A * B;
    std::ostringstream oss;
    oss << P;

    // Format: "({1, 2} x {10, 20}) = { (1, 10), (1, 20), (2, 10), (2, 20) }"
    std::string s = oss.str();
    REQUIRE(s.find("{1, 2}") != std::string::npos);
    REQUIRE(s.find("{10, 20}") != std::string::npos);
    REQUIRE(s.find("(1, 10)") != std::string::npos);
    REQUIRE(s.find("(2, 20)") != std::string::npos);
}

    SECTION("Empty Cartesian product") {
        IndexList empty;
        auto P = empty * empty;
        std::ostringstream oss;
        oss << P;

        // Should show empty sets and empty result
        REQUIRE(oss.str().find("{}") != std::string::npos);
    }

    SECTION("Cartesian with RangeView") {
        auto R = range_view(0, 2);
        IndexList B{ 10, 20 };

        auto P = R * B;
        std::ostringstream oss;
        oss << P;

        std::string s = oss.str();
        REQUIRE(s.find("range_view") != std::string::npos);
        REQUIRE(s.find("{10, 20}") != std::string::npos);
        REQUIRE(s.find("(0, 10)") != std::string::npos);
    }

    SECTION("3D Cartesian product") {
        IndexList A{ 1, 2 };
        IndexList B{ 10 };
        IndexList C{ 100, 200 };

        auto P = A * B * C;
        std::ostringstream oss;
        oss << P;

        std::string s = oss.str();
        REQUIRE(s.find("{1, 2}") != std::string::npos);
        REQUIRE(s.find("{10}") != std::string::npos);
        REQUIRE(s.find("{100, 200}") != std::string::npos);
        REQUIRE(s.find("(1, 10, 100)") != std::string::npos);
        REQUIRE(s.find("(2, 10, 200)") != std::string::npos);
    }

    SECTION("Large Cartesian product (truncated)") {
        auto R1 = range_view(0, 5);
        auto R2 = range_view(0, 5);

        auto P = R1 * R2;
        std::ostringstream oss;
        oss << P;

        std::string s = oss.str();
        // Should show "..." due to truncation
        REQUIRE(s.find("...") != std::string::npos);
    }
}

// ============================================================================
// SECTION G: INTEGRATION AND COMPLEX SCENARIOS
// ============================================================================

/**
 * @test Integration::ComplexScenarios
 * @brief Verifies complex integration scenarios
 *
 * @scenario Nested filtering, set operations with Cartesian, scheduling
 * @given Complex combinations of index operations
 * @when Composing multiple operations
 * @then Results match expected complex behavior
 *
 * @covers Integration of multiple indexing features
 */
TEST_CASE("G1: Integration::ComplexScenarios", "[integration]") {
    SECTION("Nested filtering with Cartesian") {
        // Create set of vertices
        IndexList vertices{ 0, 1, 2, 3, 4 };

        // Create all possible edges (i,j) where i != j
        auto all_edges = vertices * vertices;
        auto edges = all_edges | dsl::filter(
            [](int i, int j) { return i != j; }
        );

        // Now filter to only edges where i < j (undirected, no duplicates)
        auto undirected_edges = edges | dsl::filter(
            [](int i, int j) { return i < j; }
        );

        std::vector<std::pair<int, int>> collected;
        for (auto [i, j] : undirected_edges) {
            collected.emplace_back(i, j);
        }

        // For 5 vertices, we expect C(5,2) = 10 edges
        REQUIRE(collected.size() == 10);

        // Verify all pairs have i < j
        for (auto [i, j] : collected) {
            REQUIRE(i < j);
        }

        // Verify all combinations are present
        for (int i = 0; i < 5; ++i) {
            for (int j = i + 1; j < 5; ++j) {
                auto it = std::find(collected.begin(), collected.end(),
                    std::make_pair(i, j));
                REQUIRE(it != collected.end());
            }
        }
    }

    SECTION("Set operations combined with Cartesian") {
        IndexList A{ 1, 2, 3 };
        IndexList B{ 3, 4, 5 };

        // Union of A and B
        auto U = A + B;  // {1, 2, 3, 4, 5}

        // Cartesian product of union with itself
        auto P = U * U;

        // Filter to diagonal elements (i == j)
        auto diagonal = P | dsl::filter(
            [](int i, int j) { return i == j; }
        );

        std::vector<std::pair<int, int>> collected;
        for (auto [i, j] : diagonal) {
            collected.emplace_back(i, j);
        }

        REQUIRE(collected.size() == 5);  // One for each element in U
        for (auto [i, j] : collected) {
            REQUIRE(i == j);
        }
    }

    SECTION("RangeView with set operations") {
        auto R1 = range_view(0, 5);    // {0, 1, 2, 3, 4}
        auto R2 = range_view(3, 8);    // {3, 4, 5, 6, 7}

        // Can't directly do set ops on RangeView, need to materialize
        IndexList I1(R1);
        IndexList I2(R2);

        auto intersection = I1 & I2;  // {3, 4}
        auto union_set = I1 + I2;     // {0, 1, 2, 3, 4, 5, 6, 7}

        REQUIRE(intersection.size() == 2);
        REQUIRE(union_set.size() == 8);

        // Verify intersection contents
        REQUIRE(intersection.contains(3));
        REQUIRE(intersection.contains(4));
        REQUIRE_FALSE(intersection.contains(2));
        REQUIRE_FALSE(intersection.contains(5));
    }

    SECTION("Complex filtering with multiple conditions") {
        // Simulate a small scheduling problem
        IndexList days{ 0, 1, 2, 3, 4 };  // Monday to Friday
        IndexList times{ 9, 10, 11, 14, 15, 16 };  // 9am-12pm, 2pm-5pm
        IndexList rooms{ 101, 102, 103 };

        // All possible assignments (day, time, room)
        auto all_assignments = days * times * rooms;

        // Apply constraints:
        // 1. No meetings before 10am on Monday (day 0)
        // 2. Room 103 is only available after 2pm (time >= 14)
        // 3. No meetings at 11am in any room
        auto valid_assignments = all_assignments | dsl::filter(
            [](int day, int time, int room) {
                // Constraint 1
                if (day == 0 && time < 10) return false;
                // Constraint 2
                if (room == 103 && time < 14) return false;
                // Constraint 3
                if (time == 11) return false;
                return true;
            }
        );

        // Count valid assignments
        std::size_t count = 0;
        for (auto [d, t, r] : valid_assignments) {
            (void)d; (void)t; (void)r;  // Suppress unused warnings
            ++count;

            // Verify constraints are satisfied
            REQUIRE_FALSE((d == 0 && t < 10));
            REQUIRE_FALSE((r == 103 && t < 14));
            REQUIRE_FALSE((t == 11));
        }

        // Calculate expected count:
        // Total assignments: 5 days * 6 times * 3 rooms = 90
        // Let's count valid ones directly:
        // Valid times (not 11): 9, 10, 14, 15, 16 = 5 times
        // For rooms 101 and 102: 5 days * 5 times * 2 rooms = 50
        // For room 103 (only times >= 14): 5 days * 3 times (14,15,16) * 1 room = 15
        // But Monday 9am is invalid for all rooms: subtract 3 from above
        // Wait, let's recalculate:
        // Base: 5 days * 5 valid times (excluding 11) * 3 rooms = 75
        // But room 103 at times 9, 10 is invalid: 5 days * 2 times * 1 room = 10
        // And Monday 9am is invalid: 1 day * 1 time * 3 rooms = 3
        // Monday 9am room 103 is counted twice, add back 1
        // So: 75 - 10 - 3 + 1 = 63
        REQUIRE(count == 63);
    }

    SECTION("Filtered Cartesian with variable capture") {
        // Simulate a knapsack-like problem
        std::vector<int> weights{ 2, 3, 5, 7 };
        std::vector<int> values{ 10, 20, 30, 40 };
        int capacity = 10;

        // Create index set for items
        IndexList items(weights.size());
        std::iota(items.begin(), items.end(), 0);  // {0, 1, 2, 3}

        // All possible pairs of items
        auto all_pairs = items * items;

        // Filter pairs where:
        // 1. i != j (different items)
        // 2. weight_i + weight_j <= capacity
        auto valid_pairs = all_pairs | dsl::filter(
            [](int i, int j) { return i != j; },
            [&weights, capacity](int i, int j) {
                return weights[i] + weights[j] <= capacity;
            }
        );

        // Find pair with maximum total value
        int max_value = 0;
        std::pair<int, int> best_pair{ -1, -1 };

        for (auto [i, j] : valid_pairs) {
            int total_value = values[i] + values[j];
            if (total_value > max_value) {
                max_value = total_value;
                best_pair = { i, j };
            }
        }

        // With weights {2,3,5,7} and capacity 10:
        // Valid pairs: (0,1): weight 5, value 30
        //              (0,2): weight 7, value 40
        //              (1,2): weight 8, value 50  <-- best
        //              (0,3): weight 9, value 50  <-- also weight 9
        //              (1,3): weight 10, value 60 <-- best, weight exactly 10
        REQUIRE(max_value == 60);
        REQUIRE(weights[best_pair.first] + weights[best_pair.second] <= capacity);
        REQUIRE(best_pair.first != best_pair.second);
    }
}

// ============================================================================
// SECTION H: EDGE CASES AND ERROR CONDITIONS
// ============================================================================

/**
 * @test EdgeCases::BoundaryConditions
 * @brief Verifies edge cases and boundary conditions
 *
 * @scenario Large ranges, negative values, empty sets, self-operations
 * @given Various boundary condition inputs
 * @when Applying indexing operations
 * @then Edge cases are handled correctly
 *
 * @covers Edge case handling
 */
TEST_CASE("H1: EdgeCases::BoundaryConditions", "[edge]") {
    SECTION("Very large ranges (avoid overflow)") {
        // Test that size computation doesn't overflow
        auto R = range_view(0, 1000000, 7);

        // Should compute correct size without overflow
        // ceil(1000000 / 7) = (1000000 + 7 - 1) / 7 = 142858
        REQUIRE(R.size() == 142858);

        // First element
        REQUIRE(R[0] == 0);

        // Last element: 0 + 142857 * 7 = 999999
        REQUIRE(R[R.size() - 1] == 999999);

        // Step between elements
        if (R.size() >= 2) {
            REQUIRE(R[1] - R[0] == 7);
        }
    }

    SECTION("Negative start values") {
        auto R = range_view(-5, 3);
        REQUIRE(R.size() == 8);  // -5, -4, -3, -2, -1, 0, 1, 2

        std::vector<int> collected;
        for (int x : R) {
            collected.push_back(x);
        }

        std::vector<int> expected{ -5, -4, -3, -2, -1, 0, 1, 2 };
        REQUIRE(collected == expected);
    }

    SECTION("Cartesian product with single huge dimension") {
        // One small set, one large set
        IndexList small{ 1, 2 };
        auto large = range_view(0, 1000);

        auto P = small * large;
        REQUIRE(P.size() == 2000);

        // Verify first few elements
        auto it = P.begin();
        auto [s1, l1] = *it;
        REQUIRE(s1 == 1);
        REQUIRE(l1 == 0);

        ++it;
        auto [s2, l2] = *it;
        REQUIRE(s2 == 1);
        REQUIRE(l2 == 1);

        // Skip many elements
        for (int i = 0; i < 500; ++i) ++it;

        // Should still be valid
        REQUIRE(it != P.end());
    }

    SECTION("Filter that rejects all elements") {
        IndexList I{ 1, 2, 3, 4, 5 };

        auto filtered = I.filter([](int x) { return false; });

        REQUIRE(filtered.begin() == filtered.end());

        // Multiple iterations should still work
        int count = 0;
        for (int x : filtered) { (void)x; ++count; }
        REQUIRE(count == 0);

        for (int x : filtered) { (void)x; ++count; }
        REQUIRE(count == 0);
    }

    SECTION("Empty sets in complex expressions") {
        IndexList empty;
        IndexList A{ 1, 2 };
        IndexList B{ 10, 20 };

        // Operations with empty sets
        REQUIRE((empty + A).size() == 2);
        REQUIRE((A & empty).empty());
        REQUIRE((empty * B).empty());
        REQUIRE((A * empty).empty());

        // Chained operations
        auto P = A * empty * B;
        REQUIRE(P.empty());

        // Filtering empty product
        auto filtered = P.filter([](int a, int b, int c) { return true; });
        REQUIRE(filtered.begin() == filtered.end());
    }

    SECTION("Self-assignment and aliasing") {
        IndexList I{ 1, 2, 3 };

        // Self-union
        auto U = I + I;
        REQUIRE(U.size() == 3);  // No new elements added
        REQUIRE(std::equal(I.begin(), I.end(), U.begin()));

        // Self-intersection
        auto Inter = I & I;
        REQUIRE(Inter.size() == 3);
        REQUIRE(std::equal(I.begin(), I.end(), Inter.begin()));

        // Self-difference
        auto Diff = I - I;
        REQUIRE(Diff.empty());

        // Self-Cartesian product
        auto P = I * I;
        REQUIRE(P.size() == 9);  // 3x3
    }

    SECTION("Move semantics don't break iteration") {
        std::vector<int> v{ 1, 2, 3, 4, 5 };
        IndexList I(std::move(v));

        // v is now in valid but unspecified state
        // I should still work
        REQUIRE(I.size() == 5);
        REQUIRE(I[0] == 1);
        REQUIRE(I[4] == 5);

        // Iteration should work
        std::vector<int> collected;
        for (int x : I) {
            collected.push_back(x);
        }
        REQUIRE(collected.size() == 5);
    }
}

// ============================================================================
// SECTION I: ITERATOR PROPERTIES AND PERFORMANCE
// ============================================================================

/**
 * @test IteratorProperties::ForwardIteratorRequirements
 * @brief Verifies iterator properties and requirements
 *
 * @scenario Copy construction, equality, increment for various iterators
 * @given Iterators from IndexList, RangeView, Cartesian, Filtered
 * @when Testing forward iterator requirements
 * @then All iterators satisfy forward iterator concept
 *
 * @covers Iterator requirements
 */
TEST_CASE("I1: IteratorProperties::ForwardIteratorRequirements", "[iterator]") {
    SECTION("Forward iterator requirements for IndexList") {
        IndexList I{ 1, 2, 3 };

        auto it1 = I.begin();
        auto it2 = it1;  // Copy constructible

        REQUIRE(it1 == it2);
        REQUIRE(*it1 == *it2);

        ++it1;
        REQUIRE(it1 != it2);
        REQUIRE(*it1 == 2);

        auto it3 = it1;  // Copy after increment
        REQUIRE(*it3 == 2);

        ++it3;
        REQUIRE(*it3 == 3);
        REQUIRE(*it1 == 2);  // it1 unchanged by it3 increment
    }

    SECTION("Forward iterator requirements for RangeView") {
        auto R = range_view(0, 3);

        auto it1 = R.begin();
        auto it2 = it1;

        REQUIRE(it1 == it2);
        REQUIRE(*it1 == *it2);

        ++it1;
        REQUIRE(it1 != it2);
        REQUIRE(*it1 == 1);

        // Multiple passes over same range
        std::vector<int> pass1, pass2;
        for (int x : R) pass1.push_back(x);
        for (int x : R) pass2.push_back(x);
        REQUIRE(pass1 == pass2);
    }

    SECTION("Forward iterator requirements for Cartesian") {
        IndexList A{ 1, 2 };
        IndexList B{ 10, 20 };

        auto P = A * B;
        auto it1 = P.begin();
        auto it2 = it1;

        REQUIRE(it1 == it2);
        REQUIRE(*it1 == *it2);

        ++it1;
        REQUIRE(it1 != it2);

        // Can iterate multiple times
        std::vector<std::pair<int, int>> pass1, pass2;
        for (auto [a, b] : P) pass1.emplace_back(a, b);
        for (auto [a, b] : P) pass2.emplace_back(a, b);
        REQUIRE(pass1 == pass2);
    }

    SECTION("Filtered iterator properties") {
        IndexList I{ 1, 2, 3, 4, 5 };
        auto filtered = I.filter([](int x) { return x % 2 == 0; });

        auto it1 = filtered.begin();
        auto it2 = it1;

        REQUIRE(it1 == it2);

        // Should point to first even number (2)
        REQUIRE(*it1 == 2);

        ++it1;
        REQUIRE(*it1 == 4);
        REQUIRE(*it2 == 2);  // it2 unchanged

        // End iterator equality
        auto end1 = filtered.end();
        auto end2 = filtered.end();
        REQUIRE(end1 == end2);
    }

    SECTION("Iterator const correctness") {
        const IndexList I{ 1, 2, 3 };

        // Should be able to get const iterators
        auto it = I.begin();
        auto cit = I.begin();

        // Can read through iterator
        int x = *it;
        REQUIRE(x == 1);

        // Can't modify through iterator (const correctness)
        // *it = 10;  // Should not compile

        // Range-based for works with const
        std::vector<int> collected;
        for (int val : I) {
            collected.push_back(val);
        }
        REQUIRE(collected.size() == 3);
    }
}

/**
 * @test PerformanceSanity::LazyEvaluation
 * @brief Verifies performance characteristics
 *
 * @scenario Large ranges, products without materialization
 * @given Large conceptual data sets
 * @when Creating views without full iteration
 * @then Memory and time remain bounded
 *
 * @covers Lazy evaluation performance
 */
TEST_CASE("I2: PerformanceSanity::LazyEvaluation", "[performance]") {
    SECTION("RangeView doesn't allocate for large ranges") {
        // This is more of a compile-time check, but we can verify
        // that creating large ranges doesn't blow up
        auto R = range_view(0, 1000000);

        // Should have correct size
        REQUIRE(R.size() == 1000000);

        // First and last elements accessible
        REQUIRE(R[0] == 0);
        REQUIRE(R[999999] == 999999);

        // Memory usage should be constant (3 ints)
        // Can't test directly, but if this doesn't crash, we're good
    }

    SECTION("Cartesian product doesn't materialize") {
        // Create potentially huge Cartesian product
        auto R1 = range_view(0, 100);
        auto R2 = range_view(0, 100);

        auto P = R1 * R2;  // 10,000 elements conceptually

        // Should not allocate 10,000 elements
        // Just verify we can get the size and a few elements
        REQUIRE(P.size() == 10000);

        // First element
        auto it = P.begin();
        auto [x, y] = *it;
        REQUIRE(x == 0);
        REQUIRE(y == 0);

        // Last element
        // Don't actually iterate to end - that would be 10,000 increments
        // Just verify we can compute what it should be
        REQUIRE(R1[99] == 99);
        REQUIRE(R2[99] == 99);
    }

    SECTION("Filter doesn't copy underlying data") {
        // Create a large-ish index list
        std::vector<int> big_vec(1000);
        std::iota(big_vec.begin(), big_vec.end(), 0);
        IndexList I(big_vec);

        // Apply filter - should not copy the 1000 elements
        auto filtered = I.filter([](int x) { return x % 17 == 0; });

        // Count filtered elements
        std::size_t count = 0;
        for (int x : filtered) {
            (void)x;
            ++count;
        }

        // ceil(1000 / 17) = 59
        REQUIRE(count == 59);

        // Memory usage should be roughly constant + predicate
        // (not proportional to 1000)
    }
}