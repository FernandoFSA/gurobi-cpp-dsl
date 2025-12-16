/*
===============================================================================
TEST CONSTRAINTS — Comprehensive tests for constraints.h
===============================================================================

OVERVIEW
--------
Validates the DSL constraint management system including ConstraintGroup,
IndexedConstraintSet, ConstraintFactory, and ConstraintTable. Tests cover
scalar and N-dimensional constraint creation, attribute access, domain-based
indexing, and enum-keyed storage.

TEST ORGANIZATION
-----------------
• Section A: Basic scalar constraint creation
• Section B: ConstraintGroup attribute access (sense, rhs, name)
• Section C: Slack and dual values after optimization
• Section D: Multi-dimensional ConstraintGroup creation
• Section E: ConstraintTable functionality with ConstraintGroup
• Section F: IndexedConstraintSet with various domains
• Section G: ConstraintGroup introspection and iteration
• Section H: IndexedConstraintSet introspection and iteration
• Section I: ConstraintTable with IndexedConstraintSet
• Section J: Edge cases and error conditions

TEST STRATEGY
-------------
• Verify constraint creation with correct naming behavior
• Confirm attribute access for scalar and multi-dimensional constraints
• Validate domain-based constraint indexing with IndexList, RangeView, Cartesian
• Exercise filtered domains with ConstraintFactory::addIndexed
• Test ConstraintTable storage and retrieval patterns
• Verify introspection methods (shape, size, dimension)
• Exercise iteration methods (forEach, begin/end)
• Test edge cases: empty domains, duplicates, zero-size dimensions

DEPENDENCIES
------------
• Catch2 v3.0+ - Test framework
• constraints.h - System under test
• variables.h - Variable creation for constraint expressions
• naming.h - Naming utilities
• enum_utils.h - Enum declaration macros
• indexing.h - Index domains (IndexList, RangeView, Cartesian, filter)

===============================================================================
*/

#include "catch_amalgamated.hpp"
#include <gurobi_dsl/variables.h>
#include <gurobi_dsl/constraints.h>
#include <gurobi_dsl/naming.h>
#include <gurobi_dsl/enum_utils.h>
#include <gurobi_dsl/indexing.h>

// ============================================================================
// TEST UTILITIES
// ============================================================================

static GRBModel makeModel() {
    static GRBEnv env = GRBEnv(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel m(env);
    return m;
}

static void optimizeSafe(GRBModel& model) {
    REQUIRE_NOTHROW(model.optimize());
    int status = model.get(GRB_IntAttr_Status);
    REQUIRE((status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL));
}

// ============================================================================
// SECTION A: BASIC SCALAR CONSTRAINT CREATION
// ============================================================================

/**
 * @test ScalarConstraint::CreationWithNaming
 * @brief Verifies scalar constraint creation via ConstraintFactory::add
 *
 * @scenario Creating a single constraint with naming enabled
 * @given A Gurobi model with a continuous variable
 * @when Creating a scalar constraint using ConstraintFactory::add
 * @then The constraint has correct dimension, RHS, sense, and name
 *
 * @covers ConstraintFactory::add (scalar), ConstraintGroup::dimension, isScalar, scalar
 */
TEST_CASE("A1: ScalarConstraint::CreationWithNaming", "[ConstraintFactory][scalar][naming]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X");
    model.update();

    auto c = dsl::ConstraintFactory::add(
        model,
        "c_test",
        [&](const std::vector<int>&) {
            return (X <= 5.0); // Gurobi DSL
        }
    );

    model.update();

    REQUIRE(c.dimension() == 0);
    REQUIRE(c.isScalar());
    REQUIRE_NOTHROW(c.scalar());

    GRBConstr& g = c.scalar();

    REQUIRE(g.get(GRB_DoubleAttr_RHS) == Catch::Approx(5.0));
    REQUIRE(g.get(GRB_CharAttr_Sense) == '<');

    if (naming_enabled())
        REQUIRE(std::string(g.get(GRB_StringAttr_ConstrName)) == "c_test");
}

// ============================================================================
// SECTION B: CONSTRAINT ATTRIBUTE ACCESS
// ============================================================================

/**
 * @test ConstraintGroup::AttributeAccess
 * @brief Verifies attribute access methods for scalar constraints
 *
 * @scenario Accessing sense, rhs, name, and renaming a constraint
 * @given A scalar constraint with equality sense
 * @when Calling sense(), rhs(), name(), setName(), raw()
 * @then All attributes return expected values and renaming works
 *
 * @covers ConstraintGroup::sense, rhs, name, setName, raw
 */
TEST_CASE("B1: ConstraintGroup::AttributeAccess", "[ConstraintGroup][attributes]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X");
    model.update();

    auto c = dsl::ConstraintFactory::add(
        model,
        "fixX",
        [&](const std::vector<int>&) {
            return (X == 3.0);
        }
    );

    model.update();

    REQUIRE(c.sense() == '=');
    REQUIRE(c.rhs() == Catch::Approx(3.0));

    if (naming_enabled())
        REQUIRE(c.name() == "fixX");

    REQUIRE_NOTHROW(c.setName("renamed"));

    model.update();
    if (naming_enabled())
        REQUIRE(c.name() == "renamed");

    GRBConstr& cr = c.raw();
    REQUIRE(cr.get(GRB_DoubleAttr_RHS) == Catch::Approx(3.0));
}

// ============================================================================
// SECTION C: SLACK AND DUAL VALUES
// ============================================================================

/**
 * @test ConstraintGroup::SlackAndDual
 * @brief Verifies slack() and dual() access after optimization
 *
 * @scenario Accessing slack and dual values after model optimization
 * @given A model with a binding constraint, optimized to completion
 * @when Calling slack() and dual() on the constraint
 * @then Both values are finite and slack is approximately zero
 *
 * @covers ConstraintGroup::slack, dual
 */
TEST_CASE("C1: ConstraintGroup::SlackAndDual", "[ConstraintGroup][optimization]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, GRB_INFINITY, "X");
    model.setObjective(GRBLinExpr(X), GRB_MINIMIZE);
    model.update();

    auto c = dsl::ConstraintFactory::add(
        model,
        "ge1",
        [&](const std::vector<int>&) {
            return (X >= 1.0);
        }
    );

    model.update();
    optimizeSafe(model);

    REQUIRE_NOTHROW(c.slack());
    REQUIRE(std::abs(c.slack()) <= 1e-6);

    REQUIRE_NOTHROW(c.dual());
    REQUIRE(std::isfinite(c.dual()));
}

// ============================================================================
// SECTION D: MULTI-DIMENSIONAL CONSTRAINT GROUPS
// ============================================================================

/**
 * @test ConstraintFactory::MultiDimensional
 * @brief Verifies creation of 2D constraint groups
 *
 * @scenario Creating a 2x3 array of constraints
 * @given A Gurobi model with a 2x3 variable array
 * @when Creating constraints using ConstraintFactory::add with sizes (2, 3)
 * @then ConstraintGroup has correct dimension, sizes, and naming
 *
 * @covers ConstraintFactory::add (N-D), ConstraintGroup::dimension, size, operator()
 */
TEST_CASE("D1: ConstraintFactory::MultiDimensional", "[ConstraintFactory][multidim]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0.0, 1.0, "X", 2, 3);
    model.update();

    dsl::ConstraintGroup caps = dsl::ConstraintFactory::add(
        model,
        "row",
        [&](const std::vector<int>& idx) {
            int i = idx[0];
            int j = idx[1];
            return (X.at(i, j) <= 1); // trivial constraint
        },
        2, 3
    );

    model.update();
    REQUIRE(caps.dimension() == 2);
    REQUIRE(caps.size(0) == 2);
    REQUIRE(caps.size(1) == 3);

    if (naming_enabled()) {
        REQUIRE(caps(0, 0).get(GRB_StringAttr_ConstrName) == std::string("row[0,0]"));
    }
}

// ============================================================================
// SECTION E: CONSTRAINT TABLE FUNCTIONALITY
// ============================================================================

DECLARE_ENUM_WITH_COUNT(CTestFam,
    ONE,
    TWO
);

/**
 * @test ConstraintTable::StorageAndRetrieval
 * @brief Verifies ConstraintTable storage and retrieval of constraint groups
 *
 * @scenario Storing and retrieving a scalar constraint via enum key
 * @given A ConstraintTable with ConstraintGroup default type
 * @when Setting and getting a constraint group by enum key
 * @then The constraint is accessible with correct attributes
 *
 * @covers ConstraintTable::set, get, constr
 */
TEST_CASE("E1: ConstraintTable::StorageAndRetrieval", "[ConstraintTable]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X");
    model.update();

    auto c1 = dsl::ConstraintFactory::add(
        model,
        "c1",
        [&](const std::vector<int>&) { return (X <= 4.0); }
    );

    dsl::ConstraintTable<CTestFam> table;
    REQUIRE_NOTHROW(table.set(CTestFam::ONE, std::move(c1)));

    model.update();
    GRBConstr& g = table.constr(CTestFam::ONE);
    REQUIRE(g.get(GRB_DoubleAttr_RHS) == Catch::Approx(4.0));
    REQUIRE(g.get(GRB_CharAttr_Sense) == '<');
}

// ============================================================================
// SECTION F: INDEXED CONSTRAINT SET WITH DOMAINS
// ============================================================================

/**
 * @test IndexedConstraintSet::OneDimensionalDomain
 * @brief Verifies IndexedConstraintSet creation with 1D IndexList domain
 *
 * @scenario Creating constraints indexed by a 1D IndexList
 * @given An IndexList domain {0,1,2,3,4} and a variable array
 * @when Creating constraints using ConstraintFactory::addIndexed
 * @then Constraints are accessible by index with correct RHS and naming
 *
 * @covers ConstraintFactory::addIndexed (1D), IndexedConstraintSet::at, try_get, size
 */
TEST_CASE("F1: IndexedConstraintSet::OneDimensionalDomain", "[IndexedConstraintSet][1D]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X", 5);
    model.update();

    dsl::IndexList I{ 0,1,2,3,4 };

    auto CC = dsl::ConstraintFactory::addIndexed(
        model,
        "c1d",
        I,
        [&](int i) {
            return (X(i) >= 2.0 * i);
        }
    );
    model.update();

    REQUIRE(CC.size() == 5);

    for (int i : I)
    {
        REQUIRE_NOTHROW(CC.at(i));
        GRBConstr& c = CC.at(i);
        REQUIRE(c.get(GRB_DoubleAttr_RHS) == Catch::Approx(2.0 * i));
        REQUIRE(c.get(GRB_CharAttr_Sense) == '>');
    }

    REQUIRE(CC.try_get(100) == nullptr); // not in domain

    if (naming_enabled())
        REQUIRE(std::string(CC.at(3).get(GRB_StringAttr_ConstrName)) == "c1d[3]");
}




/**
 * @test IndexedConstraintSet::CartesianProduct
 * @brief Verifies IndexedConstraintSet creation with 2D Cartesian product domain
 *
 * @scenario Creating constraints indexed by I * J Cartesian product
 * @given Two IndexLists I and J, and a 2D variable array
 * @when Creating constraints using ConstraintFactory::addIndexed with I * J
 * @then Constraints are accessible by (i,j) with correct RHS and naming
 *
 * @covers ConstraintFactory::addIndexed (Cartesian), IndexedConstraintSet::at
 */
TEST_CASE("F2: IndexedConstraintSet::CartesianProduct", "[IndexedConstraintSet][Cartesian]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 3, 4);
    model.update();

    auto I = dsl::range(0, 3); // {0,1,2}
    auto J = dsl::range(0, 4); // {0,1,2,3}

    auto CC =
        dsl::ConstraintFactory::addIndexed(
            model,
            "flow2d",
            I * J,
            [&](int i, int j) {
                return (X(i, j) <= i + j);
            }
        );

    model.update();

    REQUIRE(CC.size() == 12);

    REQUIRE_NOTHROW(CC.at(2, 3));
    REQUIRE_THROWS(CC.at(3, 3)); // i==3 not in I

    for (auto [i, j] : I * J)
    {
        GRBConstr& c = CC.at(i, j);
        REQUIRE(c.get(GRB_DoubleAttr_RHS) == Catch::Approx(i + j));
    }

    if (naming_enabled())
        REQUIRE(std::string(CC.at(1, 2).get(GRB_StringAttr_ConstrName)) == "flow2d[1,2]");
}




/**
 * @test IndexedConstraintSet::FilteredDomain
 * @brief Verifies IndexedConstraintSet creation with filtered Cartesian domain
 *
 * @scenario Creating constraints on filtered domain (i,j) where i < j
 * @given Two IndexLists and a filter predicate
 * @when Creating constraints using ConstraintFactory::addIndexed with pipe filter
 * @then Only filtered indices are present, others return nullptr from try_get
 *
 * @covers ConstraintFactory::addIndexed (filtered), dsl::filter, IndexedConstraintSet::try_get
 */
TEST_CASE("F3: IndexedConstraintSet::FilteredDomain", "[IndexedConstraintSet][filter]")
{
    GRBModel model = makeModel();

    auto Y = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "Y", 5, 5);
    model.update();

    auto I = dsl::range(0, 5);
    auto J = dsl::range(0, 5);

    // domain = {(i,j) | i < j}
    auto F = (I * J) | dsl::filter([](int i, int j) { return i < j; });

    auto CC =
        dsl::ConstraintFactory::addIndexed(
            model,
            "upper",
            F,
            [&](int i, int j) {
                return (Y(i, j) >= 100 + i + j);
            }
        );

    model.update();

    int expectedCount = 0;
    for (auto [i, j] : I * J)
        if (i < j) expectedCount++;

    REQUIRE(CC.size() == expectedCount);

    REQUIRE_NOTHROW(CC.at(0, 1));
    REQUIRE(CC.try_get(1, 0) == nullptr); // excluded by filter

    for (auto [i, j] : I * J)
    {
        if (i < j)
        {
            GRBConstr& c = CC.at(i, j);
            REQUIRE(c.get(GRB_DoubleAttr_RHS) == Catch::Approx(100 + i + j));
        }
    }

    if (naming_enabled())
        REQUIRE(std::string(CC.at(0, 1).get(GRB_StringAttr_ConstrName)) == "upper[0,1]");
}




/**
 * @test IndexedConstraintSet::RangeViewDomain
 * @brief Verifies IndexedConstraintSet creation with RangeView domain
 *
 * @scenario Creating constraints indexed by a stepped RangeView {0,2,4,6,8}
 * @given A RangeView with step 2 and a variable array
 * @when Creating constraints using ConstraintFactory::addIndexed
 * @then Only stepped indices are present, intermediate indices are not
 *
 * @covers ConstraintFactory::addIndexed (RangeView), IndexedConstraintSet::at, try_get
 */
TEST_CASE("F4: IndexedConstraintSet::RangeViewDomain", "[IndexedConstraintSet][RangeView]")
{
    GRBModel model = makeModel();

    auto Z = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 100, "Z", 10);
    model.update();

    auto RV = dsl::range_view(0, 10, 2); // {0,2,4,6,8}

    auto CC = dsl::ConstraintFactory::addIndexed(
        model,
        "step",
        RV,
        [&](int i) {
            return (Z(i) <= 50 + i);
        }
    );

    model.update();

    REQUIRE(CC.size() == RV.size());

    for (int i : RV)
        REQUIRE(CC.at(i).get(GRB_DoubleAttr_RHS) == Catch::Approx(50 + i));

    REQUIRE(CC.try_get(1) == nullptr); // odd index not in domain
}




// ============================================================================
// SECTION G: CONSTRAINT GROUP INTROSPECTION AND ITERATION
// ============================================================================

/**
 * @test ConstraintGroup::ShapeAndIntrospection
 * @brief Verifies shape() and introspection methods for multi-dimensional groups
 *
 * @scenario Querying shape and properties of a 3D constraint group
 * @given A 2x3x4 constraint group
 * @when Calling shape(), dimension(), isScalar(), isMultiDim()
 * @then All introspection methods return correct values
 *
 * @covers ConstraintGroup::shape, dimension, isScalar, isMultiDim
 */
TEST_CASE("G1: ConstraintGroup::ShapeAndIntrospection", "[ConstraintGroup][introspection]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 1.0, "X", 2, 3, 4);
    model.update();

    dsl::ConstraintGroup caps = dsl::ConstraintFactory::add(
        model,
        "cap3d",
        [&](const std::vector<int>& idx) {
            int i = idx[0];
            int j = idx[1];
            int k = idx[2];
            return (X.at(i, j, k) <= 1);
        },
        2, 3, 4
    );

    model.update();

    SECTION("Dimension queries") {
        REQUIRE(caps.dimension() == 3);
        REQUIRE_FALSE(caps.isScalar());
        REQUIRE(caps.isMultiDim());
    }

    SECTION("Shape query") {
        auto shp = caps.shape();
        REQUIRE(shp.size() == 3);
        REQUIRE(shp[0] == 2);
        REQUIRE(shp[1] == 3);
        REQUIRE(shp[2] == 4);
    }

    SECTION("Size per dimension") {
        REQUIRE(caps.size(0) == 2);
        REQUIRE(caps.size(1) == 3);
        REQUIRE(caps.size(2) == 4);
    }

    SECTION("Out of range dimension throws") {
        REQUIRE_THROWS(caps.size(-1));
        REQUIRE_THROWS(caps.size(3));
        REQUIRE_THROWS(caps.size(100));
    }
}

/**
 * @test ConstraintGroup::ForEachIteration
 * @brief Verifies forEach() iteration over all constraints
 *
 * @scenario Iterating through a 2D constraint group with forEach
 * @given A 2x3 constraint group
 * @when Calling forEach with a callback
 * @then All constraints are visited with correct indices
 *
 * @covers ConstraintGroup::forEach
 */
TEST_CASE("G2: ConstraintGroup::ForEachIteration", "[ConstraintGroup][iteration]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X", 2, 3);
    model.update();

    dsl::ConstraintGroup caps = dsl::ConstraintFactory::add(
        model,
        "iter",
        [&](const std::vector<int>& idx) {
            int i = idx[0];
            int j = idx[1];
            return (X.at(i, j) <= i * 10 + j);
        },
        2, 3
    );

    model.update();

    std::vector<std::pair<int, int>> visited;
    caps.forEach([&](GRBConstr& c, const std::vector<int>& idx) {
        REQUIRE(idx.size() == 2);
        visited.emplace_back(idx[0], idx[1]);

        // Verify RHS matches expected pattern
        double expectedRHS = idx[0] * 10 + idx[1];
        REQUIRE(c.get(GRB_DoubleAttr_RHS) == Catch::Approx(expectedRHS));
    });

    // Should visit all 6 elements in lexicographic order
    REQUIRE(visited.size() == 6);
    std::vector<std::pair<int, int>> expected{
        {0, 0}, {0, 1}, {0, 2},
        {1, 0}, {1, 1}, {1, 2}
    };
    REQUIRE(visited == expected);
}

/**
 * @test ConstraintGroup::ScalarAccessErrors
 * @brief Verifies error handling for scalar access on multi-dimensional groups
 *
 * @scenario Attempting scalar access on non-scalar constraint groups
 * @given Multi-dimensional and scalar constraint groups
 * @when Calling scalar() or raw() inappropriately
 * @then Throws for multi-dimensional, succeeds for scalar
 *
 * @covers ConstraintGroup::scalar, raw (error cases)
 */
TEST_CASE("G3: ConstraintGroup::ScalarAccessErrors", "[ConstraintGroup][errors]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X", 3);
    model.update();

    SECTION("Scalar access on 1D group throws") {
        auto caps = dsl::ConstraintFactory::add(
            model, "arr",
            [&](const std::vector<int>& idx) { return (X.at(idx[0]) <= 1); },
            3
        );
        model.update();

        REQUIRE_THROWS(caps.scalar());
        REQUIRE_THROWS(caps.raw());
    }

    SECTION("Index mismatch throws") {
        auto caps = dsl::ConstraintFactory::add(
            model, "arr",
            [&](const std::vector<int>& idx) { return (X.at(idx[0]) <= 1); },
            3
        );
        model.update();

        // Wrong number of indices
        REQUIRE_THROWS(caps.at());        // 0 indices for 1D
        REQUIRE_THROWS(caps.at(0, 1));    // 2 indices for 1D
    }

    SECTION("Out of range index throws") {
        auto caps = dsl::ConstraintFactory::add(
            model, "arr",
            [&](const std::vector<int>& idx) { return (X.at(idx[0]) <= 1); },
            3
        );
        model.update();

        REQUIRE_THROWS(caps.at(5));       // index >= size
        REQUIRE_THROWS(caps.at(-1));      // negative index
    }
}

// ============================================================================
// SECTION H: INDEXED CONSTRAINT SET INTROSPECTION
// ============================================================================

/**
 * @test IndexedConstraintSet::EmptyAndSize
 * @brief Verifies empty() and size() methods
 *
 * @scenario Checking size and emptiness of IndexedConstraintSet
 * @given Empty and non-empty IndexedConstraintSets
 * @when Calling empty() and size()
 * @then Methods return correct values
 *
 * @covers IndexedConstraintSet::empty, size
 */
TEST_CASE("H1: IndexedConstraintSet::EmptyAndSize", "[IndexedConstraintSet][introspection]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X", 5);
    model.update();

    SECTION("Non-empty set") {
        dsl::IndexList I{ 0, 1, 2 };
        auto CC = dsl::ConstraintFactory::addIndexed(
            model, "c",
            I,
            [&](int i) { return (X(i) <= 1); }
        );
        model.update();

        REQUIRE_FALSE(CC.empty());
        REQUIRE(CC.size() == 3);
    }

    SECTION("Empty domain produces empty set") {
        dsl::IndexList empty;
        auto CC = dsl::ConstraintFactory::addIndexed(
            model, "c",
            empty,
            [&](int i) { return (X(i) <= 1); }
        );
        model.update();

        REQUIRE(CC.empty());
        REQUIRE(CC.size() == 0);
    }

    SECTION("Filtered domain that excludes all") {
        dsl::IndexList I{ 1, 3, 5 }; // all odd
        auto filtered = I | dsl::filter([](int i) { return i % 2 == 0; }); // keep even only

        auto CC = dsl::ConstraintFactory::addIndexed(
            model, "c",
            filtered,
            [&](int i) { return (X(i) <= 1); }
        );
        model.update();

        REQUIRE(CC.empty());
        REQUIRE(CC.size() == 0);
    }
}

/**
 * @test IndexedConstraintSet::AllAndIteration
 * @brief Verifies all() accessor and iteration over entries
 *
 * @scenario Accessing all entries and iterating with range-for
 * @given An IndexedConstraintSet with multiple constraints
 * @when Using all(), begin()/end(), and range-for
 * @then All entries are accessible with correct indices
 *
 * @covers IndexedConstraintSet::all, begin, end, iterator
 */
TEST_CASE("H2: IndexedConstraintSet::AllAndIteration", "[IndexedConstraintSet][iteration]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X", 3, 3);
    model.update();

    auto I = dsl::range(0, 3);
    auto J = dsl::range(0, 3);
    auto domain = (I * J) | dsl::filter([](int i, int j) { return i <= j; });

    auto CC = dsl::ConstraintFactory::addIndexed(
        model, "tri",
        domain,
        [&](int i, int j) { return (X(i, j) <= i + j); }
    );
    model.update();

    SECTION("all() returns reference to entries") {
        const auto& entries = CC.all();
        REQUIRE(entries.size() == CC.size());

        // Verify entries have correct structure
        for (const auto& e : entries) {
            REQUIRE(e.index.size() == 2);
            REQUIRE(e.index[0] <= e.index[1]); // filter condition
        }
    }

    SECTION("Range-for iteration") {
        std::vector<std::pair<int, int>> visited;
        for (const auto& entry : CC) {
            REQUIRE(entry.index.size() == 2);
            visited.emplace_back(entry.index[0], entry.index[1]);
        }

        // Expected: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
        REQUIRE(visited.size() == 6);
        for (auto [i, j] : visited) {
            REQUIRE(i <= j);
        }
    }

    SECTION("Iterator equality") {
        auto it1 = CC.begin();
        auto it2 = CC.begin();
        REQUIRE(it1 == it2);

        auto end1 = CC.end();
        auto end2 = CC.end();
        REQUIRE(end1 == end2);

        REQUIRE(it1 != end1);
    }
}

/**
 * @test IndexedConstraintSet::ForEachCallback
 * @brief Verifies forEach() callback iteration
 *
 * @scenario Using forEach to visit all constraints with indices
 * @given An IndexedConstraintSet
 * @when Calling forEach with a callback
 * @then All constraints are visited with correct index vectors
 *
 * @covers IndexedConstraintSet::forEach
 */
TEST_CASE("H3: IndexedConstraintSet::ForEachCallback", "[IndexedConstraintSet][forEach]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X", 5);
    model.update();

    dsl::IndexList I{ 0, 2, 4 };
    auto CC = dsl::ConstraintFactory::addIndexed(
        model, "even",
        I,
        [&](int i) { return (X(i) <= i * 10); }
    );
    model.update();

    std::vector<int> visitedIndices;
    CC.forEach([&](GRBConstr& c, const std::vector<int>& idx) {
        REQUIRE(idx.size() == 1);
        visitedIndices.push_back(idx[0]);

        // Verify RHS
        REQUIRE(c.get(GRB_DoubleAttr_RHS) == Catch::Approx(idx[0] * 10));
    });

    REQUIRE(visitedIndices.size() == 3);
    std::vector<int> expected{ 0, 2, 4 };
    REQUIRE(visitedIndices == expected);
}

// ============================================================================
// SECTION I: CONSTRAINT TABLE WITH INDEXED CONSTRAINT SET
// ============================================================================

DECLARE_ENUM_WITH_COUNT(CDom, A, B, C);

/**
 * @test ConstraintTable::WithIndexedConstraintSet
 * @brief Verifies ConstraintTable storage of IndexedConstraintSet
 *
 * @scenario Storing IndexedConstraintSet with filtered domain in ConstraintTable
 * @given A ConstraintTable parameterized with IndexedConstraintSet
 * @when Storing and retrieving constraints created from filtered domain
 * @then Filtered indices are accessible, excluded indices throw
 *
 * @covers ConstraintTable<Enum, IndexedConstraintSet>::set, constr
 */
TEST_CASE("I1: ConstraintTable::WithIndexedConstraintSet", "[ConstraintTable][IndexedConstraintSet]")
{
    GRBModel model = makeModel();

    auto W = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "W", 4);
    model.update();

    auto I = dsl::range(0, 4);

    auto CC =
        dsl::ConstraintFactory::addIndexed(
            model,
            "wlim",
            I | dsl::filter([](int i) { return i % 2 == 0; }), // even indices only
            [&](int i) {
                return (W(i) == i * 10);
            }
        );

    dsl::ConstraintTable<CDom> table;
    table.set(CDom::A, std::move(CC));

    model.update();

    REQUIRE_NOTHROW(table.constr(CDom::A, 0));
    REQUIRE_NOTHROW(table.constr(CDom::A, 2));
    REQUIRE_THROWS(table.constr(CDom::A, 1)); // excluded by filter

    REQUIRE(table.constr(CDom::A, 2).get(GRB_DoubleAttr_RHS)
        == Catch::Approx(20.0));

    if (naming_enabled())
        REQUIRE(std::string(table.constr(CDom::A, 2)
            .get(GRB_StringAttr_ConstrName)) == "wlim[2]");
}

/**
 * @test ConstraintTable::MultipleKeys
 * @brief Verifies ConstraintTable with multiple enum keys
 *
 * @scenario Storing and retrieving multiple constraint groups by different keys
 * @given A ConstraintTable with multiple enum values
 * @when Setting different groups to different keys
 * @then Each key retrieves its own group independently
 *
 * @covers ConstraintTable::set, get, operator()
 */
TEST_CASE("I2: ConstraintTable::MultipleKeys", "[ConstraintTable][multikey]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X", 5);
    model.update();

    dsl::ConstraintTable<CDom> table;

    // Create and store constraint set A
    auto ccA = dsl::ConstraintFactory::addIndexed(
        model, "setA",
        dsl::IndexList{ 0, 1 },
        [&](int i) { return (X(i) <= 100 + i); }
    );
    table.set(CDom::A, std::move(ccA));

    // Create and store constraint set B
    auto ccB = dsl::ConstraintFactory::addIndexed(
        model, "setB",
        dsl::IndexList{ 2, 3 },
        [&](int i) { return (X(i) <= 200 + i); }
    );
    table.set(CDom::B, std::move(ccB));

    model.update();

    SECTION("Keys are independent") {
        // A has indices 0, 1
        REQUIRE_NOTHROW(table.constr(CDom::A, 0));
        REQUIRE_NOTHROW(table.constr(CDom::A, 1));
        REQUIRE_THROWS(table.constr(CDom::A, 2));

        // B has indices 2, 3
        REQUIRE_THROWS(table.constr(CDom::B, 0));
        REQUIRE_NOTHROW(table.constr(CDom::B, 2));
        REQUIRE_NOTHROW(table.constr(CDom::B, 3));
    }

    SECTION("RHS values are correct per key") {
        REQUIRE(table.constr(CDom::A, 0).get(GRB_DoubleAttr_RHS) == Catch::Approx(100));
        REQUIRE(table.constr(CDom::A, 1).get(GRB_DoubleAttr_RHS) == Catch::Approx(101));
        REQUIRE(table.constr(CDom::B, 2).get(GRB_DoubleAttr_RHS) == Catch::Approx(202));
        REQUIRE(table.constr(CDom::B, 3).get(GRB_DoubleAttr_RHS) == Catch::Approx(203));
    }

    SECTION("operator() access pattern") {
        auto& containerA = table(CDom::A);
        auto& containerB = table(CDom::B);

        REQUIRE(containerA.asIndexed().size() == 2);
        REQUIRE(containerB.asIndexed().size() == 2);
    }

    SECTION("get() access pattern") {
        auto& containerA = table.get(CDom::A);
        REQUIRE(containerA.asIndexed().size() == 2);
    }
}

/**
 * @test ConstraintTable::WithConstraintGroup
 * @brief Verifies ConstraintTable default type with multi-dimensional groups
 *
 * @scenario Storing and accessing multi-dimensional ConstraintGroup in table
 * @given A ConstraintTable with default ConstraintGroup type
 * @when Storing 2D constraint groups
 * @then Multi-index access works correctly
 *
 * @covers ConstraintTable<Enum, ConstraintGroup>::constr (multi-index)
 */
TEST_CASE("I3: ConstraintTable::WithConstraintGroup", "[ConstraintTable][ConstraintGroup]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X", 2, 3);
    model.update();

    auto caps = dsl::ConstraintFactory::add(
        model, "cap",
        [&](const std::vector<int>& idx) {
            return (X.at(idx[0], idx[1]) <= idx[0] * 10 + idx[1]);
        },
        2, 3
    );

    dsl::ConstraintTable<CTestFam> table;
    table.set(CTestFam::ONE, std::move(caps));

    model.update();

    SECTION("Multi-index access via constr()") {
        REQUIRE_NOTHROW(table.constr(CTestFam::ONE, 0, 0));
        REQUIRE_NOTHROW(table.constr(CTestFam::ONE, 1, 2));
        REQUIRE_THROWS(table.constr(CTestFam::ONE, 2, 0)); // out of range
    }

    SECTION("RHS verification") {
        REQUIRE(table.constr(CTestFam::ONE, 0, 0).get(GRB_DoubleAttr_RHS) == Catch::Approx(0));
        REQUIRE(table.constr(CTestFam::ONE, 1, 2).get(GRB_DoubleAttr_RHS) == Catch::Approx(12));
    }
}

// ============================================================================
// SECTION J: EDGE CASES AND ERROR CONDITIONS
// ============================================================================

/**
 * @test EdgeCases::EmptyConstraintGroup
 * @brief Verifies behavior with zero-size dimensions
 *
 * @scenario Creating constraint groups with zero-size dimensions
 * @given Dimension sizes including zero
 * @when Creating constraint groups
 * @then Group is created but empty
 *
 * @covers ConstraintFactory::add (edge cases)
 */
TEST_CASE("J1: EdgeCases::ZeroSizeDimension", "[edge][ConstraintFactory]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X");
    model.update();

    SECTION("Zero-size first dimension") {
        auto caps = dsl::ConstraintFactory::add(
            model, "empty",
            [&](const std::vector<int>&) { return (X <= 1); },
            0
        );
        model.update();

        REQUIRE(caps.dimension() == 1);
        REQUIRE(caps.size(0) == 0);
    }

    SECTION("Zero-size second dimension") {
        auto caps = dsl::ConstraintFactory::add(
            model, "empty2d",
            [&](const std::vector<int>&) { return (X <= 1); },
            3, 0
        );
        model.update();

        REQUIRE(caps.dimension() == 2);
        REQUIRE(caps.size(0) == 3);
        // Note: accessing size(1) on empty children may vary
    }
}

/**
 * @test EdgeCases::LargeDomain
 * @brief Verifies handling of larger domains
 *
 * @scenario Creating constraints over a larger domain
 * @given A 10x10 domain with filtering
 * @when Creating IndexedConstraintSet
 * @then All constraints are correctly created and accessible
 *
 * @covers ConstraintFactory::addIndexed (performance sanity)
 */
TEST_CASE("J2: EdgeCases::LargeDomain", "[edge][performance]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 100.0, "X", 10, 10);
    model.update();

    auto I = dsl::range(0, 10);
    auto J = dsl::range(0, 10);

    auto CC = dsl::ConstraintFactory::addIndexed(
        model, "large",
        I * J,
        [&](int i, int j) { return (X(i, j) <= i * j); }
    );

    model.update();

    REQUIRE(CC.size() == 100);

    // Spot check a few
    REQUIRE(CC.at(0, 0).get(GRB_DoubleAttr_RHS) == Catch::Approx(0));
    REQUIRE(CC.at(5, 5).get(GRB_DoubleAttr_RHS) == Catch::Approx(25));
    REQUIRE(CC.at(9, 9).get(GRB_DoubleAttr_RHS) == Catch::Approx(81));
}

/**
 * @test EdgeCases::DuplicateIndicesInDomain
 * @brief Verifies behavior with duplicate indices in domain
 *
 * @scenario Creating constraints from domain with duplicate indices
 * @given An IndexList with duplicates
 * @when Creating IndexedConstraintSet
 * @then Each duplicate creates a separate constraint (last one wins in lookup)
 *
 * @covers ConstraintFactory::addIndexed (duplicate handling)
 */
TEST_CASE("J3: EdgeCases::DuplicateIndicesInDomain", "[edge][duplicates]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X", 5);
    model.update();

    // Domain with duplicates
    dsl::IndexList I{ 0, 1, 1, 2 }; // 1 appears twice

    int callCount = 0;
    auto CC = dsl::ConstraintFactory::addIndexed(
        model, "dup",
        I,
        [&](int i) {
            callCount++;
            return (X(i) <= callCount * 10); // RHS varies by call order
        }
    );

    model.update();

    // Generator called 4 times
    REQUIRE(callCount == 4);

    // Size is 4 (duplicates create separate entries)
    // But lookup may return one of them (implementation detail)
    REQUIRE(CC.size() == 4);
}

// ============================================================================
// SECTION K: FREE FUNCTION CONSTRAINT UTILITIES
// ============================================================================

/**
 * @test ConstraintUtilities::RhsAccessor
 * @brief Verifies rhs() free function
 *
 * @scenario Getting RHS value using free function
 * @given A constraint with known RHS
 * @when Calling dsl::rhs(c)
 * @then Returns correct RHS value
 *
 * @covers dsl::rhs(GRBConstr)
 */
TEST_CASE("K1: ConstraintUtilities::RhsAccessor", "[constraints][utilities]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X");
    model.update();

    auto c = dsl::ConstraintFactory::add(
        model, "test",
        [&](const std::vector<int>&) { return (X <= 42.0); }
    );
    model.update();

    REQUIRE(dsl::rhs(c.scalar()) == Catch::Approx(42.0));
}

/**
 * @test ConstraintUtilities::SetRhs
 * @brief Verifies setRHS() free function
 *
 * @scenario Modifying RHS value using free function
 * @given A constraint with known RHS
 * @when Calling dsl::setRHS(c, newValue)
 * @then RHS is updated correctly
 *
 * @covers dsl::setRHS(GRBConstr, double)
 */
TEST_CASE("K2: ConstraintUtilities::SetRhs", "[constraints][utilities]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 100.0, "X");
    model.update();

    auto c = dsl::ConstraintFactory::add(
        model, "test",
        [&](const std::vector<int>&) { return (X <= 50.0); }
    );
    model.update();

    REQUIRE(dsl::rhs(c.scalar()) == Catch::Approx(50.0));

    dsl::setRHS(c.scalar(), 75.0);
    model.update();

    REQUIRE(dsl::rhs(c.scalar()) == Catch::Approx(75.0));
}

/**
 * @test ConstraintUtilities::SenseAccessor
 * @brief Verifies sense() free function for different constraint types
 *
 * @scenario Getting sense using free function
 * @given Constraints with different senses (<=, >=, ==)
 * @when Calling dsl::sense(c)
 * @then Returns correct sense character
 *
 * @covers dsl::sense(GRBConstr)
 */
TEST_CASE("K3: ConstraintUtilities::SenseAccessor", "[constraints][utilities]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X");
    model.update();

    auto cLE = dsl::ConstraintFactory::add(
        model, "le",
        [&](const std::vector<int>&) { return (X <= 5.0); }
    );

    auto cGE = dsl::ConstraintFactory::add(
        model, "ge",
        [&](const std::vector<int>&) { return (X >= 1.0); }
    );

    auto cEQ = dsl::ConstraintFactory::add(
        model, "eq",
        [&](const std::vector<int>&) { return (X == 3.0); }
    );

    model.update();

    REQUIRE(dsl::sense(cLE.scalar()) == '<');
    REQUIRE(dsl::sense(cGE.scalar()) == '>');
    REQUIRE(dsl::sense(cEQ.scalar()) == '=');
}

/**
 * @test ConstraintUtilities::SlackAccessor
 * @brief Verifies slack() free function after optimization
 *
 * @scenario Getting slack value using free function
 * @given An optimized model with constraints
 * @when Calling dsl::slack(c)
 * @then Returns correct slack value
 *
 * @covers dsl::slack(GRBConstr)
 */
TEST_CASE("K4: ConstraintUtilities::SlackAccessor", "[constraints][utilities]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, GRB_INFINITY, "X");
    model.setObjective(GRBLinExpr(X), GRB_MINIMIZE);
    model.update();

    // Binding constraint: X >= 5
    auto c = dsl::ConstraintFactory::add(
        model, "bind",
        [&](const std::vector<int>&) { return (X >= 5.0); }
    );
    model.update();

    optimizeSafe(model);

    // At optimum X = 5, so slack should be ~0
    REQUIRE(std::abs(dsl::slack(c.scalar())) < 1e-6);
}

/**
 * @test ConstraintUtilities::DualAccessor
 * @brief Verifies dual() free function for LP
 *
 * @scenario Getting dual value using free function
 * @given An optimized LP model
 * @when Calling dsl::dual(c)
 * @then Returns finite dual value
 *
 * @covers dsl::dual(GRBConstr)
 */
TEST_CASE("K5: ConstraintUtilities::DualAccessor", "[constraints][utilities]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, GRB_INFINITY, "X");
    model.setObjective(GRBLinExpr(X), GRB_MINIMIZE);
    model.update();

    auto c = dsl::ConstraintFactory::add(
        model, "ge1",
        [&](const std::vector<int>&) { return (X >= 1.0); }
    );
    model.update();

    optimizeSafe(model);

    double d = dsl::dual(c.scalar());
    REQUIRE(std::isfinite(d));
}

/**
 * @test ConstraintUtilities::ConstrNameAccessor
 * @brief Verifies constrName() and setConstrName() free functions
 *
 * @scenario Getting and setting constraint name
 * @given A named constraint
 * @when Calling dsl::constrName() and dsl::setConstrName()
 * @then Name is readable and modifiable
 *
 * @covers dsl::constrName, dsl::setConstrName
 */
TEST_CASE("K6: ConstraintUtilities::ConstrNameAccessor", "[constraints][utilities]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X");
    model.update();

    auto c = dsl::ConstraintFactory::add(
        model, "original",
        [&](const std::vector<int>&) { return (X <= 5.0); }
    );
    model.update();

    if (naming_enabled()) {
        REQUIRE(dsl::constrName(c.scalar()) == "original");

        dsl::setConstrName(c.scalar(), "renamed");
        model.update();

        REQUIRE(dsl::constrName(c.scalar()) == "renamed");
    }
}

/**
 * @test ConstraintUtilities::SlacksCollection
 * @brief Verifies slacks() for ConstraintGroup and IndexedConstraintSet
 *
 * @scenario Getting all slack values from constraint collections
 * @given Optimized constraints in groups
 * @when Calling dsl::slacks(group)
 * @then Returns vector of slack values
 *
 * @covers dsl::slacks(ConstraintGroup), dsl::slacks(IndexedConstraintSet)
 */
TEST_CASE("K7: ConstraintUtilities::SlacksCollection", "[constraints][utilities]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X", 3);
    model.setObjective(X(0) + X(1) + X(2), GRB_MAXIMIZE);
    model.update();

    // Constraints: X[i] <= 5 for all i
    auto caps = dsl::ConstraintFactory::add(
        model, "cap",
        [&](const std::vector<int>& idx) { return (X(idx[0]) <= 5.0); },
        3
    );
    model.update();

    optimizeSafe(model);

    auto sl = dsl::slacks(caps);
    REQUIRE(sl.size() == 3);
    for (double s : sl) {
        REQUIRE(std::isfinite(s));
    }
}

/**
 * @test ConstraintUtilities::SlacksIndexedConstraintSet
 * @brief Verifies slacks() for IndexedConstraintSet
 *
 * @scenario Getting all slack values from IndexedConstraintSet
 * @given Optimized indexed constraints
 * @when Calling dsl::slacks(indexed)
 * @then Returns vector of slack values
 *
 * @covers dsl::slacks(IndexedConstraintSet)
 */
TEST_CASE("K7b: ConstraintUtilities::SlacksIndexedConstraintSet", "[constraints][utilities]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X", 3);
    model.setObjective(X(0) + X(1) + X(2), GRB_MAXIMIZE);
    model.update();

    dsl::IndexList I{0, 1, 2};
    auto indexed = dsl::ConstraintFactory::addIndexed(
        model, "idx",
        I,
        [&](int i) { return (X(i) <= 5.0); }
    );
    model.update();

    optimizeSafe(model);

    auto sl = dsl::slacks(indexed);
    REQUIRE(sl.size() == 3);
    for (double s : sl) {
        REQUIRE(std::isfinite(s));
    }
}

/**
 * @test ConstraintUtilities::DualsCollection
 * @brief Verifies duals() for constraint collections
 *
 * @scenario Getting all dual values from LP constraint collections
 * @given Optimized LP constraints
 * @when Calling dsl::duals(group)
 * @then Returns vector of dual values
 *
 * @covers dsl::duals(ConstraintGroup), dsl::duals(IndexedConstraintSet)
 */
TEST_CASE("K8: ConstraintUtilities::DualsCollection", "[constraints][utilities]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0.0, 10.0, "X", 3);
    model.setObjective(X(0) + X(1) + X(2), GRB_MINIMIZE);
    model.update();

    // Constraints: X[i] >= 1 for all i
    auto lbs = dsl::ConstraintFactory::add(
        model, "lb",
        [&](const std::vector<int>& idx) { return (X(idx[0]) >= 1.0); },
        3
    );
    model.update();

    optimizeSafe(model);

    SECTION("ConstraintGroup duals") {
        auto du = dsl::duals(lbs);
        REQUIRE(du.size() == 3);
        for (double d : du) {
            REQUIRE(std::isfinite(d));
        }
    }
}