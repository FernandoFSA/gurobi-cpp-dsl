/*
===============================================================================
TEST VARIABLES — Comprehensive tests for variables.h
===============================================================================

OVERVIEW
--------
Validates the DSL variable system including VariableFactory, VariableGroup,
VariableTable, and IndexedVariableSet components. Tests cover scalar and N-D
variable creation, naming behavior, dimensional structure, iteration patterns,
bounds verification, exception handling, and integration with the indexing system.

TEST ORGANIZATION
-----------------
• Section A: Scalar variable creation, naming, and bounds
• Section B: N-dimensional variable creation and naming
• Section C: Dimensional structure and shape queries
• Section D: forEach iteration functionality
• Section E: VariableTable naming and access patterns
• Section F: VariableTable structural integrity and edge cases
• Section G: VariableFactory naming mode behavior
• Section H: IndexedVariableSet with various domain types
• Section I: VariableTable with IndexedVariableSet integration
• Section J: Exception behavior and error conditions
• Section K: Variable type coverage (BINARY, CONTINUOUS, INTEGER)
• Section L: Variable modification utilities (fix, unfix, setStart, bounds)
• Section M: Solution extraction utilities (value, values, valueAt)

TEST STRATEGY
-------------
• Verify variable creation with correct bounds and types
• Confirm naming behavior respects debug/release mode via naming_enabled()
• Validate dimensional queries (shape, size, dimension)
• Exercise iteration patterns with forEach callbacks
• Test VariableTable enum-indexed storage and retrieval
• Verify IndexedVariableSet works with IndexList, Cartesian, and filtered domains
• Test exception behavior for out-of-bounds and invalid access
• Verify all GRB variable types work correctly
• Test variable modification utilities (fix, unfix, setStart, bounds)
• Validate solution extraction after optimization (value, values, valueAt)

DEPENDENCIES
------------
• Catch2 v3.0+ - Test framework
• variables.h - System under test
• indexing.h - Index domain types (IndexList, Cartesian, filter)
• naming.h - naming_enabled() function
• enum_utils.h - DECLARE_ENUM_WITH_COUNT macro
• Gurobi - GRBModel, GRBVar, GRBEnv

BUILD CONFIGURATION NOTES
-------------------------
Tests adapt to debug/release mode via naming_enabled(). In release mode,
symbolic naming tests are skipped; structural and access tests run in all modes.

===============================================================================
*/

#define CATCH_CONFIG_MAIN
#include "catch_amalgamated.hpp"
#include <gurobi_dsl/variables.h>
#include <gurobi_dsl/indexing.h>
#include <gurobi_dsl/naming.h>
#include <gurobi_dsl/enum_utils.h>

#include <set>
#include <map>

// ============================================================================
// UTILITY
// ============================================================================

static GRBModel makeModel() {
    static GRBEnv env = GRBEnv(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    return GRBModel(env);
}

// ============================================================================
// ENUM DECLARATIONS FOR VARIABLETABLE TESTS
// ============================================================================

DECLARE_ENUM_WITH_COUNT(TestVars, A, B);
DECLARE_ENUM_WITH_COUNT(LargeVars, V0, V1, V2, V3, V4);

// ============================================================================
// SECTION A: SCALAR VARIABLE CREATION, NAMING, AND BOUNDS
// ============================================================================

/**
 * @test ScalarNaming::MatchesDSLDesign
 * @brief Verifies scalar variable naming behavior matches DSL design
 *
 * @scenario A scalar variable is created with a symbolic name
 * @given A GRBModel and VariableFactory
 * @when Creating a scalar binary variable with name "x"
 * @then The variable name matches in debug mode; structure validated in release
 *
 * @covers VariableFactory::add() for scalar variables
 * @covers naming_enabled() integration
 */
TEST_CASE("A1: ScalarNaming::MatchesDSLDesign", "[variables][scalar][naming]")
{
    GRBModel model = makeModel();

    auto x = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "x");
    model.update();

    std::string nm = x.get(GRB_StringAttr_VarName);

    if (naming_enabled()) {
        REQUIRE(nm == "x");
    }
    else {
        SUCCEED("Release mode: naming unspecified; scalar variable exists.");
    }
}

/**
 * @test ScalarBounds::BoundsAreCorrectlySet
 * @brief Verifies scalar variable bounds are correctly applied
 *
 * @scenario Scalar variables are created with various bounds
 * @given A GRBModel and VariableFactory
 * @when Creating scalar variables with different lb/ub values
 * @then The bounds are correctly set on the GRBVar
 *
 * @covers VariableFactory::add() bounds parameter
 */
TEST_CASE("A2: ScalarBounds::BoundsAreCorrectlySet", "[variables][scalar][bounds]")
{
    GRBModel model = makeModel();

    SECTION("Binary variable bounds") {
        auto x = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "x");
        model.update();

        REQUIRE(x.get(GRB_DoubleAttr_LB) == Catch::Approx(0.0));
        REQUIRE(x.get(GRB_DoubleAttr_UB) == Catch::Approx(1.0));
    }

    SECTION("Continuous variable with custom bounds") {
        auto y = dsl::VariableFactory::add(model, GRB_CONTINUOUS, -10.5, 25.3, "y");
        model.update();

        REQUIRE(y.get(GRB_DoubleAttr_LB) == Catch::Approx(-10.5));
        REQUIRE(y.get(GRB_DoubleAttr_UB) == Catch::Approx(25.3));
    }

    SECTION("Integer variable with zero-based bounds") {
        auto z = dsl::VariableFactory::add(model, GRB_INTEGER, 0, 100, "z");
        model.update();

        REQUIRE(z.get(GRB_DoubleAttr_LB) == Catch::Approx(0.0));
        REQUIRE(z.get(GRB_DoubleAttr_UB) == Catch::Approx(100.0));
    }
}

/**
 * @test ScalarTypes::AllVariableTypesSupported
 * @brief Verifies all GRB variable types can be created as scalars
 *
 * @scenario Scalar variables of each type are created
 * @given A GRBModel and VariableFactory
 * @when Creating BINARY, CONTINUOUS, and INTEGER scalars
 * @then Each variable has the correct type attribute
 *
 * @covers VariableFactory::add() with all GRB types
 */
TEST_CASE("A3: ScalarTypes::AllVariableTypesSupported", "[variables][scalar][types]")
{
    GRBModel model = makeModel();

    auto binary = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "bin");
    auto continuous = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "cont");
    auto integer = dsl::VariableFactory::add(model, GRB_INTEGER, 0, 10, "int");
    model.update();

    REQUIRE(binary.get(GRB_CharAttr_VType) == GRB_BINARY);
    REQUIRE(continuous.get(GRB_CharAttr_VType) == GRB_CONTINUOUS);
    REQUIRE(integer.get(GRB_CharAttr_VType) == GRB_INTEGER);
}

// ============================================================================
// SECTION B: N-DIMENSIONAL VARIABLE CREATION AND NAMING
// ============================================================================

/**
 * @test NDNaming::ProducesCorrectSymbolicNames
 * @brief Verifies N-D variable naming produces correct symbolic names
 *
 * @scenario A 2D variable array is created with symbolic naming
 * @given A GRBModel and VariableFactory
 * @when Creating a 3x4 continuous variable array with name "X"
 * @then Variable names follow "X_i_j" format in debug mode
 *
 * @covers VariableFactory::add() for N-D variables
 * @covers VariableGroup::at() access
 */
TEST_CASE("B1: NDNaming::ProducesCorrectSymbolicNames", "[variables][nd][naming]")
{
    GRBModel model = makeModel();
    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 5, "X", 3, 4);
    model.update();

    if (!naming_enabled()) {
        SUCCEED("Release mode: naming not checked.");
        return;
    }

    REQUIRE(X.at(0, 0).get(GRB_StringAttr_VarName) == "X_0_0");
    REQUIRE(X.at(2, 3).get(GRB_StringAttr_VarName) == "X_2_3");
}

/**
 * @test NDDimensions::1DArrayCreation
 * @brief Verifies 1D variable array creation and access
 *
 * @scenario A 1D variable array is created
 * @given A GRBModel and VariableFactory
 * @when Creating a 1D array with 5 elements
 * @then All elements are accessible with correct properties
 *
 * @covers VariableFactory::add() for 1D arrays
 * @covers VariableGroup::dimension() for 1D
 */
TEST_CASE("B2: NDDimensions::1DArrayCreation", "[variables][nd][1D]")
{
    GRBModel model = makeModel();
    auto V = dsl::VariableFactory::add(model, GRB_INTEGER, 0, 50, "V", 5);
    model.update();

    REQUIRE(V.dimension() == 1);
    REQUIRE(V.size(0) == 5);

    for (int i = 0; i < 5; ++i) {
        REQUIRE_NOTHROW(V.at(i));
        REQUIRE(V.at(i).get(GRB_DoubleAttr_UB) == Catch::Approx(50.0));
    }

    if (naming_enabled()) {
        REQUIRE(V.at(0).get(GRB_StringAttr_VarName) == "V_0");
        REQUIRE(V.at(4).get(GRB_StringAttr_VarName) == "V_4");
    }
}

/**
 * @test NDDimensions::3DArrayCreation
 * @brief Verifies 3D variable array creation and access
 *
 * @scenario A 3D variable tensor is created
 * @given A GRBModel and VariableFactory
 * @when Creating a 2x3x4 tensor
 * @then All 24 elements are accessible with correct naming
 *
 * @covers VariableFactory::add() for 3D arrays
 * @covers VariableGroup with 3 dimensions
 */
TEST_CASE("B3: NDDimensions::3DArrayCreation", "[variables][nd][3D]")
{
    GRBModel model = makeModel();
    auto T = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "T", 2, 3, 4);
    model.update();

    REQUIRE(T.dimension() == 3);

    auto shp = T.shape();
    REQUIRE(shp.size() == 3);
    REQUIRE(shp[0] == 2);
    REQUIRE(shp[1] == 3);
    REQUIRE(shp[2] == 4);

    // Count total variables
    int count = 0;
    T.forEach([&](GRBVar&, const std::vector<int>&) { count++; });
    REQUIRE(count == 24);  // 2 * 3 * 4

    // Check corner elements
    REQUIRE_NOTHROW(T.at(0, 0, 0));
    REQUIRE_NOTHROW(T.at(1, 2, 3));

    if (naming_enabled()) {
        REQUIRE(T.at(0, 0, 0).get(GRB_StringAttr_VarName) == "T_0_0_0");
        REQUIRE(T.at(1, 2, 3).get(GRB_StringAttr_VarName) == "T_1_2_3");
    }
}

/**
 * @test NDBounds::AllElementsHaveCorrectBounds
 * @brief Verifies all elements in N-D array have correct bounds
 *
 * @scenario A 2D array is created with specific bounds
 * @given A GRBModel and VariableFactory
 * @when Creating a 3x3 continuous array with bounds [-5, 15]
 * @then All 9 elements have the correct bounds
 *
 * @covers VariableFactory::add() bounds propagation
 */
TEST_CASE("B4: NDBounds::AllElementsHaveCorrectBounds", "[variables][nd][bounds]")
{
    GRBModel model = makeModel();
    auto M = dsl::VariableFactory::add(model, GRB_CONTINUOUS, -5.0, 15.0, "M", 3, 3);
    model.update();

    M.forEach([](GRBVar& v, const std::vector<int>&) {
        REQUIRE(v.get(GRB_DoubleAttr_LB) == Catch::Approx(-5.0));
        REQUIRE(v.get(GRB_DoubleAttr_UB) == Catch::Approx(15.0));
    });
}

/**
 * @test NDAccess::OperatorParenthesisEquivalentToAt
 * @brief Verifies operator() is equivalent to at()
 *
 * @scenario A 2D array is accessed with both syntaxes
 * @given A 2D variable array
 * @when Accessing elements with operator() and at()
 * @then Both return references to the same variable
 *
 * @covers VariableGroup::operator()
 * @covers VariableGroup::at()
 */
TEST_CASE("B5: NDAccess::OperatorParenthesisEquivalentToAt", "[variables][nd][access]")
{
    GRBModel model = makeModel();
    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 3, 4);
    model.update();

    // Both should return references to the same variable
    GRBVar& v1 = X.at(1, 2);
    GRBVar& v2 = X(1, 2);

    // Compare by address
    REQUIRE(&v1 == &v2);
}

// ============================================================================
// SECTION C: DIMENSIONAL STRUCTURE AND SHAPE QUERIES
// ============================================================================

/**
 * @test DimensionalStructure::ShapeAndSizeCorrect
 * @brief Verifies dimensional structure and shape() are correct
 *
 * @scenario A 2D variable array is queried for dimensional properties
 * @given A 3x4 binary variable array
 * @when Querying dimension(), shape(), and size()
 * @then All dimensional queries return correct values
 *
 * @covers VariableGroup::dimension()
 * @covers VariableGroup::shape()
 * @covers VariableGroup::size()
 * @covers VariableGroup::at()
 */
TEST_CASE("C1: DimensionalStructure::ShapeAndSizeCorrect", "[variables][structure][shape]")
{
    GRBModel model = makeModel();

    auto M = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "M", 3, 4);
    model.update();

    REQUIRE(M.dimension() == 2);

    auto shp = M.shape();
    REQUIRE(shp.size() == 2);
    REQUIRE(shp[0] == 3);
    REQUIRE(shp[1] == 4);

    REQUIRE(M.size(0) == 3);
    REQUIRE(M.size(1) == 4);

    REQUIRE_NOTHROW(M.at(0, 0));
    REQUIRE_NOTHROW(M.at(2, 3));
}

/**
 * @test DimensionalStructure::ScalarIsZeroDimensional
 * @brief Verifies scalar variables are 0-dimensional
 *
 * @scenario A scalar variable is queried for dimensional properties
 * @given A scalar GRBVar wrapped in VariableGroup
 * @when Querying dimension(), isScalar(), isMultiDimensional()
 * @then Scalar-specific properties are correct
 *
 * @covers VariableGroup::isScalar()
 * @covers VariableGroup::isMultiDimensional()
 * @covers VariableGroup::scalar()
 */
TEST_CASE("C2: DimensionalStructure::ScalarIsZeroDimensional", "[variables][structure][scalar]")
{
    GRBModel model = makeModel();

    auto x = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "x");
    dsl::VariableGroup scalar(x);
    model.update();

    REQUIRE(scalar.dimension() == 0);
    REQUIRE(scalar.isScalar() == true);
    REQUIRE(scalar.isMultiDimensional() == false);

    // Shape should be empty for scalar
    auto shp = scalar.shape();
    REQUIRE(shp.empty());

    // scalar() should work
    REQUIRE_NOTHROW(scalar.scalar());
}

/**
 * @test DimensionalStructure::SizeOutOfRangeThrows
 * @brief Verifies size() throws for invalid dimension indices
 *
 * @scenario size() is called with out-of-range dimension
 * @given A 2D variable array
 * @when Calling size() with dim < 0 or dim >= dimension()
 * @then std::out_of_range is thrown
 *
 * @covers VariableGroup::size() exception behavior
 */
TEST_CASE("C3: DimensionalStructure::SizeOutOfRangeThrows", "[variables][structure][exception]")
{
    GRBModel model = makeModel();
    auto M = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "M", 3, 4);
    model.update();

    REQUIRE_THROWS_AS(M.size(-1), std::out_of_range);
    REQUIRE_THROWS_AS(M.size(2), std::out_of_range);
    REQUIRE_THROWS_AS(M.size(10), std::out_of_range);
}

/**
 * @test DimensionalStructure::VectorIndexAccess
 * @brief Verifies vector-based index access works correctly
 *
 * @scenario A 2D array is accessed with vector indices
 * @given A 2D variable array
 * @when Accessing with at(vector) method
 * @then Same variable is returned as with variadic at()
 *
 * @covers VariableGroup::at(vector)
 */
TEST_CASE("C4: DimensionalStructure::VectorIndexAccess", "[variables][structure][access]")
{
    GRBModel model = makeModel();
    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 3, 4);
    model.update();

    std::vector<int> idx = {1, 2};
    GRBVar& v1 = X.at(idx);
    GRBVar& v2 = X.at(1, 2);

    REQUIRE(&v1 == &v2);

    // Empty vector for scalar
    auto s = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "s");
    dsl::VariableGroup scalar(s);
    model.update();

    std::vector<int> emptyIdx = {};
    REQUIRE_NOTHROW(scalar.at(emptyIdx));
}

/**
 * @test DimensionalStructure::CountReturnsTotalVariables
 * @brief Verifies count() returns total number of variables
 *
 * @scenario Various VariableGroups are queried for total count
 * @given Scalar, 1D, 2D, and 3D variable groups
 * @when Calling count() on each
 * @then Correct total variable count is returned
 *
 * @covers VariableGroup::count()
 */
TEST_CASE("C5: DimensionalStructure::CountReturnsTotalVariables", "[variables][structure][count]")
{
    GRBModel model = makeModel();

    SECTION("Scalar has count 1") {
        auto x = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "x");
        dsl::VariableGroup scalar(x);
        model.update();

        REQUIRE(scalar.count() == 1);
    }

    SECTION("1D array count equals size") {
        auto V = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "V", 5);
        model.update();

        REQUIRE(V.count() == 5);
    }

    SECTION("2D array count is product of dimensions") {
        auto M = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "M", 3, 4);
        model.update();

        REQUIRE(M.count() == 12);  // 3 * 4
    }

    SECTION("3D tensor count is product of all dimensions") {
        auto T = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "T", 2, 3, 4);
        model.update();

        REQUIRE(T.count() == 24);  // 2 * 3 * 4
    }
}

// ============================================================================
// SECTION D: FOREACH ITERATION FUNCTIONALITY
// ============================================================================

/**
 * @test ForEachIteration::IteratesAllElements
 * @brief Verifies forEach iterates over all GRBVar elements with indices
 *
 * @scenario A 2D variable array is iterated using forEach
 * @given A 2x3 binary variable array
 * @when Iterating with forEach callback
 * @then Callback is invoked 6 times with valid indices
 *
 * @covers VariableGroup::forEach()
 */
TEST_CASE("D1: ForEachIteration::IteratesAllElements", "[variables][iteration][forEach]")
{
    GRBModel model = makeModel();
    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 2, 3);
    model.update();

    int count = 0;

    X.forEach([&](GRBVar&, const std::vector<int>& idx) {
        REQUIRE(idx.size() == 2);

        REQUIRE(idx[0] >= 0);
        REQUIRE(idx[0] < 2);

        REQUIRE(idx[1] >= 0);
        REQUIRE(idx[1] < 3);

        count++;
    });

    REQUIRE(count == 6); // 2 × 3
}

/**
 * @test ForEachIteration::1DIteration
 * @brief Verifies forEach works correctly for 1D arrays
 *
 * @scenario A 1D variable array is iterated
 * @given A 1D array with 5 elements
 * @when Iterating with forEach
 * @then Each index vector has size 1 with correct values
 *
 * @covers VariableGroup::forEach() for 1D
 */
TEST_CASE("D2: ForEachIteration::1DIteration", "[variables][iteration][1D]")
{
    GRBModel model = makeModel();
    auto V = dsl::VariableFactory::add(model, GRB_INTEGER, 0, 10, "V", 5);
    model.update();

    std::set<int> seen;

    V.forEach([&](GRBVar&, const std::vector<int>& idx) {
        REQUIRE(idx.size() == 1);
        REQUIRE(idx[0] >= 0);
        REQUIRE(idx[0] < 5);
        seen.insert(idx[0]);
    });

    REQUIRE(seen.size() == 5);
    for (int i = 0; i < 5; ++i) {
        REQUIRE(seen.count(i) == 1);
    }
}

/**
 * @test ForEachIteration::3DIteration
 * @brief Verifies forEach works correctly for 3D arrays
 *
 * @scenario A 3D variable tensor is iterated
 * @given A 2x2x2 tensor
 * @when Iterating with forEach
 * @then All 8 elements are visited with correct 3-element indices
 *
 * @covers VariableGroup::forEach() for 3D
 */
TEST_CASE("D3: ForEachIteration::3DIteration", "[variables][iteration][3D]")
{
    GRBModel model = makeModel();
    auto T = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "T", 2, 2, 2);
    model.update();

    int count = 0;

    T.forEach([&](GRBVar&, const std::vector<int>& idx) {
        REQUIRE(idx.size() == 3);
        for (int d = 0; d < 3; ++d) {
            REQUIRE(idx[d] >= 0);
            REQUIRE(idx[d] < 2);
        }
        count++;
    });

    REQUIRE(count == 8);  // 2^3
}

/**
 * @test ForEachIteration::ScalarForEach
 * @brief Verifies forEach works for scalar VariableGroup
 *
 * @scenario A scalar VariableGroup is iterated
 * @given A scalar variable
 * @when Iterating with forEach
 * @then Callback is invoked once with empty index vector
 *
 * @covers VariableGroup::forEach() for scalars
 */
TEST_CASE("D4: ForEachIteration::ScalarForEach", "[variables][iteration][scalar]")
{
    GRBModel model = makeModel();
    auto x = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "x");
    dsl::VariableGroup scalar(x);
    model.update();

    int count = 0;

    scalar.forEach([&](GRBVar&, const std::vector<int>& idx) {
        REQUIRE(idx.empty());
        count++;
    });

    REQUIRE(count == 1);
}

/**
 * @test ForEachIteration::ConstForEach
 * @brief Verifies const forEach works correctly
 *
 * @scenario A const VariableGroup is iterated
 * @given A const 2D variable array
 * @when Iterating with forEach on const object
 * @then Iteration works without modification capability
 *
 * @covers VariableGroup::forEach() const
 */
TEST_CASE("D5: ForEachIteration::ConstForEach", "[variables][iteration][const]")
{
    GRBModel model = makeModel();
    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 2, 3);
    model.update();

    const auto& constX = X;
    int count = 0;

    constX.forEach([&](const GRBVar&, const std::vector<int>& idx) {
        REQUIRE(idx.size() == 2);
        count++;
    });

    REQUIRE(count == 6);
}

// ============================================================================
// SECTION E: VARIABLETABLE NAMING AND ACCESS PATTERNS
// ============================================================================

/**
 * @test VariableTableNaming::NameGenerationAndAccess
 * @brief Verifies VariableTable name generation and access correctness
 *
 * @scenario Variables are stored and retrieved from VariableTable
 * @given A VariableTable with enum-indexed entries
 * @when Setting and retrieving variables by enum key
 * @then Names are correct in debug mode; access works in all modes
 *
 * @covers VariableTable::set()
 * @covers VariableTable::var()
 * @covers VariableTable with enum keys
 */
TEST_CASE("E1: VariableTableNaming::NameGenerationAndAccess", "[variables][table][naming]")
{
    GRBModel model = makeModel();

    dsl::VariableTable<TestVars> vt;  // MAX inferred from TestVars::COUNT

    auto A = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "A");
    auto B = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "B", 2);

    vt.set(TestVars::A, A);
    vt.set(TestVars::B, std::move(B));
    model.update();

    // Naming correctness in debug mode
    if (naming_enabled()) {
        REQUIRE(vt.var(TestVars::A).get(GRB_StringAttr_VarName) == "A");
        REQUIRE(vt.var(TestVars::B, 1).get(GRB_StringAttr_VarName) == "B_1");
    }
    else {
        SUCCEED("Release: names are unspecified, access verified below.");
    }

    // Access correctness regardless of naming mode
    REQUIRE_NOTHROW(vt.var(TestVars::A));
    REQUIRE_NOTHROW(vt.var(TestVars::B, 0));
    REQUIRE_NOTHROW(vt.var(TestVars::B, 1));
}

/**
 * @test VariableTableAccess::OperatorParenthesisEquivalentToGet
 * @brief Verifies operator() is equivalent to get()
 *
 * @scenario A VariableTable is accessed with both syntaxes
 * @given A VariableTable with entries
 * @when Accessing with operator() and get()
 * @then Both return references to the same VariableContainer
 *
 * @covers VariableTable::operator()
 * @covers VariableTable::get()
 */
TEST_CASE("E2: VariableTableAccess::OperatorParenthesisEquivalentToGet", "[variables][table][access]")
{
    GRBModel model = makeModel();
    dsl::VariableTable<TestVars> vt;

    auto A = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "A", 3);
    vt.set(TestVars::A, std::move(A));
    model.update();

    dsl::VariableContainer& c1 = vt.get(TestVars::A);
    dsl::VariableContainer& c2 = vt(TestVars::A);

    REQUIRE(&c1 == &c2);
}

/**
 * @test VariableTableAccess::ScalarVarAccess
 * @brief Verifies var() without indices accesses scalar
 *
 * @scenario A scalar variable is accessed via var()
 * @given A VariableTable with a scalar entry
 * @when Calling var(key) without indices
 * @then The scalar variable is returned
 *
 * @covers VariableTable::var() for scalars
 */
TEST_CASE("E3: VariableTableAccess::ScalarVarAccess", "[variables][table][scalar]")
{
    GRBModel model = makeModel();
    dsl::VariableTable<TestVars> vt;

    auto x = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "x");
    vt.set(TestVars::A, x);
    model.update();

    GRBVar& v = vt.var(TestVars::A);
    REQUIRE(v.get(GRB_DoubleAttr_LB) == Catch::Approx(0.0));
    REQUIRE(v.get(GRB_DoubleAttr_UB) == Catch::Approx(1.0));
}

/**
 * @test VariableTableAccess::LargeEnumSupport
 * @brief Verifies VariableTable works with larger enums
 *
 * @scenario A VariableTable with 5 enum values is used
 * @given A VariableTable with LargeVars enum
 * @when Setting and retrieving all 5 entries
 * @then All entries are accessible
 *
 * @covers VariableTable with larger enums
 */
TEST_CASE("E4: VariableTableAccess::LargeEnumSupport", "[variables][table][enum]")
{
    GRBModel model = makeModel();
    dsl::VariableTable<LargeVars> vt;

    auto V0 = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "V0");
    auto V1 = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "V1", 2);
    auto V2 = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "V2", 3);
    auto V3 = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "V3", 2, 2);
    auto V4 = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "V4");

    vt.set(LargeVars::V0, V0);
    vt.set(LargeVars::V1, std::move(V1));
    vt.set(LargeVars::V2, std::move(V2));
    vt.set(LargeVars::V3, std::move(V3));
    vt.set(LargeVars::V4, V4);
    model.update();

    // Access mode-specific properties via asGroup()
    REQUIRE(vt.get(LargeVars::V0).isScalar());
    REQUIRE(vt.get(LargeVars::V1).asGroup().dimension() == 1);
    REQUIRE(vt.get(LargeVars::V2).asGroup().dimension() == 1);
    REQUIRE(vt.get(LargeVars::V3).asGroup().dimension() == 2);
    REQUIRE(vt.get(LargeVars::V4).isScalar());

    REQUIRE_NOTHROW(vt.var(LargeVars::V3, 1, 1));
}

// ============================================================================
// SECTION F: VARIABLETABLE STRUCTURAL INTEGRITY AND EDGE CASES
// ============================================================================

/**
 * @test VariableTableStructure::IntegrityIndependentOfNaming
 * @brief Verifies VariableTable structure is independent of naming mode
 *
 * @scenario VariableTable stores scalar and 1D variables
 * @given A VariableTable with mixed-dimension variables
 * @when Querying structural properties
 * @then Structure is correct regardless of naming mode
 *
 * @covers VariableTable::get()
 * @covers VariableGroup::isScalar()
 * @covers VariableGroup::dimension()
 */
TEST_CASE("F1: VariableTableStructure::IntegrityIndependentOfNaming", "[variables][table][structure]")
{
    GRBModel model = makeModel();

    dsl::VariableTable<TestVars> vt;

    auto A = dsl::VariableFactory::add(model, GRB_INTEGER, 0, 10, "A");
    auto B = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 5, "B", 3);

    vt.set(TestVars::A, A);
    vt.set(TestVars::B, std::move(B));
    model.update();

    REQUIRE(vt.get(TestVars::A).isScalar());
    REQUIRE(vt.get(TestVars::B).asGroup().dimension() == 1);

    REQUIRE_NOTHROW(vt.var(TestVars::B, 2));
}

/**
 * @test VariableTableStructure::OverwriteEntry
 * @brief Verifies VariableTable entries can be overwritten
 *
 * @scenario A VariableTable entry is set twice
 * @given A VariableTable with an entry
 * @when Setting the same key with a different value
 * @then The new value replaces the old one
 *
 * @covers VariableTable::set() overwrite behavior
 */
TEST_CASE("F2: VariableTableStructure::OverwriteEntry", "[variables][table][overwrite]")
{
    GRBModel model = makeModel();
    dsl::VariableTable<TestVars> vt;

    // First assignment: scalar
    auto x1 = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "x1");
    vt.set(TestVars::A, x1);
    model.update();

    REQUIRE(vt.get(TestVars::A).isScalar());

    // Second assignment: 1D array
    auto x2 = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "x2", 5);
    vt.set(TestVars::A, std::move(x2));
    model.update();

    REQUIRE(vt.get(TestVars::A).asGroup().dimension() == 1);
    REQUIRE(vt.get(TestVars::A).asGroup().size(0) == 5);
}

/**
 * @test VariableTableStructure::ConstAccess
 * @brief Verifies const access to VariableTable works
 *
 * @scenario A const VariableTable is accessed
 * @given A const reference to VariableTable
 * @when Calling get() and var() on const object
 * @then Access works correctly
 *
 * @covers VariableTable::get() const
 * @covers VariableTable::var() const
 */
TEST_CASE("F3: VariableTableStructure::ConstAccess", "[variables][table][const]")
{
    GRBModel model = makeModel();
    dsl::VariableTable<TestVars> vt;

    auto A = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "A", 3);
    vt.set(TestVars::A, std::move(A));
    model.update();

    const auto& constVt = vt;

    REQUIRE(constVt.get(TestVars::A).asGroup().dimension() == 1);
    REQUIRE_NOTHROW(constVt.var(TestVars::A, 0));
    REQUIRE_NOTHROW(constVt.var(TestVars::A, 2));
}

// ============================================================================
// SECTION G: VARIABLEFACTORY NAMING MODE BEHAVIOR
// ============================================================================

/**
 * @test VariableFactoryNaming::NamingEnabledAffectsOnlySymbolic
 * @brief Verifies naming_enabled affects only symbolic naming
 *
 * @scenario A 1D variable is created and validated in both modes
 * @given A 1D binary variable array
 * @when Checking names in debug mode, structure in all modes
 * @then Names match in debug; structure always correct
 *
 * @covers VariableFactory::add()
 * @covers naming_enabled() behavior
 */
TEST_CASE("G1: VariableFactoryNaming::NamingEnabledAffectsOnlySymbolic", "[variables][factory][naming]")
{
    GRBModel model = makeModel();

    auto C = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "C", 2);
    model.update();

    if (naming_enabled()) {
        REQUIRE(C.at(1).get(GRB_StringAttr_VarName) == "C_1");
    }
    else {
        SUCCEED("Release: VarName ignored; existence and structure validated.");
    }

    REQUIRE(C.dimension() == 1);
    REQUIRE_NOTHROW(C.at(0));
    REQUIRE_NOTHROW(C.at(1));
}

// ============================================================================
// SECTION H: INDEXEDVARIABLESET WITH VARIOUS DOMAIN TYPES
// ============================================================================

/**
 * @test IndexedVariableSet1D::DomainUsingIndexList
 * @brief Verifies IndexedVariableSet with 1D IndexList domain
 *
 * @scenario Variables are created over an IndexList domain
 * @given An IndexList {0, 1, 2, 3} as domain
 * @when Creating IndexedVariableSet and accessing elements
 * @then All domain indices are accessible; out-of-domain returns nullptr
 *
 * @covers VariableFactory::addIndexed() with IndexList
 * @covers IndexedVariableSet::at()
 * @covers IndexedVariableSet::try_get()
 */
TEST_CASE("H1: IndexedVariableSet1D::DomainUsingIndexList", "[variables][indexed][IndexList]")
{
    GRBModel model = makeModel();

    auto I = dsl::IndexList{ 0, 1, 2, 3 };
    auto X = dsl::VariableFactory::addIndexed(
        model,
        GRB_CONTINUOUS,
        0.0,
        10.0,
        "X",
        I
    );
    model.update();

    REQUIRE(X.size() == I.size());
    for (int i : I) {
        REQUIRE_NOTHROW(X.at(i));
        GRBVar& v = X.at(i);
        REQUIRE(v.get(GRB_DoubleAttr_LB) == Catch::Approx(0.0));
        REQUIRE(v.get(GRB_DoubleAttr_UB) == Catch::Approx(10.0));
    }

    REQUIRE(X.try_get(99) == nullptr);

    if (naming_enabled()) {
        REQUIRE(std::string(X.at(0).get(GRB_StringAttr_VarName)) == "X_0");
        REQUIRE(std::string(X.at(3).get(GRB_StringAttr_VarName)) == "X_3");
    }
}

/**
 * @test IndexedVariableSet2D::CartesianDomain
 * @brief Verifies IndexedVariableSet with 2D Cartesian domain
 *
 * @scenario Variables are created over a Cartesian product I * J
 * @given Two IndexLists forming a Cartesian product domain
 * @when Creating IndexedVariableSet and accessing elements
 * @then All (i,j) pairs in domain are accessible; invalid pairs throw/return nullptr
 *
 * @covers VariableFactory::addIndexed() with Cartesian product
 * @covers IndexedVariableSet::at() with 2D indices
 * @covers IndexedVariableSet::try_get() with 2D indices
 */
TEST_CASE("H2: IndexedVariableSet2D::CartesianDomain", "[variables][indexed][Cartesian]")
{
    GRBModel model = makeModel();

    auto I = dsl::range(0, 2); // {0,1}
    auto J = dsl::range(0, 3); // {0,1,2}

    auto Y = dsl::VariableFactory::addIndexed(
        model,
        GRB_BINARY,
        0.0,
        1.0,
        "Y",
        I * J
    );
    model.update();

    REQUIRE(Y.size() == 6);

    REQUIRE_NOTHROW(Y.at(0, 0));
    REQUIRE_NOTHROW(Y.at(1, 2));
    REQUIRE_THROWS(Y.at(2, 0));  // i==2 not in I
    REQUIRE(Y.try_get(1, 1) != nullptr);
    REQUIRE(Y.try_get(5, 5) == nullptr);

    if (naming_enabled()) {
        REQUIRE(std::string(Y.at(0, 0).get(GRB_StringAttr_VarName)) == "Y_0_0");
        REQUIRE(std::string(Y.at(1, 2).get(GRB_StringAttr_VarName)) == "Y_1_2");
    }
}



/**
 * @test IndexedVariableSet2D::FilteredDomain
 * @brief Verifies IndexedVariableSet with filtered 2D domain
 *
 * @scenario Variables are created over a filtered Cartesian domain (i < j)
 * @given A Cartesian product filtered by predicate i < j
 * @when Creating IndexedVariableSet and accessing elements
 * @then Only filtered pairs are accessible; excluded pairs return nullptr
 *
 * @covers VariableFactory::addIndexed() with filtered domain
 * @covers dsl::filter with Cartesian product
 */
TEST_CASE("H3: IndexedVariableSet2D::FilteredDomain", "[variables][indexed][filter]")
{
    GRBModel model = makeModel();

    auto I = dsl::range(0, 4);
    auto J = dsl::range(0, 4);

    auto F = (I * J) | dsl::filter([](int i, int j) { return i < j; });

    auto W = dsl::VariableFactory::addIndexed(
        model,
        GRB_CONTINUOUS,
        0.0,
        5.0,
        "W",
        F
    );
    model.update();

    int expectedCount = 0;
    for (auto [i, j] : I * J)
        if (i < j) expectedCount++;

    REQUIRE(W.size() == static_cast<std::size_t>(expectedCount));

    REQUIRE_NOTHROW(W.at(0, 1));
    REQUIRE_NOTHROW(W.at(2, 3));
    REQUIRE(W.try_get(1, 0) == nullptr);   // excluded by filter (i<j)
    REQUIRE(W.try_get(3, 3) == nullptr);   // diagonal excluded

    if (naming_enabled()) {
        REQUIRE(std::string(W.at(0, 1).get(GRB_StringAttr_VarName)) == "W_0_1");
    }
}



/**
 * @test IndexedVariableSet1D::RangeViewDomain
 * @brief Verifies IndexedVariableSet with range_view as domain
 *
 * @scenario Variables are created over a stepped range_view domain
 * @given A range_view(0, 10, 2) producing {0, 2, 4, 6, 8}
 * @when Creating IndexedVariableSet and accessing elements
 * @then Only stepped indices are accessible; others return nullptr
 *
 * @covers VariableFactory::addIndexed() with RangeView
 * @covers range_view with step parameter
 */
TEST_CASE("H4: IndexedVariableSet1D::RangeViewDomain", "[variables][indexed][RangeView]")
{
    GRBModel model = makeModel();

    auto RV = dsl::range_view(0, 10, 2); // {0,2,4,6,8}
    auto Z = dsl::VariableFactory::addIndexed(
        model,
        GRB_CONTINUOUS,
        0.0,
        100.0,
        "Z",
        RV
    );
    model.update();

    REQUIRE(Z.size() == RV.size());

    for (int i : RV) {
        REQUIRE_NOTHROW(Z.at(i));
        REQUIRE(Z.at(i).get(GRB_DoubleAttr_UB) == Catch::Approx(100.0));
    }

    REQUIRE(Z.try_get(1) == nullptr); // not in domain

    if (naming_enabled())
        REQUIRE(std::string(Z.at(2).get(GRB_StringAttr_VarName)) == "Z_2");
}



// ============================================================================
// SECTION I: VARIABLETABLE WITH INDEXEDVARIABLESET INTEGRATION
// ============================================================================

DECLARE_ENUM_WITH_COUNT(VarDom, P, Q, R);

// ============================================================================
// NEW SECTION: UNIFIED VARIABLECONTAINER AND MIXED-MODE VARIABLETABLE
// ============================================================================

/**
 * @test VariableContainer::MixedDenseSparseInSameTable
 * @brief Verifies VariableTable can hold both dense and sparse variables
 *
 * @scenario A single VariableTable stores both VariableGroup and IndexedVariableSet
 * @given A VariableTable with mixed dense and sparse entries
 * @when Storing dense (rectangular) and sparse (filtered) variables in same table
 * @then Both are accessible with unified interface; mode queries work correctly
 *
 * @covers VariableContainer unified storage
 * @covers VariableTable::set() with VariableGroup and IndexedVariableSet
 * @covers VariableTable::isDense() and VariableTable::isSparse()
 */
TEST_CASE("I0: VariableContainer::MixedDenseSparseInSameTable", "[variables][container][mixed]")
{
    GRBModel model = makeModel();

    // Create dense variable (rectangular 2D array)
    auto X_dense = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 3, 4);

    // Create sparse variable (filtered domain: only i < j)
    auto I = dsl::range(0, 5);
    auto J = dsl::range(0, 5);
    auto Y_sparse = dsl::VariableFactory::addIndexed(
        model, GRB_CONTINUOUS, 0, 10, "Y",
        (I * J) | dsl::filter([](int i, int j) { return i < j; })
    );

    // Store BOTH in the same VariableTable!
    dsl::VariableTable<VarDom> vt;
    vt.set(VarDom::P, std::move(X_dense));   // Dense
    vt.set(VarDom::Q, std::move(Y_sparse));  // Sparse

    model.update();

    // Verify mode detection
    REQUIRE(vt.isDense(VarDom::P));
    REQUIRE(!vt.isSparse(VarDom::P));

    REQUIRE(vt.isSparse(VarDom::Q));
    REQUIRE(!vt.isDense(VarDom::Q));

    REQUIRE(vt.isEmpty(VarDom::R));  // Uninitialized entry

    // Unified access works for both
    REQUIRE_NOTHROW(vt.var(VarDom::P, 2, 3));  // Dense access
    REQUIRE_NOTHROW(vt.var(VarDom::Q, 0, 1));  // Sparse access (0 < 1)
    REQUIRE_THROWS(vt.var(VarDom::Q, 1, 0));   // Sparse: filtered out (1 >= 0)

    // Mode-specific access via asGroup() / asIndexed()
    REQUIRE(vt.get(VarDom::P).asGroup().dimension() == 2);
    REQUIRE(vt.get(VarDom::P).asGroup().shape()[0] == 3);
    REQUIRE(vt.get(VarDom::P).asGroup().shape()[1] == 4);

    REQUIRE(vt.get(VarDom::Q).asIndexed().size() == 10);  // 5*4/2 = 10 pairs where i < j

    // Unified count() works for both
    REQUIRE(vt.get(VarDom::P).count() == 12);  // 3 * 4
    REQUIRE(vt.get(VarDom::Q).count() == 10);  // triangular

    // forEach works uniformly
    int denseCount = 0, sparseCount = 0;
    vt.get(VarDom::P).forEach([&](GRBVar&, const std::vector<int>&) { denseCount++; });
    vt.get(VarDom::Q).forEach([&](GRBVar&, const std::vector<int>&) { sparseCount++; });
    REQUIRE(denseCount == 12);
    REQUIRE(sparseCount == 10);
}

/**
 * @test VariableContainer::ScalarInContainer
 * @brief Verifies scalar variables work correctly in VariableContainer
 *
 * @scenario A scalar variable is stored in VariableContainer
 * @given A VariableTable with a scalar entry
 * @when Accessing scalar via container
 * @then isScalar() returns true; scalar() works
 *
 * @covers VariableContainer with scalar GRBVar
 * @covers VariableContainer::isScalar()
 * @covers VariableContainer::scalar()
 */
TEST_CASE("I0b: VariableContainer::ScalarInContainer", "[variables][container][scalar]")
{
    GRBModel model = makeModel();

    auto x = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "x");

    dsl::VariableTable<VarDom> vt;
    vt.set(VarDom::P, x);  // Direct GRBVar assignment
    model.update();

    REQUIRE(vt.isDense(VarDom::P));
    REQUIRE(vt.get(VarDom::P).isScalar());
    REQUIRE_NOTHROW(vt.get(VarDom::P).scalar());
    REQUIRE_NOTHROW(vt.var(VarDom::P));  // Scalar access via var()

    REQUIRE(vt.var(VarDom::P).get(GRB_CharAttr_VType) == GRB_BINARY);
}

/**
 * @test VariableContainer::TryGetUnifiedAccess
 * @brief Verifies try_get() works uniformly for both modes
 *
 * @scenario try_get() is called on both dense and sparse containers
 * @given A VariableTable with mixed entries
 * @when Calling try_get() for valid and invalid indices
 * @then Returns pointer for valid indices, nullptr for invalid
 *
 * @covers VariableContainer::try_get()
 */
TEST_CASE("I0c: VariableContainer::TryGetUnifiedAccess", "[variables][container][try_get]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 3, 3);
    auto Y = dsl::VariableFactory::addIndexed(
        model, GRB_BINARY, 0, 1, "Y",
        dsl::IndexList{0, 2, 4}
    );

    dsl::VariableTable<VarDom> vt;
    vt.set(VarDom::P, std::move(X));
    vt.set(VarDom::Q, std::move(Y));
    model.update();

    // Dense: valid indices return pointer, invalid return nullptr (via exception catch)
    REQUIRE(vt.get(VarDom::P).try_get(1, 1) != nullptr);
    REQUIRE(vt.get(VarDom::P).try_get(5, 5) == nullptr);  // Out of bounds

    // Sparse: valid indices return pointer, missing return nullptr
    REQUIRE(vt.get(VarDom::Q).try_get(0) != nullptr);
    REQUIRE(vt.get(VarDom::Q).try_get(2) != nullptr);
    REQUIRE(vt.get(VarDom::Q).try_get(1) == nullptr);  // Not in domain
    REQUIRE(vt.get(VarDom::Q).try_get(99) == nullptr); // Not in domain
}

/**
 * @test VariableTableIndexed::WorksWithIndexedVariableSet
 * @brief Verifies VariableTable works with IndexedVariableSet
 *
 * @scenario An IndexedVariableSet is stored in a VariableTable
 * @given A filtered IndexedVariableSet (even indices only)
 * @when Storing in VariableTable and accessing via enum key
 * @then Filtered access works; excluded indices throw
 *
 * @covers VariableTable with IndexedVariableSet template parameter
 * @covers VariableTable::set() with IndexedVariableSet
 * @covers VariableTable::var() with indexed access
 */
TEST_CASE("I1: VariableTableIndexed::WorksWithIndexedVariableSet", "[variables][table][indexed][integration]")
{
    GRBModel model = makeModel();

    auto I = dsl::range(0, 4);

    auto XE = dsl::VariableFactory::addIndexed(
        model,
        GRB_CONTINUOUS,
        0.0,
        1.0,
        "XE",
        I | dsl::filter([](int i) { return i % 2 == 0; })  // even only
    );

    // New unified VariableTable - no second template parameter needed!
    dsl::VariableTable<VarDom> vt;
    vt.set(VarDom::P, std::move(XE));

    model.update();

    // Verify it's in sparse mode
    REQUIRE(vt.isSparse(VarDom::P));

    REQUIRE_NOTHROW(vt.var(VarDom::P, 0));
    REQUIRE_NOTHROW(vt.var(VarDom::P, 2));
    REQUIRE_THROWS(vt.var(VarDom::P, 1));  // excluded by filter

    GRBVar& v2 = vt.var(VarDom::P, 2);
    REQUIRE(v2.get(GRB_DoubleAttr_UB) == Catch::Approx(1.0));

    if (naming_enabled())
        REQUIRE(std::string(v2.get(GRB_StringAttr_VarName)) == "XE_2");
}

/**
 * @test VariableTableIndexed::MultipleIndexedSets
 * @brief Verifies VariableTable can hold multiple IndexedVariableSets
 *
 * @scenario Multiple IndexedVariableSets are stored in a VariableTable
 * @given A VariableTable with IndexedVariableSet storage
 * @when Storing multiple sets with different domains
 * @then All sets are accessible and independent
 *
 * @covers VariableTable with multiple IndexedVariableSet entries
 */
TEST_CASE("I2: VariableTableIndexed::MultipleIndexedSets", "[variables][table][indexed]")
{
    GRBModel model = makeModel();

    auto I = dsl::range(0, 3);
    auto J = dsl::range(0, 2);

    auto P = dsl::VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "P", I);
    auto Q = dsl::VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "Q", J);

    // New unified VariableTable
    dsl::VariableTable<VarDom> vt;
    vt.set(VarDom::P, std::move(P));
    vt.set(VarDom::Q, std::move(Q));
    model.update();

    // Use count() which works for both modes
    REQUIRE(vt.get(VarDom::P).count() == 3);
    REQUIRE(vt.get(VarDom::Q).count() == 2);

    REQUIRE_NOTHROW(vt.var(VarDom::P, 2));
    REQUIRE_NOTHROW(vt.var(VarDom::Q, 1));
    REQUIRE_THROWS(vt.var(VarDom::P, 3));  // out of P's domain
    REQUIRE_THROWS(vt.var(VarDom::Q, 2));  // out of Q's domain
}

/**
 * @test IndexedVariableSetIteration::ForEachWorks
 * @brief Verifies forEach iteration on IndexedVariableSet
 *
 * @scenario An IndexedVariableSet is iterated with forEach
 * @given An IndexedVariableSet with filtered domain
 * @when Iterating with forEach
 * @then All variables and their indices are visited
 *
 * @covers IndexedVariableSet::forEach()
 */
TEST_CASE("I3: IndexedVariableSetIteration::ForEachWorks", "[variables][indexed][iteration]")
{
    GRBModel model = makeModel();

    auto I = dsl::range(0, 5);
    auto X = dsl::VariableFactory::addIndexed(
        model,
        GRB_BINARY,
        0, 1,
        "X",
        I | dsl::filter([](int i) { return i % 2 == 0; })  // 0, 2, 4
    );
    model.update();

    int count = 0;
    std::set<int> seen;

    X.forEach([&](GRBVar&, const std::vector<int>& idx) {
        REQUIRE(idx.size() == 1);
        REQUIRE(idx[0] % 2 == 0);  // All should be even
        seen.insert(idx[0]);
        count++;
    });

    REQUIRE(count == 3);
    REQUIRE(seen.count(0) == 1);
    REQUIRE(seen.count(2) == 1);
    REQUIRE(seen.count(4) == 1);
}

/**
 * @test IndexedVariableSetIteration::BeginEndIterators
 * @brief Verifies begin/end iterators on IndexedVariableSet
 *
 * @scenario An IndexedVariableSet is iterated with range-based for
 * @given An IndexedVariableSet
 * @when Using begin()/end() iterators
 * @then All entries are accessible
 *
 * @covers IndexedVariableSet::begin()
 * @covers IndexedVariableSet::end()
 * @covers IndexedVariableSet::Entry structure
 */
TEST_CASE("I4: IndexedVariableSetIteration::BeginEndIterators", "[variables][indexed][iterator]")
{
    GRBModel model = makeModel();

    auto I = dsl::IndexList{1, 3, 5};
    auto X = dsl::VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "X", I);
    model.update();

    int count = 0;
    for (auto& entry : X) {
        REQUIRE(entry.index.size() == 1);
        REQUIRE((entry.index[0] == 1 || entry.index[0] == 3 || entry.index[0] == 5));
        count++;
    }

    REQUIRE(count == 3);
}

// ============================================================================
// SECTION J: EXCEPTION BEHAVIOR AND ERROR CONDITIONS
// ============================================================================

/**
 * @test ExceptionBehavior::VariableGroupOutOfBounds
 * @brief Verifies VariableGroup throws on out-of-bounds access
 *
 * @scenario A VariableGroup is accessed with invalid indices
 * @given A 2D variable array
 * @when Accessing with out-of-range indices
 * @then std::out_of_range is thrown
 *
 * @covers VariableGroup::at() exception behavior
 */
TEST_CASE("J1: ExceptionBehavior::VariableGroupOutOfBounds", "[variables][exception][bounds]")
{
    GRBModel model = makeModel();
    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 3, 4);
    model.update();

    SECTION("Positive out of bounds") {
        REQUIRE_THROWS_AS(X.at(3, 0), std::out_of_range);
        REQUIRE_THROWS_AS(X.at(0, 4), std::out_of_range);
        REQUIRE_THROWS_AS(X.at(10, 10), std::out_of_range);
    }

    SECTION("Negative indices") {
        REQUIRE_THROWS_AS(X.at(-1, 0), std::out_of_range);
        REQUIRE_THROWS_AS(X.at(0, -1), std::out_of_range);
    }
}

/**
 * @test ExceptionBehavior::VariableGroupWrongDimensionCount
 * @brief Verifies VariableGroup throws on wrong number of indices
 *
 * @scenario A VariableGroup is accessed with wrong number of indices
 * @given A 2D variable array
 * @when Accessing with 1 or 3 indices
 * @then std::runtime_error is thrown
 *
 * @covers VariableGroup::at() dimension validation
 */
TEST_CASE("J2: ExceptionBehavior::VariableGroupWrongDimensionCount", "[variables][exception][dimension]")
{
    GRBModel model = makeModel();
    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 3, 4);
    model.update();

    // Wrong number of indices
    REQUIRE_THROWS_AS(X.at(0), std::runtime_error);

    // Vector-based access with wrong size
    std::vector<int> oneIdx = {0};
    std::vector<int> threeIdx = {0, 0, 0};
    REQUIRE_THROWS_AS(X.at(oneIdx), std::runtime_error);
    REQUIRE_THROWS_AS(X.at(threeIdx), std::runtime_error);
}

/**
 * @test ExceptionBehavior::ScalarAccessOnNonScalar
 * @brief Verifies scalar() throws on non-scalar VariableGroup
 *
 * @scenario scalar() is called on a multi-dimensional VariableGroup
 * @given A 1D variable array
 * @when Calling scalar()
 * @then std::runtime_error is thrown
 *
 * @covers VariableGroup::scalar() exception behavior
 */
TEST_CASE("J3: ExceptionBehavior::ScalarAccessOnNonScalar", "[variables][exception][scalar]")
{
    GRBModel model = makeModel();
    auto V = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "V", 5);
    model.update();

    REQUIRE_THROWS_AS(V.scalar(), std::runtime_error);
}

/**
 * @test ExceptionBehavior::IndexedVariableSetNotFound
 * @brief Verifies IndexedVariableSet throws on missing index
 *
 * @scenario at() is called with non-existent index
 * @given An IndexedVariableSet with sparse domain
 * @when Accessing with index not in domain
 * @then std::out_of_range is thrown
 *
 * @covers IndexedVariableSet::at() exception behavior
 */
TEST_CASE("J4: ExceptionBehavior::IndexedVariableSetNotFound", "[variables][exception][indexed]")
{
    GRBModel model = makeModel();

    auto I = dsl::IndexList{0, 2, 4};  // Sparse domain
    auto X = dsl::VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "X", I);
    model.update();

    REQUIRE_THROWS_AS(X.at(1), std::out_of_range);
    REQUIRE_THROWS_AS(X.at(3), std::out_of_range);
    REQUIRE_THROWS_AS(X.at(99), std::out_of_range);

    // try_get should return nullptr instead
    REQUIRE(X.try_get(1) == nullptr);
    REQUIRE(X.try_get(3) == nullptr);
    REQUIRE(X.try_get(99) == nullptr);
}

/**
 * @test ExceptionBehavior::IndexedVariableSetVectorAccess
 * @brief Verifies IndexedVariableSet vector access throws on missing index
 *
 * @scenario at(vector) is called with non-existent index
 * @given A 2D IndexedVariableSet
 * @when Accessing with vector index not in domain
 * @then std::runtime_error is thrown
 *
 * @covers IndexedVariableSet::at(vector) exception behavior
 */
TEST_CASE("J5: ExceptionBehavior::IndexedVariableSetVectorAccess", "[variables][exception][indexed]")
{
    GRBModel model = makeModel();

    auto I = dsl::range(0, 2);
    auto J = dsl::range(0, 2);
    auto X = dsl::VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "X", I * J);
    model.update();

    std::vector<int> valid = {0, 1};
    std::vector<int> invalid = {5, 5};

    REQUIRE_NOTHROW(X.at(valid));
    REQUIRE_THROWS_AS(X.at(invalid), std::runtime_error);

    REQUIRE(X.try_get(valid) != nullptr);
    REQUIRE(X.try_get(invalid) == nullptr);
}

// ============================================================================
// SECTION K: VARIABLE TYPE COVERAGE (BINARY, CONTINUOUS, INTEGER)
// ============================================================================

/**
 * @test TypeCoverage::BinaryVariables
 * @brief Verifies binary variables work correctly in all contexts
 *
 * @scenario Binary variables are created in various dimensions
 * @given GRBModel and VariableFactory
 * @when Creating scalar, 1D, and 2D binary variables
 * @then All have correct type and bounds
 *
 * @covers GRB_BINARY type in VariableFactory
 */
TEST_CASE("K1: TypeCoverage::BinaryVariables", "[variables][types][binary]")
{
    GRBModel model = makeModel();

    SECTION("Scalar binary") {
        auto x = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "x");
        model.update();
        REQUIRE(x.get(GRB_CharAttr_VType) == GRB_BINARY);
    }

    SECTION("1D binary array") {
        auto V = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "V", 5);
        model.update();
        V.forEach([](GRBVar& v, const std::vector<int>&) {
            REQUIRE(v.get(GRB_CharAttr_VType) == GRB_BINARY);
        });
    }

    SECTION("2D binary matrix") {
        auto M = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "M", 3, 3);
        model.update();
        M.forEach([](GRBVar& v, const std::vector<int>&) {
            REQUIRE(v.get(GRB_CharAttr_VType) == GRB_BINARY);
        });
    }

    SECTION("Indexed binary") {
        auto I = dsl::range(0, 3);
        auto X = dsl::VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "X", I);
        model.update();
        X.forEach([](GRBVar& v, const std::vector<int>&) {
            REQUIRE(v.get(GRB_CharAttr_VType) == GRB_BINARY);
        });
    }
}

/**
 * @test TypeCoverage::ContinuousVariables
 * @brief Verifies continuous variables work correctly in all contexts
 *
 * @scenario Continuous variables are created in various dimensions
 * @given GRBModel and VariableFactory
 * @when Creating scalar, 1D, and 2D continuous variables
 * @then All have correct type and bounds
 *
 * @covers GRB_CONTINUOUS type in VariableFactory
 */
TEST_CASE("K2: TypeCoverage::ContinuousVariables", "[variables][types][continuous]")
{
    GRBModel model = makeModel();

    SECTION("Scalar continuous") {
        auto x = dsl::VariableFactory::add(model, GRB_CONTINUOUS, -100.5, 200.5, "x");
        model.update();
        REQUIRE(x.get(GRB_CharAttr_VType) == GRB_CONTINUOUS);
        REQUIRE(x.get(GRB_DoubleAttr_LB) == Catch::Approx(-100.5));
        REQUIRE(x.get(GRB_DoubleAttr_UB) == Catch::Approx(200.5));
    }

    SECTION("1D continuous array") {
        auto V = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 1000, "V", 5);
        model.update();
        V.forEach([](GRBVar& v, const std::vector<int>&) {
            REQUIRE(v.get(GRB_CharAttr_VType) == GRB_CONTINUOUS);
            REQUIRE(v.get(GRB_DoubleAttr_UB) == Catch::Approx(1000.0));
        });
    }

    SECTION("Indexed continuous") {
        auto I = dsl::range(0, 3);
        auto J = dsl::range(0, 2);
        auto X = dsl::VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 50, "X", I * J);
        model.update();
        X.forEach([](GRBVar& v, const std::vector<int>&) {
            REQUIRE(v.get(GRB_CharAttr_VType) == GRB_CONTINUOUS);
        });
    }
}

/**
 * @test TypeCoverage::IntegerVariables
 * @brief Verifies integer variables work correctly in all contexts
 *
 * @scenario Integer variables are created in various dimensions
 * @given GRBModel and VariableFactory
 * @when Creating scalar, 1D, and indexed integer variables
 * @then All have correct type and bounds
 *
 * @covers GRB_INTEGER type in VariableFactory
 */
TEST_CASE("K3: TypeCoverage::IntegerVariables", "[variables][types][integer]")
{
    GRBModel model = makeModel();

    SECTION("Scalar integer") {
        auto x = dsl::VariableFactory::add(model, GRB_INTEGER, 0, 100, "x");
        model.update();
        REQUIRE(x.get(GRB_CharAttr_VType) == GRB_INTEGER);
    }

    SECTION("1D integer array") {
        auto V = dsl::VariableFactory::add(model, GRB_INTEGER, -50, 50, "V", 5);
        model.update();
        V.forEach([](GRBVar& v, const std::vector<int>&) {
            REQUIRE(v.get(GRB_CharAttr_VType) == GRB_INTEGER);
            REQUIRE(v.get(GRB_DoubleAttr_LB) == Catch::Approx(-50.0));
            REQUIRE(v.get(GRB_DoubleAttr_UB) == Catch::Approx(50.0));
        });
    }

    SECTION("Indexed integer with filter") {
        auto I = dsl::range(0, 10);
        auto X = dsl::VariableFactory::addIndexed(
            model, GRB_INTEGER, 0, 999, "X",
            I | dsl::filter([](int i) { return i % 3 == 0; })
        );
        model.update();

        REQUIRE(X.size() == 4);  // 0, 3, 6, 9
        X.forEach([](GRBVar& v, const std::vector<int>&) {
            REQUIRE(v.get(GRB_CharAttr_VType) == GRB_INTEGER);
        });
    }
}

/**
 * @test TypeCoverage::MixedTypesInTable
 * @brief Verifies VariableTable can hold mixed variable types
 *
 * @scenario A VariableTable stores variables of different types
 * @given A VariableTable with multiple entries
 * @when Storing BINARY, CONTINUOUS, and INTEGER variables
 * @then Each retains its correct type
 *
 * @covers VariableTable with mixed GRB types
 */
TEST_CASE("K4: TypeCoverage::MixedTypesInTable", "[variables][types][table]")
{
    GRBModel model = makeModel();
    dsl::VariableTable<LargeVars> vt;

    auto binary = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "bin", 2);
    auto continuous = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "cont", 2);
    auto integer = dsl::VariableFactory::add(model, GRB_INTEGER, 0, 100, "int", 2);

    vt.set(LargeVars::V0, std::move(binary));
    vt.set(LargeVars::V1, std::move(continuous));
    vt.set(LargeVars::V2, std::move(integer));
    model.update();

    REQUIRE(vt.var(LargeVars::V0, 0).get(GRB_CharAttr_VType) == GRB_BINARY);
    REQUIRE(vt.var(LargeVars::V1, 0).get(GRB_CharAttr_VType) == GRB_CONTINUOUS);
    REQUIRE(vt.var(LargeVars::V2, 0).get(GRB_CharAttr_VType) == GRB_INTEGER);
}

// ============================================================================
// SECTION L: VARIABLE MODIFICATION UTILITIES
// ============================================================================

/**
 * @test VariableModification::FixSingleVariable
 * @brief Verifies fix() sets both LB and UB to the specified value
 *
 * @scenario A variable is fixed to a specific value
 * @given A continuous variable with bounds [0, 10]
 * @when Calling dsl::fix(v, 5.0)
 * @then Both LB and UB become 5.0
 *
 * @covers dsl::fix()
 */
TEST_CASE("L1: VariableModification::FixSingleVariable", "[variables][modification][fix]")
{
    GRBModel model = makeModel();
    auto x = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "x");
    model.update();

    REQUIRE(x.get(GRB_DoubleAttr_LB) == Catch::Approx(0.0));
    REQUIRE(x.get(GRB_DoubleAttr_UB) == Catch::Approx(10.0));

    dsl::fix(x, 5.0);
    model.update();

    REQUIRE(x.get(GRB_DoubleAttr_LB) == Catch::Approx(5.0));
    REQUIRE(x.get(GRB_DoubleAttr_UB) == Catch::Approx(5.0));
}

/**
 * @test VariableModification::UnfixRestoresBounds
 * @brief Verifies unfix() restores variable bounds
 *
 * @scenario A fixed variable is unfixed with new bounds
 * @given A variable fixed to 5.0
 * @when Calling dsl::unfix(v, 0.0, 10.0)
 * @then Bounds are restored to [0, 10]
 *
 * @covers dsl::unfix()
 */
TEST_CASE("L2: VariableModification::UnfixRestoresBounds", "[variables][modification][unfix]")
{
    GRBModel model = makeModel();
    auto x = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "x");
    model.update();

    dsl::fix(x, 5.0);
    model.update();

    REQUIRE(x.get(GRB_DoubleAttr_LB) == Catch::Approx(5.0));
    REQUIRE(x.get(GRB_DoubleAttr_UB) == Catch::Approx(5.0));

    dsl::unfix(x, 0.0, 10.0);
    model.update();

    REQUIRE(x.get(GRB_DoubleAttr_LB) == Catch::Approx(0.0));
    REQUIRE(x.get(GRB_DoubleAttr_UB) == Catch::Approx(10.0));
}

/**
 * @test VariableModification::UnfixInvalidBoundsThrows
 * @brief Verifies unfix() throws when lb > ub
 *
 * @scenario unfix() is called with invalid bounds
 * @given A variable
 * @when Calling dsl::unfix(v, 10.0, 5.0) where lb > ub
 * @then std::invalid_argument is thrown
 *
 * @covers dsl::unfix() exception behavior
 */
TEST_CASE("L3: VariableModification::UnfixInvalidBoundsThrows", "[variables][modification][exception]")
{
    GRBModel model = makeModel();
    auto x = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "x");
    model.update();

    REQUIRE_THROWS_AS(dsl::unfix(x, 10.0, 5.0), std::invalid_argument);
}

/**
 * @test VariableModification::SetStartSingleVariable
 * @brief Verifies setStart() sets the MIP start attribute
 *
 * @scenario A MIP start hint is provided for a variable
 * @given A binary variable
 * @when Calling dsl::setStart(v, 1.0)
 * @then The Start attribute is set to 1.0
 *
 * @covers dsl::setStart()
 */
TEST_CASE("L4: VariableModification::SetStartSingleVariable", "[variables][modification][start]")
{
    GRBModel model = makeModel();
    auto x = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "x");
    model.update();

    dsl::setStart(x, 1.0);
    model.update();

    REQUIRE(x.get(GRB_DoubleAttr_Start) == Catch::Approx(1.0));
}

/**
 * @test VariableModification::ClearStartSetsUndefined
 * @brief Verifies clearStart() clears the MIP start value
 *
 * @scenario A MIP start hint is cleared
 * @given A variable with a start value set
 * @when Calling dsl::clearStart(v)
 * @then The Start attribute becomes GRB_UNDEFINED
 *
 * @covers dsl::clearStart()
 */
TEST_CASE("L5: VariableModification::ClearStartSetsUndefined", "[variables][modification][start]")
{
    GRBModel model = makeModel();
    auto x = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "x");
    model.update();

    dsl::setStart(x, 1.0);
    model.update();
    REQUIRE(x.get(GRB_DoubleAttr_Start) == Catch::Approx(1.0));

    dsl::clearStart(x);
    model.update();
    REQUIRE(x.get(GRB_DoubleAttr_Start) == GRB_UNDEFINED);
}

/**
 * @test VariableModification::LbUbAccessors
 * @brief Verifies lb() and ub() return correct bounds
 *
 * @scenario Variable bounds are queried using accessor functions
 * @given A variable with bounds [-5, 15]
 * @when Calling dsl::lb(v) and dsl::ub(v)
 * @then Correct bounds are returned
 *
 * @covers dsl::lb()
 * @covers dsl::ub()
 */
TEST_CASE("L6: VariableModification::LbUbAccessors", "[variables][modification][bounds]")
{
    GRBModel model = makeModel();
    auto x = dsl::VariableFactory::add(model, GRB_CONTINUOUS, -5.0, 15.0, "x");
    model.update();

    REQUIRE(dsl::lb(x) == Catch::Approx(-5.0));
    REQUIRE(dsl::ub(x) == Catch::Approx(15.0));
}

/**
 * @test VariableModification::SetLbSetUbModifiers
 * @brief Verifies setLB() and setUB() modify bounds correctly
 *
 * @scenario Variable bounds are modified individually
 * @given A variable with bounds [0, 10]
 * @when Calling dsl::setLB(v, 2.0) and dsl::setUB(v, 8.0)
 * @then Bounds become [2, 8]
 *
 * @covers dsl::setLB()
 * @covers dsl::setUB()
 */
TEST_CASE("L7: VariableModification::SetLbSetUbModifiers", "[variables][modification][bounds]")
{
    GRBModel model = makeModel();
    auto x = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "x");
    model.update();

    dsl::setLB(x, 2.0);
    dsl::setUB(x, 8.0);
    model.update();

    REQUIRE(dsl::lb(x) == Catch::Approx(2.0));
    REQUIRE(dsl::ub(x) == Catch::Approx(8.0));
}

/**
 * @test VariableModification::FixAllVariableGroup
 * @brief Verifies fixAll() fixes all variables in a VariableGroup
 *
 * @scenario All variables in a 1D array are fixed
 * @given A 1D variable array with 3 elements
 * @when Calling dsl::fixAll(vg, {1.0, 2.0, 3.0})
 * @then All variables are fixed to their respective values
 *
 * @covers dsl::fixAll(VariableGroup&, vector)
 */
TEST_CASE("L8: VariableModification::FixAllVariableGroup", "[variables][modification][fixAll]")
{
    GRBModel model = makeModel();
    auto V = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "V", 3);
    model.update();

    std::vector<double> fixValues = {1.0, 2.0, 3.0};
    dsl::fixAll(V, fixValues);
    model.update();

    REQUIRE(V.at(0).get(GRB_DoubleAttr_LB) == Catch::Approx(1.0));
    REQUIRE(V.at(0).get(GRB_DoubleAttr_UB) == Catch::Approx(1.0));
    REQUIRE(V.at(1).get(GRB_DoubleAttr_LB) == Catch::Approx(2.0));
    REQUIRE(V.at(1).get(GRB_DoubleAttr_UB) == Catch::Approx(2.0));
    REQUIRE(V.at(2).get(GRB_DoubleAttr_LB) == Catch::Approx(3.0));
    REQUIRE(V.at(2).get(GRB_DoubleAttr_UB) == Catch::Approx(3.0));
}

/**
 * @test VariableModification::FixAllWrongSizeThrows
 * @brief Verifies fixAll() throws when value count mismatches
 *
 * @scenario fixAll() is called with wrong number of values
 * @given A 1D variable array with 3 elements
 * @when Calling fixAll with 2 or 4 values
 * @then std::invalid_argument is thrown
 *
 * @covers dsl::fixAll() exception behavior
 */
TEST_CASE("L9: VariableModification::FixAllWrongSizeThrows", "[variables][modification][exception]")
{
    GRBModel model = makeModel();
    auto V = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "V", 3);
    model.update();

    std::vector<double> tooFew = {1.0, 2.0};
    std::vector<double> tooMany = {1.0, 2.0, 3.0, 4.0};

    REQUIRE_THROWS_AS(dsl::fixAll(V, tooFew), std::invalid_argument);
    REQUIRE_THROWS_AS(dsl::fixAll(V, tooMany), std::invalid_argument);
}

/**
 * @test VariableModification::SetStartAllVariableGroup
 * @brief Verifies setStartAll() sets start values for all variables
 *
 * @scenario MIP start hints are provided for a variable group
 * @given A 1D binary variable array
 * @when Calling dsl::setStartAll(vg, values)
 * @then All Start attributes are set
 *
 * @covers dsl::setStartAll(VariableGroup&, vector)
 */
TEST_CASE("L10: VariableModification::SetStartAllVariableGroup", "[variables][modification][startAll]")
{
    GRBModel model = makeModel();
    auto V = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "V", 3);
    model.update();

    std::vector<double> startValues = {1.0, 0.0, 1.0};
    dsl::setStartAll(V, startValues);
    model.update();

    REQUIRE(V.at(0).get(GRB_DoubleAttr_Start) == Catch::Approx(1.0));
    REQUIRE(V.at(1).get(GRB_DoubleAttr_Start) == Catch::Approx(0.0));
    REQUIRE(V.at(2).get(GRB_DoubleAttr_Start) == Catch::Approx(1.0));
}

/**
 * @test VariableModification::FixAllIndexedVariableSet
 * @brief Verifies fixAll() works with IndexedVariableSet
 *
 * @scenario All variables in an IndexedVariableSet are fixed
 * @given An IndexedVariableSet over domain {0, 2, 4}
 * @when Calling dsl::fixAll(vs, values)
 * @then All variables are fixed in storage order
 *
 * @covers dsl::fixAll(IndexedVariableSet&, vector)
 */
TEST_CASE("L11: VariableModification::FixAllIndexedVariableSet", "[variables][modification][indexed]")
{
    GRBModel model = makeModel();
    auto I = dsl::IndexList{0, 2, 4};
    auto X = dsl::VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 10, "X", I);
    model.update();

    std::vector<double> fixValues = {1.0, 2.0, 3.0};
    dsl::fixAll(X, fixValues);
    model.update();

    REQUIRE(X.at(0).get(GRB_DoubleAttr_LB) == Catch::Approx(1.0));
    REQUIRE(X.at(2).get(GRB_DoubleAttr_LB) == Catch::Approx(2.0));
    REQUIRE(X.at(4).get(GRB_DoubleAttr_LB) == Catch::Approx(3.0));
}

/**
 * @test VariableModification::SetStartAllIndexedVariableSet
 * @brief Verifies setStartAll() works with IndexedVariableSet
 *
 * @scenario MIP start hints are provided for an IndexedVariableSet
 * @given An IndexedVariableSet
 * @when Calling dsl::setStartAll(vs, values)
 * @then All Start attributes are set in storage order
 *
 * @covers dsl::setStartAll(IndexedVariableSet&, vector)
 */
TEST_CASE("L12: VariableModification::SetStartAllIndexedVariableSet", "[variables][modification][indexed]")
{
    GRBModel model = makeModel();
    auto I = dsl::range(0, 3);
    auto X = dsl::VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "X", I);
    model.update();

    std::vector<double> startValues = {0.0, 1.0, 0.0};
    dsl::setStartAll(X, startValues);
    model.update();

    REQUIRE(X.at(0).get(GRB_DoubleAttr_Start) == Catch::Approx(0.0));
    REQUIRE(X.at(1).get(GRB_DoubleAttr_Start) == Catch::Approx(1.0));
    REQUIRE(X.at(2).get(GRB_DoubleAttr_Start) == Catch::Approx(0.0));
}

// ============================================================================
// SECTION M: SOLUTION EXTRACTION UTILITIES
// ============================================================================

/**
 * @test SolutionExtraction::ValueSingleVariable
 * @brief Verifies value() extracts solution from a single variable
 *
 * @scenario A trivial LP is solved and solution extracted
 * @given A single continuous variable with objective
 * @when Calling dsl::value(v) after optimization
 * @then The optimal value is returned
 *
 * @covers dsl::value()
 */
TEST_CASE("M1: SolutionExtraction::ValueSingleVariable", "[variables][solution][value]")
{
    GRBModel model = makeModel();
    auto x = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "x");

    // Maximize x subject to x <= 5
    model.setObjective(GRBLinExpr(x), GRB_MAXIMIZE);
    model.addConstr(x <= 5);
    model.optimize();

    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
    REQUIRE(dsl::value(x) == Catch::Approx(5.0));
}

/**
 * @test SolutionExtraction::ValuesVariableGroup
 * @brief Verifies values() extracts all solution values from VariableGroup
 *
 * @scenario A multi-variable LP is solved
 * @given A 1D variable array with constraints
 * @when Calling dsl::values(vg) after optimization
 * @then Vector of optimal values is returned in forEach order
 *
 * @covers dsl::values(VariableGroup)
 */
TEST_CASE("M2: SolutionExtraction::ValuesVariableGroup", "[variables][solution][values]")
{
    GRBModel model = makeModel();
    auto V = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "V", 3);

    // Fix each variable to a known value via constraints
    model.addConstr(V.at(0) == 1.0);
    model.addConstr(V.at(1) == 2.0);
    model.addConstr(V.at(2) == 3.0);
    model.setObjective(GRBLinExpr(0.0), GRB_MINIMIZE);  // Feasibility only
    model.optimize();

    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);

    auto vals = dsl::values(V);
    REQUIRE(vals.size() == 3);
    REQUIRE(vals[0] == Catch::Approx(1.0));
    REQUIRE(vals[1] == Catch::Approx(2.0));
    REQUIRE(vals[2] == Catch::Approx(3.0));
}

/**
 * @test SolutionExtraction::ValuesIndexedVariableSet
 * @brief Verifies values() extracts solutions from IndexedVariableSet
 *
 * @scenario An indexed variable set LP is solved
 * @given An IndexedVariableSet with constraints
 * @when Calling dsl::values(vs) after optimization
 * @then Vector of optimal values is returned in storage order
 *
 * @covers dsl::values(IndexedVariableSet)
 */
TEST_CASE("M3: SolutionExtraction::ValuesIndexedVariableSet", "[variables][solution][indexed]")
{
    GRBModel model = makeModel();
    auto I = dsl::IndexList{0, 2, 4};
    auto X = dsl::VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 10, "X", I);

    model.addConstr(X.at(0) == 5.0);
    model.addConstr(X.at(2) == 6.0);
    model.addConstr(X.at(4) == 7.0);
    model.setObjective(GRBLinExpr(0.0), GRB_MINIMIZE);
    model.optimize();

    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);

    auto vals = dsl::values(X);
    REQUIRE(vals.size() == 3);
    REQUIRE(vals[0] == Catch::Approx(5.0));
    REQUIRE(vals[1] == Catch::Approx(6.0));
    REQUIRE(vals[2] == Catch::Approx(7.0));
}

/**
 * @test SolutionExtraction::ValueAtVariableGroup
 * @brief Verifies valueAt() extracts solution at specific indices
 *
 * @scenario A 2D variable array LP is solved
 * @given A 2D variable array with constraints
 * @when Calling dsl::valueAt(vg, i, j) after optimization
 * @then Correct value at indices is returned
 *
 * @covers dsl::valueAt(VariableGroup, ...)
 */
TEST_CASE("M4: SolutionExtraction::ValueAtVariableGroup", "[variables][solution][valueAt]")
{
    GRBModel model = makeModel();
    auto M = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 100, "M", 2, 2);

    model.addConstr(M.at(0, 0) == 10.0);
    model.addConstr(M.at(0, 1) == 20.0);
    model.addConstr(M.at(1, 0) == 30.0);
    model.addConstr(M.at(1, 1) == 40.0);
    model.setObjective(GRBLinExpr(0.0), GRB_MINIMIZE);
    model.optimize();

    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);

    REQUIRE(dsl::valueAt(M, 0, 0) == Catch::Approx(10.0));
    REQUIRE(dsl::valueAt(M, 0, 1) == Catch::Approx(20.0));
    REQUIRE(dsl::valueAt(M, 1, 0) == Catch::Approx(30.0));
    REQUIRE(dsl::valueAt(M, 1, 1) == Catch::Approx(40.0));
}

/**
 * @test SolutionExtraction::ValueAtIndexedVariableSet
 * @brief Verifies valueAt() works with IndexedVariableSet
 *
 * @scenario An indexed variable set LP is solved
 * @given A 2D IndexedVariableSet
 * @when Calling dsl::valueAt(vs, i, j) after optimization
 * @then Correct value at indices is returned
 *
 * @covers dsl::valueAt(IndexedVariableSet, ...)
 */
TEST_CASE("M5: SolutionExtraction::ValueAtIndexedVariableSet", "[variables][solution][valueAt]")
{
    GRBModel model = makeModel();
    auto I = dsl::range(0, 2);
    auto J = dsl::range(0, 2);
    auto X = dsl::VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 100, "X", I * J);

    model.addConstr(X.at(0, 0) == 1.0);
    model.addConstr(X.at(0, 1) == 2.0);
    model.addConstr(X.at(1, 0) == 3.0);
    model.addConstr(X.at(1, 1) == 4.0);
    model.setObjective(GRBLinExpr(0.0), GRB_MINIMIZE);
    model.optimize();

    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);

    REQUIRE(dsl::valueAt(X, 0, 0) == Catch::Approx(1.0));
    REQUIRE(dsl::valueAt(X, 0, 1) == Catch::Approx(2.0));
    REQUIRE(dsl::valueAt(X, 1, 0) == Catch::Approx(3.0));
    REQUIRE(dsl::valueAt(X, 1, 1) == Catch::Approx(4.0));
}

/**
 * @test SolutionExtraction::ValuesWithIndexVariableGroup
 * @brief Verifies valuesWithIndex() returns index-value pairs
 *
 * @scenario A 2D variable array LP is solved
 * @given A 2D variable array
 * @when Calling dsl::valuesWithIndex(vg) after optimization
 * @then Vector of (index, value) pairs is returned
 *
 * @covers dsl::valuesWithIndex(VariableGroup)
 */
TEST_CASE("M6: SolutionExtraction::ValuesWithIndexVariableGroup", "[variables][solution][valuesWithIndex]")
{
    GRBModel model = makeModel();
    auto M = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 100, "M", 2, 2);

    model.addConstr(M.at(0, 0) == 1.0);
    model.addConstr(M.at(0, 1) == 2.0);
    model.addConstr(M.at(1, 0) == 3.0);
    model.addConstr(M.at(1, 1) == 4.0);
    model.setObjective(GRBLinExpr(0.0), GRB_MINIMIZE);
    model.optimize();

    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);

    auto pairs = dsl::valuesWithIndex(M);
    REQUIRE(pairs.size() == 4);

    // Verify we can find all expected index-value pairs
    std::map<std::vector<int>, double> resultMap;
    for (const auto& [idx, val] : pairs) {
        resultMap[idx] = val;
    }

    REQUIRE(resultMap[{0, 0}] == Catch::Approx(1.0));
    REQUIRE(resultMap[{0, 1}] == Catch::Approx(2.0));
    REQUIRE(resultMap[{1, 0}] == Catch::Approx(3.0));
    REQUIRE(resultMap[{1, 1}] == Catch::Approx(4.0));
}

/**
 * @test SolutionExtraction::ValuesWithIndexIndexedVariableSet
 * @brief Verifies valuesWithIndex() works with IndexedVariableSet
 *
 * @scenario An IndexedVariableSet LP is solved
 * @given A filtered IndexedVariableSet
 * @when Calling dsl::valuesWithIndex(vs) after optimization
 * @then Vector of (index, value) pairs for valid indices is returned
 *
 * @covers dsl::valuesWithIndex(IndexedVariableSet)
 */
TEST_CASE("M7: SolutionExtraction::ValuesWithIndexIndexedVariableSet", "[variables][solution][valuesWithIndex]")
{
    GRBModel model = makeModel();
    auto I = dsl::IndexList{1, 3, 5};
    auto X = dsl::VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 100, "X", I);

    model.addConstr(X.at(1) == 10.0);
    model.addConstr(X.at(3) == 30.0);
    model.addConstr(X.at(5) == 50.0);
    model.setObjective(GRBLinExpr(0.0), GRB_MINIMIZE);
    model.optimize();

    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);

    auto pairs = dsl::valuesWithIndex(X);
    REQUIRE(pairs.size() == 3);

    // Verify index-value pairs
    std::map<std::vector<int>, double> resultMap;
    for (const auto& [idx, val] : pairs) {
        resultMap[idx] = val;
    }

    REQUIRE(resultMap[{1}] == Catch::Approx(10.0));
    REQUIRE(resultMap[{3}] == Catch::Approx(30.0));
    REQUIRE(resultMap[{5}] == Catch::Approx(50.0));
}

/**
 * @test SolutionExtraction::RoundTripFixAllValues
 * @brief Verifies values() and fixAll() work together for round-trip
 *
 * @scenario Solution is extracted, model modified, then fixed to previous solution
 * @given A solved LP
 * @when Extracting values, then using fixAll to restore them
 * @then Variables are fixed to their original solution values
 *
 * @covers Round-trip: dsl::values() + dsl::fixAll()
 */
TEST_CASE("M8: SolutionExtraction::RoundTripFixAllValues", "[variables][solution][roundtrip]")
{
    GRBModel model = makeModel();
    auto V = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "V", 3);

    // First solve
    model.addConstr(V.at(0) == 1.0);
    model.addConstr(V.at(1) == 2.0);
    model.addConstr(V.at(2) == 3.0);
    model.setObjective(GRBLinExpr(0.0), GRB_MINIMIZE);
    model.optimize();

    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);

    // Extract solution
    auto solution = dsl::values(V);
    REQUIRE(solution.size() == 3);

    // Fix all to extracted values
    dsl::fixAll(V, solution);
    model.update();

    // Verify bounds match solution
    REQUIRE(dsl::lb(V.at(0)) == Catch::Approx(1.0));
    REQUIRE(dsl::ub(V.at(0)) == Catch::Approx(1.0));
    REQUIRE(dsl::lb(V.at(1)) == Catch::Approx(2.0));
    REQUIRE(dsl::ub(V.at(1)) == Catch::Approx(2.0));
    REQUIRE(dsl::lb(V.at(2)) == Catch::Approx(3.0));
    REQUIRE(dsl::ub(V.at(2)) == Catch::Approx(3.0));
}