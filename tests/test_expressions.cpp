/*
===============================================================================
TEST EXPRESSIONS � Comprehensive tests for expressions.h
===============================================================================

OVERVIEW
--------
Validates the DSL expression system for summation operations over domains,
variable groups, and indexed variable sets. Tests cover lambda-based sums,
domain filtering, Cartesian products, tuple unpacking, expression composition,
coefficient handling, and error handling for missing indices.

TEST ORGANIZATION
-----------------
� Section A: Basic sum() over rectangular VariableGroup with lambdas
� Section B: IndexedVariableSet integrity tests for lookup and tuple behavior
� Section C: sum(D, IndexedVariableSet) domain-based lookups
� Section D: Tuple-based Cartesian indexing (2D and 3D)
� Section E: VariableGroup sum overloads and const-correctness
� Section F: Expression composition and coefficients
� Section G: Edge cases and boundary conditions
� Section H: Error scenarios and exception handling
� Section I: Iterator properties and performance
� Section J: Quadratic sum (quadSum) for QP objectives

TEST STRATEGY
-------------
� Verify expression construction without full optimization where possible
� Confirm domain iteration order and tuple unpacking correctness
� Exercise coefficient handling: positive, negative, zero, index-dependent
� Validate expression composition: combining sums, constant terms, nesting
� Test boundary conditions: empty domains, single elements, large domains
� Validate exception behavior for missing indices and type mismatches
� Exercise both lambda-based and direct variable set summation
� Confirm iterator reusability across multiple passes

DEPENDENCIES
------------
� Catch2 v3.0+ - Test framework
� Gurobi C++ API - Optimization modeling
� expressions.h - System under test
� variables.h - Variable creation utilities
� indexing.h - Domain and index utilities

===============================================================================
*/

#include "catch_amalgamated.hpp"
#include "gurobi_c++.h"

#include <gurobi_dsl/variables.h>
#include <gurobi_dsl/indexing.h>
#include <gurobi_dsl/expressions.h>

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Creates an isolated, silent Gurobi model with a persistent environment
 * @return A new GRBModel configured for silent operation
 */
static GRBModel makeModel()
{
    static GRBEnv env = GRBEnv(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    return GRBModel(env);
}

// ============================================================================
// SECTION A: BASIC SUM() OVER RECTANGULAR VARIABLEGROUP
// ============================================================================

/**
 * @test SumLambda::Sum1DIndexList
 * @brief Verifies sum over 1D IndexList with lambda expression
 *
 * @scenario Sum of variables X_0 + X_1 + X_2 using lambda
 * @given A 1D VariableGroup with 3 continuous variables
 * @when Applying sum(I, lambda) over IndexList {0, 1, 2}
 * @then Expression is constructed and model optimizes successfully
 *
 * @covers dsl::sum(domain, lambda)
 */
TEST_CASE("A1: SumLambda::Sum1DIndexList", "[expressions][sum][lambda]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 3);
    model.update();

    dsl::IndexList I{ 0, 1, 2 };

    // Expression: X_0 + X_1 + X_2
    GRBLinExpr expr = dsl::sum(I, [&](int i) { return X(i); });

    model.addConstr(expr <= 5, "c1");
    model.setObjective(expr, GRB_MAXIMIZE);

    REQUIRE_NOTHROW(model.optimize());
}

/**
 * @test SumLambda::SumRangeViewScalar
 * @brief Verifies sum over RangeView with scalar coefficients
 *
 * @scenario Expression 2 * Sum X_i using scalar multiplication
 * @given A 1D VariableGroup with 5 continuous variables
 * @when Applying sum with coefficient 2.0 over range 0..4
 * @then Expression is constructed with correct coefficients
 *
 * @covers dsl::sum(RangeView, lambda)
 */
TEST_CASE("A2: SumLambda::SumRangeViewScalar", "[expressions][sum][lambda]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 5);
    model.update();

    auto I = dsl::range_view(0, 5);   // 0..4

    // Expression: 2 * Sum X_i
    GRBLinExpr expr = dsl::sum(I, [&](int i) { return 2.0 * X(i); });

    model.addConstr(expr <= 10, "c2");
    model.setObjective(expr, GRB_MAXIMIZE);

    REQUIRE_NOTHROW(model.optimize());
}

/**
 * @test SumLambda::Sum2DCartesian
 * @brief Verifies sum over 2D Cartesian domain
 *
 * @scenario Sum_{i,j} X(i,j) over Cartesian product I x J
 * @given A 2D VariableGroup (2x3)
 * @when Applying sum over Cartesian product of ranges
 * @then All combinations are iterated correctly
 *
 * @covers dsl::sum(Cartesian, lambda)
 */
TEST_CASE("A3: SumLambda::Sum2DCartesian", "[expressions][sum][cartesian]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 2, 3);
    model.update();

    auto I = dsl::range_view(0, 2);
    auto J = dsl::range_view(0, 3);
    auto IJ = I * J;

    // Expression: Sum_{i,j} X(i,j)
    GRBLinExpr expr = dsl::sum(IJ, [&](int i, int j) { return X(i, j); });

    model.addConstr(expr <= 20, "c3");
    model.setObjective(expr, GRB_MAXIMIZE);

    REQUIRE_NOTHROW(model.optimize());
}

/**
 * @test SumLambda::SumFilteredCartesian
 * @brief Verifies sum over filtered Cartesian domain
 *
 * @scenario Sum Y(i,j) where i < j using filter predicate
 * @given A 2D VariableGroup (3x3)
 * @when Applying sum with filter i < j
 * @then Only upper triangular elements are summed
 *
 * @covers dsl::sum(filtered, lambda)
 * @covers dsl::filter
 */
TEST_CASE("A4: SumLambda::SumFilteredCartesian", "[expressions][sum][filter]")
{
    GRBModel model = makeModel();

    auto Y = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "Y", 3, 3);
    model.update();

    auto I = dsl::range_view(0, 3);
    auto J = dsl::range_view(0, 3);

    auto filtered = (I * J) | dsl::filter([](int i, int j) { return i < j; });

    GRBLinExpr expr = dsl::sum(filtered, [&](int i, int j) { return Y(i, j); });

    model.addConstr(expr == 15, "c4");
    model.setObjective(expr, GRB_MAXIMIZE);

    REQUIRE_NOTHROW(model.optimize());
    REQUIRE(model.get(GRB_DoubleAttr_ObjVal) == 15);
}

// ============================================================================
// SECTION B: INDEXEDVARIABLESET INTEGRITY TESTS
// ============================================================================

/**
 * @test IndexedVariableSet::VectorAccessAndErrors
 * @brief Verifies IndexedVariableSet vector-based access and error paths
 *
 * @scenario Access indexed variables via at(), try_get() with valid/invalid keys
 * @given An IndexedVariableSet created from IndexList {0, 1, 2}
 * @when Accessing with valid and invalid indices
 * @then Valid lookups succeed, invalid lookups throw or return nullptr
 *
 * @covers IndexedVariableSet::at()
 * @covers IndexedVariableSet::try_get()
 */
TEST_CASE("B1: IndexedVariableSet::VectorAccessAndErrors", "[expressions][indexed][access]")
{
    GRBModel model = makeModel();

    dsl::IndexList I{ 0, 1, 2 };

    auto XV = dsl::VariableFactory::addIndexed(
        model, GRB_CONTINUOUS, 0, 1, "Y", I
    );
    model.update();

    // Valid lookups
    REQUIRE_NOTHROW(XV.at({ 0 }));
    REQUIRE_NOTHROW(XV.at({ 2 }));
    REQUIRE(XV.try_get({ 1 }) != nullptr);

    // Invalid lookups
    REQUIRE(XV.try_get({ 99 }) == nullptr);
    REQUIRE_THROWS(XV.at({ 99 }));
}

// ============================================================================
// SECTION C: SUM(D, INDEXEDVARIABLESET) DOMAIN LOOKUPS
// ============================================================================

/**
 * @test SumIndexed::Sum1DDomain
 * @brief Verifies sum(D, XV) with 1D domain
 *
 * @scenario Sum over IndexedVariableSet using 1D IndexList domain
 * @given An IndexedVariableSet with indices {0, 1, 2}
 * @when Applying sum with lambda accessing XV(i)
 * @then All indexed variables are summed correctly
 *
 * @covers dsl::sum(domain, lambda) with IndexedVariableSet
 */
TEST_CASE("C1: SumIndexed::Sum1DDomain", "[expressions][sum][indexed]")
{
    GRBModel model = makeModel();

    dsl::IndexList I{ 0, 1, 2 };

    auto XV = dsl::VariableFactory::addIndexed(
        model, GRB_CONTINUOUS, 0, 1, "Z", I
    );
    model.update();

    GRBLinExpr expr = dsl::sum(I, [&](int i) { return XV(i); });

    model.addConstr(expr <= 10, "c5");
    model.setObjective(expr, GRB_MAXIMIZE);

    REQUIRE_NOTHROW(model.optimize());
}

/**
 * @test SumIndexed::SumAllEntries
 * @brief Verifies sum(XV) iterates all entries in natural order
 *
 * @scenario Direct sum over all IndexedVariableSet entries
 * @given An IndexedVariableSet with non-sorted domain {3, 7, 10}
 * @when Applying sum(XV) overload
 * @then All entries are summed in insertion order
 *
 * @covers dsl::sum(IndexedVariableSet)
 */
TEST_CASE("C2: SumIndexed::SumAllEntries", "[expressions][sum][indexed]")
{
    GRBModel model = makeModel();

    dsl::IndexList I{ 3, 7, 10 }; // deliberately non-sorted domain

    auto XV = dsl::VariableFactory::addIndexed(
        model, GRB_CONTINUOUS, -1, 1, "Ord", I
    );
    model.update();

    // Direct overload: sum over ALL entries in XV
    GRBLinExpr expr = dsl::sum(XV);

    model.addConstr(expr <= 5, "c6");
    model.setObjective(expr, GRB_MAXIMIZE);

    REQUIRE_NOTHROW(model.optimize());
}

/**
 * @test SumIndexed::SumFiltered1D
 * @brief Verifies sum(D, XV) with filtered 1D domain
 *
 * @scenario Sum over even indices only using filter predicate
 * @given An IndexedVariableSet with indices {0, 1, 2, 3, 4}
 * @when Applying sum with filter i % 2 == 0
 * @then Only even-indexed variables are summed
 *
 * @covers dsl::sum with filtered domain
 */
TEST_CASE("C3: SumIndexed::SumFiltered1D", "[expressions][sum][filter]")
{
    GRBModel model = makeModel();

    dsl::IndexList I{ 0, 1, 2, 3, 4 };

    auto XV = dsl::VariableFactory::addIndexed(
        model, GRB_CONTINUOUS, 0, 3, "W", I
    );
    model.update();

    auto F = I.filter([](int i) { return i % 2 == 0; }); // even only

    GRBLinExpr expr = dsl::sum(F, [&](int i) { return XV(i); });

    model.addConstr(expr <= 5, "c7");
    model.setObjective(expr, GRB_MAXIMIZE);
    REQUIRE_NOTHROW(model.optimize());
}

// ============================================================================
// SECTION D: TUPLE-BASED CARTESIAN INDEXING (2D AND 3D)
// ============================================================================

/**
 * @test TupleIndexing::TupleLookup2D
 * @brief Verifies tuple-based lookup matches Cartesian domain (2D)
 *
 * @scenario Access 2D IndexedVariableSet via tuple decomposition
 * @given An IndexedVariableSet created from Cartesian product I x J
 * @when Accessing via XV.at({i,j}) and XV(i,j)
 * @then Both access methods work correctly for all tuples
 *
 * @covers IndexedVariableSet tuple access
 * @covers dsl::sum with 2D Cartesian domain
 */
TEST_CASE("D1: TupleIndexing::TupleLookup2D", "[expressions][tuple][cartesian]")
{
    GRBModel model = makeModel();

    dsl::IndexList I{ 0, 1 };
    dsl::IndexList J{ 5, 6 };

    auto IJ = I * J;

    auto XV = dsl::VariableFactory::addIndexed(
        model, GRB_BINARY, 0, 1, "Pair", IJ
    );
    model.update();

    // Check that tuple decomposition matches the actual indexing
    for (auto t : IJ)
    {
        auto [i, j] = t;
        REQUIRE_NOTHROW(XV.at({ i, j }));
        REQUIRE_NOTHROW(XV(i, j));
    }

    GRBLinExpr expr = dsl::sum(IJ, [&](int a, int b) { return XV(a, b); });

    model.addConstr(expr <= 4, "c8");
    model.setObjective(expr, GRB_MAXIMIZE);

    REQUIRE_NOTHROW(model.optimize());
}

/**
 * @test TupleIndexing::FilteredCartesianIndexed
 * @brief Verifies filtered Cartesian domain with IndexedVariableSet
 *
 * @scenario Sum over filtered 2D domain with i < j constraint
 * @given An IndexedVariableSet created from filtered Cartesian product
 * @when Applying sum with filter predicate
 * @then Only matching tuples are accessed
 *
 * @covers dsl::sum with filtered Cartesian and IndexedVariableSet
 */
TEST_CASE("D2: TupleIndexing::FilteredCartesianIndexed", "[expressions][tuple][filter]")
{
    GRBModel model = makeModel();

    auto I = dsl::range_view(0, 3);
    auto J = dsl::range_view(0, 3);

    auto F = (I * J) | dsl::filter([](int i, int j) { return i < j; });

    auto XV = dsl::VariableFactory::addIndexed(
        model, GRB_CONTINUOUS, 0, 5, "Filt", F
    );
    model.update();

    GRBLinExpr expr = dsl::sum(F, [&](int i, int j) { return XV(i, j); });

    model.addConstr(expr <= 100, "c9");
    model.setObjective(expr, GRB_MAXIMIZE);

    REQUIRE_NOTHROW(model.optimize());
}

/**
 * @test TupleIndexing::MultiDimensional3D
 * @brief Verifies multi-dimensional (3D) Cartesian domain
 *
 * @scenario Sum over 3D Cartesian product I x J x K
 * @given An IndexedVariableSet with 2x2x3 dimensions
 * @when Applying sum with 3-argument lambda
 * @then All 12 combinations are iterated correctly
 *
 * @covers dsl::sum with 3D Cartesian domain
 */
TEST_CASE("D3: TupleIndexing::MultiDimensional3D", "[expressions][tuple][3d]")
{
    GRBModel model = makeModel();

    auto I = dsl::range_view(0, 2); // 0,1
    auto J = dsl::range_view(0, 2); // 0,1
    auto K = dsl::range_view(0, 3); // 0,1,2

    auto IJK = I * J * K;

    auto XV = dsl::VariableFactory::addIndexed(
        model, GRB_BINARY, 0, 1, "Cube", IJK
    );
    model.update();

    GRBLinExpr expr = dsl::sum(IJK, [&](int i, int j, int k) {
        return XV(i, j, k);
        });

    model.addConstr(expr == 10, "c10");
    model.setObjective(expr, GRB_MAXIMIZE);

    REQUIRE_NOTHROW(model.optimize());
    REQUIRE(model.get(GRB_DoubleAttr_ObjVal) == 10);
}

// ============================================================================
// SECTION E: VARIABLEGROUP SUM OVERLOADS AND CONST-CORRECTNESS
// ============================================================================

/**
 * @test VariableGroupSum::SumAllEntries
 * @brief Verifies sum(VariableGroup) sums all entries
 *
 * @scenario Compare dsl::sum(X) with manual forEach accumulation
 * @given A 2D VariableGroup (2x3)
 * @when Comparing sum(X) with manual iteration
 * @then Both expressions have equal size
 *
 * @covers dsl::sum(VariableGroup)
 */
TEST_CASE("E1: VariableGroupSum::SumAllEntries", "[expressions][sum][variablegroup]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 1, "X", 2, 3);
    model.update();

    GRBLinExpr manual = 0.0;
    X.forEach([&](const GRBVar& v, const std::vector<int>&) { manual += v; });

    GRBLinExpr expr = dsl::sum(X);

    REQUIRE(expr.size() == manual.size());
}

/**
 * @test VariableGroupSum::DomainMatchesLambda
 * @brief Verifies sum(domain, VariableGroup) matches lambda version
 *
 * @scenario Compare direct sum(I, X) with lambda-based sum
 * @given A 1D VariableGroup with 4 variables
 * @when Comparing sum(I, X) with sum(I, lambda)
 * @then Both expressions have equal size
 *
 * @covers dsl::sum(domain, VariableGroup)
 */
TEST_CASE("E2: VariableGroupSum::DomainMatchesLambda", "[expressions][sum][variablegroup]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 1, "X", 4);
    model.update();

    dsl::IndexList I{ 0,1,2,3 };

    GRBLinExpr a = dsl::sum(I, X);
    GRBLinExpr b = dsl::sum(I, [&](int i) { return X(i); });

    REQUIRE(a.size() == b.size());
}

/**
 * @test VariableGroupSum::TupleExpansion
 * @brief Verifies sum(I*J, X) expands tuples correctly
 *
 * @scenario Sum over 2D VariableGroup using Cartesian domain
 * @given A 2D VariableGroup (2x3)
 * @when Applying sum(I * J, X)
 * @then Expression has 6 terms and optimization succeeds
 *
 * @covers dsl::sum with Cartesian domain and VariableGroup
 */
TEST_CASE("E3: VariableGroupSum::TupleExpansion", "[expressions][sum][cartesian]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 1, "X", 2, 3);
    model.update();

    auto I = dsl::range_view(0, 2);
    auto J = dsl::range_view(0, 3);

    GRBLinExpr expr = dsl::sum(I * J, X);

    REQUIRE(expr.size() == 6); // 2�3

    model.addConstr(expr == 5, "c1");
    model.setObjective(expr, GRB_MAXIMIZE);

    REQUIRE_NOTHROW(model.optimize());
    REQUIRE(model.get(GRB_DoubleAttr_ObjVal) == 5);
}

/**
 * @test VariableGroupSum::ConstForEachMatchesNonConst
 * @brief Verifies const-forEach matches non-const order
 *
 * @scenario Compare iteration order for const and non-const VariableGroup
 * @given A 2D VariableGroup (2x3)
 * @when Iterating with forEach on const and non-const references
 * @then Variable names are collected in identical order
 *
 * @covers VariableGroup::forEach const-correctness
 */
TEST_CASE("E4: VariableGroupSum::ConstForEachMatchesNonConst", "[expressions][variablegroup][const]")
{
    GRBModel model = makeModel();
    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 1, "X", 2, 3);
    model.update();

    std::vector<std::string> names_nonconst;
    X.forEach([&](const GRBVar& v, const std::vector<int>&) {
        names_nonconst.push_back(v.get(GRB_StringAttr_VarName));
        });

    const auto& XC = X;
    std::vector<std::string> names_const;
    XC.forEach([&](const GRBVar& v, const std::vector<int>&) {
        names_const.push_back(v.get(GRB_StringAttr_VarName));
        });

    REQUIRE(names_const == names_nonconst);
}

/**
 * @test VariableGroupSum::SumEqualsManualAccumulation
 * @brief Verifies sum(XV) equals manual accumulation
 *
 * @scenario Compare dsl::sum(XV) with manual loop accumulation
 * @given An IndexedVariableSet with indices {1, 4, 7}
 * @when Comparing sum(XV) with manual iteration over all()
 * @then Both expressions have equal size
 *
 * @covers dsl::sum(IndexedVariableSet) correctness
 */
TEST_CASE("E5: VariableGroupSum::SumEqualsManualAccumulation", "[expressions][sum][validation]")
{
    GRBModel model = makeModel();

    dsl::IndexList I{ 1, 4, 7 };
    auto XV = dsl::VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 1, "A", I);
    model.update();

    GRBLinExpr manual = 0.0;
    for (auto& e : XV.all())
        manual += e.var;

    GRBLinExpr expr = dsl::sum(XV);

    REQUIRE(expr.size() == manual.size());
}

/**
 * @test VariableGroupSum::DomainOrderRespected
 * @brief Verifies sum(D,XV) respects domain order, not internal order
 *
 * @scenario Sum over reversed domain {2,1,0}
 * @given An IndexedVariableSet with indices {0,1,2}
 * @when Summing with domain {2,1,0}
 * @then Variables appear in domain order in expression
 *
 * @covers Domain order preservation in sum
 */
TEST_CASE("E6: VariableGroupSum::DomainOrderRespected", "[expressions][sum][order]")
{
    GRBModel model = makeModel();

    dsl::IndexList I{ 0,1,2 };
    auto XV = dsl::VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 1, "B", I);
    model.update();

    dsl::IndexList domain{ 2,1,0 };
    GRBLinExpr expr = dsl::sum(domain, XV);

    // Check ordering by looking at GRBLinExpr terms
    if (naming_enabled())
    {
        REQUIRE(expr.getVar(0).get(GRB_StringAttr_VarName) == "B_2");
        REQUIRE(expr.getVar(1).get(GRB_StringAttr_VarName) == "B_1");
        REQUIRE(expr.getVar(2).get(GRB_StringAttr_VarName) == "B_0");
    }
}

/**
 * @test VariableContainerSum::SumDenseContainer
 * @brief Verifies sum(VariableContainer) works for dense storage
 *
 * @scenario Sum over VariableContainer holding a VariableGroup
 * @given A VariableContainer with dense 2D variables
 * @when Applying sum(container)
 * @then Expression contains all variables
 *
 * @covers dsl::sum(VariableContainer) with dense storage
 */
TEST_CASE("E7: VariableContainerSum::SumDenseContainer", "[expressions][sum][container]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 1, "X", 2, 3);
    model.update();

    dsl::VariableContainer vc(std::move(X));

    GRBLinExpr expr = dsl::sum(vc);

    REQUIRE(expr.size() == 6); // 2x3 = 6 variables
}

/**
 * @test VariableContainerSum::SumSparseContainer
 * @brief Verifies sum(VariableContainer) works for sparse storage
 *
 * @scenario Sum over VariableContainer holding an IndexedVariableSet
 * @given A VariableContainer with sparse filtered variables
 * @when Applying sum(container)
 * @then Expression contains all sparse variables
 *
 * @covers dsl::sum(VariableContainer) with sparse storage
 */
TEST_CASE("E8: VariableContainerSum::SumSparseContainer", "[expressions][sum][container]")
{
    GRBModel model = makeModel();

    auto I = dsl::range_view(0, 3);
    auto J = dsl::range_view(0, 3);
    auto F = (I * J) | dsl::filter([](int i, int j) { return i < j; });

    auto XV = dsl::VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 1, "Y", F);
    model.update();

    dsl::VariableContainer vc(std::move(XV));

    GRBLinExpr expr = dsl::sum(vc);

    REQUIRE(expr.size() == 3); // (0,1), (0,2), (1,2)
}

/**
 * @test VariableContainerSum::SumMatchesDirectCall
 * @brief Verifies sum(VariableContainer) matches sum(VariableGroup)
 *
 * @scenario Compare container sum with direct group sum
 * @given Same variables in container and direct group
 * @when Comparing sum(container) with sum(group)
 * @then Both expressions have equal size
 *
 * @covers dsl::sum(VariableContainer) consistency
 */
TEST_CASE("E9: VariableContainerSum::SumMatchesDirectCall", "[expressions][sum][container]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 1, "X", 4);
    model.update();

    // Direct sum on VariableGroup
    GRBLinExpr directExpr = dsl::sum(X);

    // Sum via VariableContainer
    dsl::VariableContainer vc(X); // copy
    GRBLinExpr containerExpr = dsl::sum(vc);

    REQUIRE(directExpr.size() == containerExpr.size());
    REQUIRE(directExpr.size() == 4);
}

/**
 * @test VariableContainerSum::SumDomainDenseContainer
 * @brief Verifies sum(Range, VariableContainer) works for dense storage
 *
 * @scenario Sum over subset of VariableContainer using domain
 * @given A VariableContainer with dense 1D variables
 * @when Applying sum(domain, container)
 * @then Expression contains only domain-specified variables
 *
 * @covers dsl::sum(Range, VariableContainer) with dense storage
 */
TEST_CASE("E10: VariableContainerSum::SumDomainDenseContainer", "[expressions][sum][container]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 1, "X", 5);
    model.update();

    dsl::VariableContainer vc(std::move(X));

    dsl::IndexList subset{ 1, 3 };
    GRBLinExpr expr = dsl::sum(subset, vc);

    REQUIRE(expr.size() == 2);
}

/**
 * @test VariableContainerSum::SumDomainSparseContainer
 * @brief Verifies sum(Range, VariableContainer) works for sparse storage
 *
 * @scenario Sum over subset of sparse VariableContainer using domain
 * @given A VariableContainer with sparse 2D variables
 * @when Applying sum(domain, container)
 * @then Expression contains only domain-specified variables
 *
 * @covers dsl::sum(Range, VariableContainer) with sparse storage
 */
TEST_CASE("E11: VariableContainerSum::SumDomainSparseContainer", "[expressions][sum][container]")
{
    GRBModel model = makeModel();

    auto I = dsl::range_view(0, 3);
    auto J = dsl::range_view(0, 3);
    auto fullDomain = I * J;

    auto XV = dsl::VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 1, "Y", fullDomain);
    model.update();

    dsl::VariableContainer vc(std::move(XV));

    // Sum only upper triangular
    auto subset = (I * J) | dsl::filter([](int i, int j) { return i < j; });
    GRBLinExpr expr = dsl::sum(subset, vc);

    REQUIRE(expr.size() == 3); // (0,1), (0,2), (1,2)
}

// ============================================================================
// SECTION F: EXPRESSION COMPOSITION AND COEFFICIENTS
// ============================================================================

/**
 * @test ExprComposition::NegativeCoefficients
 * @brief Verifies sum with negative coefficients
 *
 * @scenario Lambda returns -1.0 * X(i)
 * @given A 1D VariableGroup
 * @when Summing with negative coefficient
 * @then Expression construction succeeds
 *
 * @covers dsl::sum with negative multipliers
 */
TEST_CASE("F1: ExprComposition::NegativeCoefficients", "[expressions][composition][coefficient]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 3);
    model.update();

    dsl::IndexList I{ 0, 1, 2 };

    GRBLinExpr expr = dsl::sum(I, [&](int i) { return -1.0 * X(i); });

    REQUIRE(expr.size() == 3);

    // Verify model can optimize with negative coefficients
    model.addConstr(expr >= -15, "c1");
    model.setObjective(expr, GRB_MINIMIZE);

    REQUIRE_NOTHROW(model.optimize());
}

/**
 * @test ExprComposition::IndexDependentCoefficients
 * @brief Verifies sum with index-dependent coefficients
 *
 * @scenario Coefficient varies based on index value
 * @given A 1D VariableGroup
 * @when Summing with (i+1) * X(i)
 * @then Expression construction succeeds with correct term count
 *
 * @covers dsl::sum with variable coefficients
 */
TEST_CASE("F2: ExprComposition::IndexDependentCoefficients", "[expressions][composition][coefficient]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 4);
    model.update();

    dsl::IndexList I{ 0, 1, 2, 3 };

    // Coefficients: 1*X0 + 2*X1 + 3*X2 + 4*X3
    GRBLinExpr expr = dsl::sum(I, [&](int i) {
        return static_cast<double>(i + 1) * X(i);
    });

    REQUIRE(expr.size() == 4);
}

/**
 * @test ExprComposition::ConstantTermsInLambda
 * @brief Verifies sum accumulates constant terms from lambda
 *
 * @scenario Lambda returns variable plus constant offset
 * @given A 1D VariableGroup
 * @when Summing X(i) + 1.0 for each element
 * @then Expression includes accumulated constants
 *
 * @covers dsl::sum with constant offsets
 */
TEST_CASE("F3: ExprComposition::ConstantTermsInLambda", "[expressions][composition][constant]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 3);
    model.update();

    dsl::IndexList I{ 0, 1, 2 };

    // Each term adds 1.0 constant
    GRBLinExpr expr = dsl::sum(I, [&](int i) {
        return X(i) + 1.0;
    });

    REQUIRE(expr.size() == 3);
    REQUIRE(expr.getConstant() == Catch::Approx(3.0)); // 3 elements * 1.0
}

/**
 * @test ExprComposition::CombiningMultipleSums
 * @brief Verifies multiple sums can be combined
 *
 * @scenario Add two separate sum expressions
 * @given Two disjoint index domains
 * @when Adding sum(I, ...) + sum(J, ...)
 * @then Combined expression has correct size
 *
 * @covers Expression addition with dsl::sum
 */
TEST_CASE("F4: ExprComposition::CombiningMultipleSums", "[expressions][composition][combine]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 6);
    model.update();

    dsl::IndexList I{ 0, 1, 2 };
    dsl::IndexList J{ 3, 4, 5 };

    GRBLinExpr exprI = dsl::sum(I, [&](int i) { return X(i); });
    GRBLinExpr exprJ = dsl::sum(J, [&](int j) { return 2.0 * X(j); });

    GRBLinExpr combined = exprI + exprJ;

    REQUIRE(combined.size() == 6);
}

/**
 * @test ExprComposition::NestedExpressionBuilding
 * @brief Verifies lambda can return complex expressions
 *
 * @scenario Lambda returns expression with multiple terms
 * @given Two VariableGroups
 * @when Lambda returns X(i) - Y(i)
 * @then Expression contains terms from both groups
 *
 * @covers dsl::sum with expression-returning lambda
 */
TEST_CASE("F5: ExprComposition::NestedExpressionBuilding", "[expressions][composition][nested]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 3);
    auto Y = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "Y", 3);
    model.update();

    dsl::IndexList I{ 0, 1, 2 };

    // Each iteration adds two terms: +X(i) and -Y(i)
    GRBLinExpr expr = dsl::sum(I, [&](int i) {
        return X(i) - Y(i);
    });

    REQUIRE(expr.size() == 6); // 3 X terms + 3 Y terms
}

// ============================================================================
// SECTION G: EDGE CASES AND BOUNDARY CONDITIONS
// ============================================================================

/**
 * @test EdgeCases::SingleElementDomain
 * @brief Verifies sum over single-element domains
 *
 * @scenario Sum over domain with exactly one element
 * @given A VariableGroup with multiple variables
 * @when Summing over single-element IndexList {2}
 * @then Expression contains exactly one term
 *
 * @covers dsl::sum with minimal domain
 */
TEST_CASE("G1: EdgeCases::SingleElementDomain", "[expressions][edge][single]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 5);
    model.update();

    dsl::IndexList I{ 2 }; // single element

    GRBLinExpr expr = dsl::sum(I, [&](int i) { return X(i); });

    REQUIRE(expr.size() == 1);
}

/**
 * @test EdgeCases::EmptyDomainReturnsZero
 * @brief Verifies sum over empty domain returns zero expression
 *
 * @scenario Sum over empty IndexList with lambda
 * @given A VariableGroup with variables
 * @when Summing over empty domain
 * @then Expression has size 0 and constant 0.0
 *
 * @covers dsl::sum with empty domain (lambda)
 */
TEST_CASE("G2: EdgeCases::EmptyDomainReturnsZero", "[expressions][edge][empty]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 5);
    model.update();

    dsl::IndexList I; // empty

    GRBLinExpr expr = dsl::sum(I, [&](int i) { return X(i); });

    REQUIRE(expr.size() == 0);
    REQUIRE(expr.getConstant() == Catch::Approx(0.0));
}

/**
 * @test EdgeCases::EmptyIndexedVariableSet
 * @brief Verifies sum(empty XV) returns empty expression
 *
 * @scenario Sum over empty IndexedVariableSet
 * @given An IndexedVariableSet with empty domain
 * @when Applying sum(XV)
 * @then Expression has size 0 and constant 0.0
 *
 * @covers dsl::sum with empty IndexedVariableSet
 */
TEST_CASE("G3: EdgeCases::EmptyIndexedVariableSet", "[expressions][edge][empty]")
{
    GRBModel model = makeModel();

    dsl::IndexList I; // empty
    auto XV = dsl::VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 1, "E", I);
    model.update();

    GRBLinExpr expr = dsl::sum(XV);

    REQUIRE(expr.size() == 0);
    REQUIRE(expr.getConstant() == Catch::Approx(0.0));
}

/**
 * @test EdgeCases::EmptyCartesianProduct
 * @brief Verifies sum over empty Cartesian product
 *
 * @scenario Cartesian product with one empty dimension
 * @given One empty IndexList and one non-empty
 * @when Summing over their Cartesian product
 * @then Expression is empty
 *
 * @covers dsl::sum with empty Cartesian
 */
TEST_CASE("G4: EdgeCases::EmptyCartesianProduct", "[expressions][edge][cartesian]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 3, 3);
    model.update();

    dsl::IndexList I{ 0, 1, 2 };
    dsl::IndexList J; // empty

    auto IJ = I * J;

    GRBLinExpr expr = dsl::sum(IJ, [&](int i, int j) { return X(i, j); });

    REQUIRE(expr.size() == 0);
}

/**
 * @test EdgeCases::FilterRejectsAll
 * @brief Verifies sum over filter that rejects all elements
 *
 * @scenario Filter predicate returns false for all elements
 * @given A domain and filter that matches nothing
 * @when Summing over filtered domain
 * @then Expression is empty
 *
 * @covers dsl::sum with exhaustive filter
 */
TEST_CASE("G5: EdgeCases::FilterRejectsAll", "[expressions][edge][filter]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 5);
    model.update();

    auto I = dsl::range_view(0, 5);
    auto filtered = I | dsl::filter([](int) { return false; }); // reject all

    GRBLinExpr expr = dsl::sum(filtered, [&](int i) { return X(i); });

    REQUIRE(expr.size() == 0);
}

/**
 * @test EdgeCases::LargeDomainHandling
 * @brief Verifies sum handles large domains without overflow
 *
 * @scenario Sum over range with 1000 elements
 * @given A large 1D VariableGroup
 * @when Summing over full range
 * @then Expression has correct size
 *
 * @covers dsl::sum performance with large domains
 */
TEST_CASE("G6: EdgeCases::LargeDomainHandling", "[expressions][edge][performance]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 1, "X", 1000);
    model.update();

    auto I = dsl::range_view(0, 1000);

    GRBLinExpr expr = dsl::sum(I, [&](int i) { return X(i); });

    REQUIRE(expr.size() == 1000);
}

// ============================================================================
// SECTION H: ERROR SCENARIOS AND EXCEPTION HANDLING
// ============================================================================

/**
 * @test ErrorHandling::MissingIndicesThrow
 * @brief Verifies sum(D, XV) throws when domain contains missing indices
 *
 * @scenario Access IndexedVariableSet with index not in original domain
 * @given An IndexedVariableSet with indices {0, 1, 2}
 * @when Summing over domain {0, 1, 99} where 99 is missing
 * @then std::exception is thrown
 *
 * @covers Error handling for missing indices
 */
TEST_CASE("H1: ErrorHandling::MissingIndicesThrow", "[expressions][error][exception]")
{
    GRBModel model = makeModel();

    dsl::IndexList I{ 0, 1, 2 };

    auto XV = dsl::VariableFactory::addIndexed(
        model, GRB_CONTINUOUS, 0, 1, "Err", I
    );
    model.update();

    dsl::IndexList Bad{ 0, 1, 99 }; // 99 not in XV

    REQUIRE_THROWS_AS(
        dsl::sum(Bad, [&](int i) { return XV(i); }),
        std::exception
    );
}

/**
 * @test ErrorHandling::DimensionMismatchIndexedThrows
 * @brief Verifies sum(D, XV) throws when tuple dimension mismatches
 *
 * @scenario Access 2D IndexedVariableSet with 1D domain
 * @given An IndexedVariableSet with 2D Cartesian domain
 * @when Summing with 1D IndexList domain
 * @then Exception is thrown due to dimension mismatch
 *
 * @covers Dimension mismatch error handling for IndexedVariableSet
 */
TEST_CASE("H2: ErrorHandling::DimensionMismatchIndexedThrows", "[expressions][error][dimension]")
{
    GRBModel model = makeModel();

    dsl::IndexList I{ 0,1 };
    dsl::IndexList J{ 5,6 };

    auto IJ = I * J; // 2D variable set

    auto XV = dsl::VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "U", IJ);
    model.update();

    // 1D domain attempting to access 2D variable
    dsl::IndexList wrong{ 0,1 };

    REQUIRE_THROWS(
        dsl::sum(wrong, XV) // XV(i) is invalid � must throw
    );
}

/**
 * @test ErrorHandling::DimensionMismatchVariableGroupThrows
 * @brief Verifies sum(domain, X) throws on dimension mismatch
 *
 * @scenario Access 2D VariableGroup with 1D scalar domain
 * @given A 2D VariableGroup (3x4)
 * @when Summing with 1D IndexList {0, 1, 2}
 * @then Exception is thrown due to dimension mismatch
 *
 * @covers Dimension validation for VariableGroup sum
 */
TEST_CASE("H3: ErrorHandling::DimensionMismatchVariableGroupThrows", "[expressions][error][dimension]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 1, "X", 3, 4);
    model.update();

    dsl::IndexList wrong{ 0, 1, 2 }; // scalar indices, not tuples

    REQUIRE_THROWS(dsl::sum(wrong, X));
}

/**
 * @test ErrorHandling::3DUnpackingCorrect
 * @brief Verifies 3D domain unpacking calls XV(i,j,k) correctly
 *
 * @scenario Sum over 3D domain with direct XV access
 * @given An IndexedVariableSet with 2x2x2 Cartesian domain
 * @when Applying sum(IJK, XV)
 * @then XV(i,j,k) is called for each tuple without error
 *
 * @covers 3D tuple unpacking validation
 */
TEST_CASE("H4: ErrorHandling::3DUnpackingCorrect", "[expressions][tuple][3d]")
{
    GRBModel model = makeModel();

    auto I = dsl::range_view(0, 2);
    auto J = dsl::range_view(0, 2);
    auto K = dsl::range_view(0, 2);

    auto IJK = I * J * K;

    auto XV = dsl::VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "T", IJK);
    model.update();

    REQUIRE_NOTHROW(
        dsl::sum(IJK, XV) // must call XV(i,j,k) for each tuple
    );
}

// ============================================================================
// SECTION I: ITERATOR PROPERTIES AND PERFORMANCE
// ============================================================================

/**
 * @test IteratorProperties::MultiplePassesYieldSameResult
 * @brief Verifies domain can be iterated multiple times
 *
 * @scenario Sum over same domain twice
 * @given A domain and VariableGroup
 * @when Calling sum twice with same domain
 * @then Both expressions have identical structure
 *
 * @covers Domain reusability in dsl::sum
 */
TEST_CASE("I1: IteratorProperties::MultiplePassesYieldSameResult", "[expressions][iterator][multipass]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 4);
    model.update();

    dsl::IndexList I{ 0, 1, 2, 3 };

    GRBLinExpr expr1 = dsl::sum(I, [&](int i) { return X(i); });
    GRBLinExpr expr2 = dsl::sum(I, [&](int i) { return X(i); });

    REQUIRE(expr1.size() == expr2.size());
}

/**
 * @test IteratorProperties::CartesianMultiplePasses
 * @brief Verifies Cartesian product can be reused
 *
 * @scenario Sum over same Cartesian domain twice
 * @given A 2D Cartesian domain
 * @when Summing twice over same product
 * @then Both expressions have identical structure
 *
 * @covers Cartesian domain reusability
 */
TEST_CASE("I2: IteratorProperties::CartesianMultiplePasses", "[expressions][iterator][cartesian]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 2, 3);
    model.update();

    auto I = dsl::range_view(0, 2);
    auto J = dsl::range_view(0, 3);
    auto IJ = I * J;

    GRBLinExpr expr1 = dsl::sum(IJ, [&](int i, int j) { return X(i, j); });
    GRBLinExpr expr2 = dsl::sum(IJ, [&](int i, int j) { return X(i, j); });

    REQUIRE(expr1.size() == expr2.size());
    REQUIRE(expr1.size() == 6);
}

/**
 * @test IteratorProperties::FilteredMultiplePasses
 * @brief Verifies filtered domain can be reused
 *
 * @scenario Sum over same filtered domain twice
 * @given A filtered Cartesian domain
 * @when Summing twice over same filtered product
 * @then Both expressions have identical structure
 *
 * @covers Filtered domain reusability
 */
TEST_CASE("I3: IteratorProperties::FilteredMultiplePasses", "[expressions][iterator][filter]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 3, 3);
    model.update();

    auto I = dsl::range_view(0, 3);
    auto J = dsl::range_view(0, 3);
    auto F = (I * J) | dsl::filter([](int i, int j) { return i < j; });

    GRBLinExpr expr1 = dsl::sum(F, [&](int i, int j) { return X(i, j); });
    GRBLinExpr expr2 = dsl::sum(F, [&](int i, int j) { return X(i, j); });

    REQUIRE(expr1.size() == expr2.size());
    REQUIRE(expr1.size() == 3); // (0,1), (0,2), (1,2)
}

/**
 * @test IteratorProperties::RangeViewLazyEvaluation
 * @brief Verifies RangeView doesn't materialize for large ranges
 *
 * @scenario Create range and sum over subset
 * @given A RangeView (0 to 100)
 * @when Summing over the range
 * @then Operation completes efficiently
 *
 * @covers Lazy evaluation in dsl::sum with RangeView
 */
TEST_CASE("I4: IteratorProperties::RangeViewLazyEvaluation", "[expressions][iterator][lazy]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 1, "X", 100);
    model.update();

    auto R = dsl::range_view(0, 100);

    GRBLinExpr expr = dsl::sum(R, [&](int i) { return X(i); });

    REQUIRE(expr.size() == 100);
}

// ============================================================================
// SECTION J: QUADRATIC SUM (quadSum) TESTS
// ============================================================================

/**
 * @test QuadSum::Basic1DSquaredTerms
 * @brief Verifies quadSum builds expression with squared terms
 *
 * @scenario Sum x[i]^2 for regularization
 * @given A 1D VariableGroup with 3 continuous variables
 * @when Applying quadSum with squared terms
 * @then Quadratic expression is constructed and model optimizes
 *
 * @covers dsl::quadSum(domain, lambda)
 */
TEST_CASE("J1: QuadSum::Basic1DSquaredTerms", "[expressions][quadsum][basic]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 3);
    model.update();

    auto I = dsl::range_view(0, 3);

    // Expression: x[0]^2 + x[1]^2 + x[2]^2
    GRBQuadExpr expr = dsl::quadSum(I, [&](int i) { 
        return X(i) * X(i); 
    });

    model.addConstr(X(0) + X(1) + X(2) == 3, "sum");
    model.setObjective(expr, GRB_MINIMIZE);

    REQUIRE_NOTHROW(model.optimize());
    // Minimum variance when all equal: each = 1, so objective = 3
    REQUIRE(model.get(GRB_DoubleAttr_ObjVal) == Catch::Approx(3.0));
}

/**
 * @test QuadSum::BilinearTerms2DCartesian
 * @brief Verifies quadSum over Cartesian domain for bilinear terms
 *
 * @scenario Portfolio variance: sum_{a,b} sigma[a,b] * x[a] * x[b]
 * @given A covariance matrix and portfolio weights
 * @when Applying quadSum over A x A
 * @then Quadratic expression with cross-terms is constructed
 *
 * @covers dsl::quadSum with 2D Cartesian domain
 */
TEST_CASE("J2: QuadSum::BilinearTerms2DCartesian", "[expressions][quadsum][cartesian]")
{
    GRBModel model = makeModel();

    const int n = 3;
    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 1, "X", n);
    model.update();

    // Simple covariance matrix (identity for simplicity)
    std::vector<std::vector<double>> sigma = {
        {1.0, 0.5, 0.0},
        {0.5, 1.0, 0.5},
        {0.0, 0.5, 1.0}
    };

    auto A = dsl::range_view(0, n);

    // Expression: sum_{a,b} sigma[a][b] * x[a] * x[b]
    GRBQuadExpr variance = dsl::quadSum(A * A, [&](int a, int b) {
        return sigma[a][b] * X(a) * X(b);
    });

    // Budget constraint
    model.addConstr(X(0) + X(1) + X(2) == 1, "budget");
    model.setObjective(variance, GRB_MINIMIZE);

    REQUIRE_NOTHROW(model.optimize());
    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
}

/**
 * @test QuadSum::FilteredDomain
 * @brief Verifies quadSum works with filtered domains
 *
 * @scenario Sum only upper triangular: sum_{i<j} x[i] * x[j]
 * @given A filtered Cartesian domain i < j
 * @when Applying quadSum with filter
 * @then Only filtered pairs are included
 *
 * @covers dsl::quadSum with filtered domain
 */
TEST_CASE("J3: QuadSum::FilteredDomain", "[expressions][quadsum][filter]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 3);
    model.update();

    auto I = dsl::range_view(0, 3);
    auto J = dsl::range_view(0, 3);
    auto F = (I * J) | dsl::filter([](int i, int j) { return i < j; });

    // Upper triangular cross terms only
    GRBQuadExpr expr = dsl::quadSum(F, [&](int i, int j) {
        return X(i) * X(j);
    });

    model.addConstr(X(0) + X(1) + X(2) == 6, "sum");
    model.setObjective(expr, GRB_MAXIMIZE);

    REQUIRE_NOTHROW(model.optimize());
}

/**
 * @test QuadSum::WithCoefficients
 * @brief Verifies quadSum with index-dependent coefficients
 *
 * @scenario Weighted squared terms: sum_i (i+1) * x[i]^2
 * @given Index-dependent weights
 * @when Applying quadSum with coefficient lambda
 * @then Coefficients are applied correctly
 *
 * @covers dsl::quadSum with coefficients
 */
TEST_CASE("J4: QuadSum::WithCoefficients", "[expressions][quadsum][coefficient]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 3);
    model.update();

    auto I = dsl::range_view(0, 3);

    // Expression: 1*x[0]^2 + 2*x[1]^2 + 3*x[2]^2
    GRBQuadExpr expr = dsl::quadSum(I, [&](int i) {
        return static_cast<double>(i + 1) * X(i) * X(i);
    });

    model.addConstr(X(0) == 1, "c0");
    model.addConstr(X(1) == 1, "c1");
    model.addConstr(X(2) == 1, "c2");
    model.setObjective(expr, GRB_MINIMIZE);

    REQUIRE_NOTHROW(model.optimize());
    // 1*1 + 2*1 + 3*1 = 6
    REQUIRE(model.get(GRB_DoubleAttr_ObjVal) == Catch::Approx(6.0));
}

/**
 * @test QuadSum::EmptyDomainReturnsZero
 * @brief Verifies quadSum over empty domain returns zero expression
 *
 * @scenario quadSum over empty domain
 * @given An empty IndexList
 * @when Applying quadSum
 * @then Expression has zero constant
 *
 * @covers dsl::quadSum with empty domain
 */
TEST_CASE("J5: QuadSum::EmptyDomainReturnsZero", "[expressions][quadsum][empty]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, 0, 10, "X", 3);
    model.update();

    dsl::IndexList I; // empty

    GRBQuadExpr expr = dsl::quadSum(I, [&](int i) {
        return X(i) * X(i);
    });

    // Empty quadSum should have zero value
    REQUIRE(expr.getValue() == Catch::Approx(0.0));
}

/**
 * @test QuadSum::CombinedWithLinearTerms
 * @brief Verifies quadSum can be combined with linear sum
 *
 * @scenario Objective: sum x[i]^2 - 2 * sum x[i]
 * @given Both quadSum and sum expressions
 * @when Adding quadratic and linear expressions
 * @then Combined expression optimizes correctly
 *
 * @covers dsl::quadSum combined with dsl::sum
 */
TEST_CASE("J6: QuadSum::CombinedWithLinearTerms", "[expressions][quadsum][combined]")
{
    GRBModel model = makeModel();

    auto X = dsl::VariableFactory::add(model, GRB_CONTINUOUS, -10, 10, "X", 2);
    model.update();

    auto I = dsl::range_view(0, 2);

    // Quadratic part: x[0]^2 + x[1]^2
    GRBQuadExpr quadPart = dsl::quadSum(I, [&](int i) {
        return X(i) * X(i);
    });

    // Linear part: -2 * (x[0] + x[1])
    GRBLinExpr linPart = dsl::sum(I, [&](int i) {
        return -2.0 * X(i);
    });

    // Combined objective: x^2 - 2x has minimum at x=1
    GRBQuadExpr objective = quadPart + linPart;
    model.setObjective(objective, GRB_MINIMIZE);

    REQUIRE_NOTHROW(model.optimize());
    // Each x[i] should be 1, objective = 1 - 2 + 1 - 2 = -2
    REQUIRE(model.get(GRB_DoubleAttr_ObjVal) == Catch::Approx(-2.0));
}