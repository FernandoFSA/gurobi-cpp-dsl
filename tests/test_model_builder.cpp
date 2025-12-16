/*
===============================================================================
TEST MODEL BUILDER — Comprehensive tests for model_builder.h
===============================================================================

OVERVIEW
--------
Validates the DSL ModelBuilder template class for orchestrating optimization
model construction. Tests cover initialization lifecycle, template method hooks,
variable and constraint creation, parameter application, environment configuration,
and full optimization workflows.

TEST ORGANIZATION
-----------------
• Section A: Orchestration and lifecycle
• Section B: Variable creation and access
• Section C: Constraint creation and access
• Section D: Parameter and environment configuration
• Section E: Advanced multi-dimensional models
• Section F: Integration and complex scenarios
• Section G: Lazy initialization and const access
• Section H: External model support
• Section I: DataStore integration
• Section J: Edge cases and error conditions
• Section K: Solution diagnostics utilities
• Section L: Objective helpers (minimize/maximize)
• Section M: Parameter presets and convenience setters

TEST STRATEGY
-------------
• Verify template method hook invocation order
• Confirm lazy initialization behavior
• Validate variable and constraint table integration
• Exercise full optimization workflow with Gurobi solver
• Test warm-start and post-optimization hooks
• Verify external model construction path
• Test DataStore for metadata and parameter storage
• Exercise edge cases including repeated optimize calls
• Verify solution diagnostics helpers (status, objVal, mipGap, etc.)
• Test objective helpers (minimize, maximize)
• Test parameter presets and convenience setters with tracking

DEPENDENCIES
------------
• Catch2 v3.0+ - Test framework
• model_builder.h - System under test
• variables.h, constraints.h - Supporting components
• expressions.h, indexing.h - DSL utilities
• Gurobi C++ API - Solver backend

===============================================================================
*/

#define CATCH_CONFIG_MAIN
#include "catch_amalgamated.hpp"

#include <gurobi_dsl/model_builder.h>
#include <gurobi_dsl/variables.h>
#include <gurobi_dsl/constraints.h>
#include <gurobi_dsl/expressions.h>
#include <gurobi_dsl/indexing.h>
#include <gurobi_dsl/naming.h>

using namespace dsl;

// ============================================================================
// TEST UTILITIES AND FIXTURES
// ============================================================================

// Dummy enums for basic tests
DECLARE_ENUM_WITH_COUNT(TestVars, X);
DECLARE_ENUM_WITH_COUNT(TestCons, Cap);

/**
 * @class BasicBuilder
 * @brief Test fixture for orchestration and basic functionality tests
 *
 * @details Provides a minimal ModelBuilder implementation that tracks
 *          hook invocations and creates simple binary variables with
 *          basic constraints for testing the template method pattern.
 */
class BasicBuilder : public ModelBuilder<TestVars, TestCons>
{
public:
    using Base = ModelBuilder<TestVars, TestCons>;
    using Base::Base;

    int calls_addVariables = 0;
    int calls_addConstraints = 0;
    int calls_addParameters = 0;
    int calls_addObjective = 0;
    int calls_beforeOptimize = 0;
    int calls_afterOptimize = 0;

    int n = 3;

    void configureEnvironment(GRBEnv& env) override
    {
        // silence output
        env.set(GRB_IntParam_OutputFlag, 0);
    }

    void addVariables() override
    {
        calls_addVariables++;

        auto X = VariableFactory::add(
            model(), GRB_BINARY, 0.0, 1.0, "x", n);

        variables().set(TestVars::X, std::move(X));
    }

    void addConstraints() override
    {
        calls_addConstraints++;

        auto& X = variables()(TestVars::X);

        auto G = ConstraintFactory::add(
            model(), "cap",
            [&](const std::vector<int>& idx)
            {
                int i = idx[0];
                return X(i) <= 1.0;
            },
            n
        );

        constraints().set(TestCons::Cap, std::move(G));
    }

    void addParameters() override
    {
        calls_addParameters++;
        setParam(GRB_IntParam_Threads, 1);
        setParam(GRB_DoubleParam_TimeLimit, 10.0);
    }

    void addObjective() override
    {
        calls_addObjective++;

        auto& X = variables()(TestVars::X);
        GRBLinExpr obj = 0;
        for (int i = 0; i < n; i++)
            obj += X(i);

        model().setObjective(obj, GRB_MAXIMIZE);
    }

    void beforeOptimize() override
    {
        calls_beforeOptimize++;
    }

    void afterOptimize() override
    {
        calls_afterOptimize++;
    }
};

// ============================================================================
// SECTION A: ORCHESTRATION AND LIFECYCLE
// ============================================================================

/**
 * @test OrchestrationOrder::HookInvocation
 * @brief Verifies template method hooks are called in correct order
 *
 * @scenario ModelBuilder.optimize() orchestrates hook invocation
 * @given A BasicBuilder with tracking counters
 * @when Calling optimize()
 * @then All hooks are called exactly once in expected order
 *
 * @covers ModelBuilder::optimize()
 */
TEST_CASE("A1: OrchestrationOrder::HookInvocation", "[ModelBuilder][orchestration]")
{
    BasicBuilder builder;
    builder.optimize();

    REQUIRE(builder.calls_addVariables == 1);
    REQUIRE(builder.calls_addConstraints == 1);
    REQUIRE(builder.calls_addParameters == 1);
    REQUIRE(builder.calls_addObjective == 1);
    REQUIRE(builder.calls_beforeOptimize == 1);
    REQUIRE(builder.calls_afterOptimize == 1);
}

// ============================================================================
// SECTION B: VARIABLE CREATION AND ACCESS
// ============================================================================

/**
 * @test VariableCreation::BasicVariables
 * @brief Verifies variables are created correctly via addVariables hook
 *
 * @scenario BasicBuilder creates binary variables
 * @given A BasicBuilder with n=3 variables
 * @when Calling optimize() and accessing variables
 * @then Variables have correct bounds, type, and dimensions
 *
 * @covers ModelBuilder::addVariables(), VariableTable integration
 */
TEST_CASE("B1: VariableCreation::BasicVariables", "[ModelBuilder][variables]")
{
    BasicBuilder builder;
    builder.optimize();

    auto& X = builder.variables()(TestVars::X);
    REQUIRE(X.asGroup().dimension() == 1);
    REQUIRE(X.asGroup().size(0) == builder.n);

    for (int i = 0; i < builder.n; i++)
    {
        auto v = X(i);
        REQUIRE(v.get(GRB_DoubleAttr_LB) == Catch::Approx(0.0));
        REQUIRE(v.get(GRB_DoubleAttr_UB) == Catch::Approx(1.0));
        REQUIRE(v.get(GRB_CharAttr_VType) == GRB_BINARY);
    }
}

// ============================================================================
// SECTION C: CONSTRAINT CREATION AND ACCESS
// ============================================================================

/**
 * @test ConstraintCreation::BasicConstraints
 * @brief Verifies constraints are created correctly via addConstraints hook
 *
 * @scenario BasicBuilder creates capacity constraints
 * @given A BasicBuilder with n=3 constraints
 * @when Calling optimize() and accessing constraints
 * @then Constraints have correct sense and dimensions
 *
 * @covers ModelBuilder::addConstraints(), ConstraintTable integration
 */
TEST_CASE("C1: ConstraintCreation::BasicConstraints", "[ModelBuilder][constraints]")
{
    BasicBuilder builder;
    builder.optimize();

    auto& G = builder.constraints()(TestCons::Cap);
    REQUIRE(G.asGroup().dimension() == 1);
    REQUIRE(G.asGroup().size(0) == builder.n);

    for (int i = 0; i < builder.n; i++)
    {
        auto c = G(i);
        REQUIRE(c.get(GRB_CharAttr_Sense) == GRB_LESS_EQUAL);
    }
}

// ============================================================================
// SECTION D: PARAMETER AND ENVIRONMENT CONFIGURATION
// ============================================================================

/**
 * @test ParameterApplication::ModelParameters
 * @brief Verifies model parameters are applied via addParameters hook
 *
 * @scenario BasicBuilder sets Threads and TimeLimit parameters
 * @given A BasicBuilder that sets specific parameter values
 * @when Calling optimize() and querying parameters
 * @then Parameters have expected values on the model
 *
 * @covers ModelBuilder::addParameters(), setParam()
 */
TEST_CASE("D1: ParameterApplication::ModelParameters", "[ModelBuilder][parameters]")
{
    BasicBuilder builder;
    builder.optimize();

    REQUIRE(builder.model().get(GRB_IntParam_Threads) == 1);
    REQUIRE(builder.model().get(GRB_DoubleParam_TimeLimit) == Catch::Approx(10.0));
}

/**
 * @class EnvTestBuilder
 * @brief Test fixture for environment configuration tests
 *
 * @details Minimal builder that tracks configureEnvironment invocations
 *          to verify the environment hook is called correctly.
 */
class EnvTestBuilder : public ModelBuilder<TestVars, TestCons>
{
public:
    using Base = ModelBuilder<TestVars, TestCons>;
    using Base::Base;

    int envConfigured = 0;

    void configureEnvironment(GRBEnv& env) override
    {
        envConfigured++;
        env.set(GRB_IntParam_OutputFlag, 0);
    }
};

/**
 * @test EnvironmentConfiguration::HookInvocation
 * @brief Verifies configureEnvironment hook is called during initialization
 *
 * @scenario EnvTestBuilder tracks environment configuration
 * @given An EnvTestBuilder with configuration counter
 * @when Calling optimize()
 * @then configureEnvironment is called exactly once
 *
 * @covers ModelBuilder::configureEnvironment(), initialize()
 */
TEST_CASE("D2: EnvironmentConfiguration::HookInvocation", "[ModelBuilder][environment]")
{
    EnvTestBuilder builder;
    builder.optimize();

    REQUIRE(builder.envConfigured == 1);
}

// ============================================================================
// SECTION E: ADVANCED MULTI-DIMENSIONAL MODELS
// ============================================================================

// Advanced enums for complex tests
DECLARE_ENUM_WITH_COUNT(AdvVars, X, Y);
DECLARE_ENUM_WITH_COUNT(AdvCons, Bal, Coupling, Quad);

/**
 * @class AdvancedBuilder
 * @brief Test fixture for advanced multi-dimensional model tests
 *
 * @details Provides a comprehensive ModelBuilder implementation with:
 *          - 2D continuous variables X[I,J]
 *          - 1D binary variables Y[I]
 *          - Balance, coupling, and scalar constraints
 *          - Warm-start support in beforeOptimize
 *          - Solution validation in afterOptimize
 */
class AdvancedBuilder : public ModelBuilder<AdvVars, AdvCons>
{
public:
    using Base = ModelBuilder<AdvVars, AdvCons>;
    using Base::Base;

    int I = 4;
    int J = 3;
    std::vector<double> warm_Y;

    GRBLinExpr objective_expr = 0;

    void configureEnvironment(GRBEnv& env) override
    {
        env.set(GRB_IntParam_OutputFlag, 0);
    }

    void addVariables() override
    {
        auto X = VariableFactory::add(
            model(), GRB_CONTINUOUS, 0.0, 10.0, "X", I, J);

        auto Y = VariableFactory::add(
            model(), GRB_BINARY, 0.0, 1.0, "Y", I);

        variables().set(AdvVars::X, std::move(X));
        variables().set(AdvVars::Y, std::move(Y));
    }

    void addConstraints() override
    {
        auto& X = variables()(AdvVars::X);
        auto& Y = variables()(AdvVars::Y);

        // Bal[i]: sum_j X[i,j] == (i+1)*Y[i]
        auto Bal = ConstraintFactory::add(
            model(), "bal",
            [&](const std::vector<int>& idx)
            {
                int i = idx[0];
                GRBLinExpr lhs =
                    sum(range(0, J), [&](int j) { return X(i, j); });
                GRBLinExpr rhs = (i + 1) * Y(i);
                return lhs == rhs;
            },
            I
        );
        constraints().set(AdvCons::Bal, std::move(Bal));

        // Coupling[i,j]: X[i,j] <= 5
        auto Cpl = ConstraintFactory::add(
            model(), "cpl",
            [&](const std::vector<int>& idx)
            {
                int i = idx[0], j = idx[1];
                return X(i, j) <= 5.0;
            },
            I, J
        );
        constraints().set(AdvCons::Coupling, std::move(Cpl));

        // Quad-like scalar constraint
        auto Quad = ConstraintFactory::add(
            model(), "quadtest",
            [&](const std::vector<int>&)
            {
                GRBLinExpr s =
                    sum(range(0, I) * range(0, J),
                        [&](int i, int j) { return X(i, j); });
                return s <= 1e9;
            }
        );
        constraints().set(AdvCons::Quad, std::move(Quad));
    }

    void addParameters() override
    {
        setParam(GRB_IntParam_Threads, 1);
        setParam(GRB_DoubleParam_TimeLimit, 5.0);
    }

    void addObjective() override
    {
        auto& X = variables()(AdvVars::X);

        objective_expr =
            sum(range(0, I) * range(0, J),
                [&](int i, int j) { return (i + 1) * X(i, j); });

        model().setObjective(objective_expr, GRB_MAXIMIZE);
    }

    void beforeOptimize() override
    {
        auto& Y = variables()(AdvVars::Y);
        warm_Y.resize(I);

        for (int i = 0; i < I; i++)
        {
            Y(i).set(GRB_DoubleAttr_Start, 1.0);
            warm_Y[i] = 1.0;
        }
    }

    void afterOptimize() override
    {
        auto& Y = variables()(AdvVars::Y);

        for (int i = 0; i < I; i++)
            REQUIRE_THAT(
                Y(i).get(GRB_DoubleAttr_X),
                Catch::Matchers::WithinAbs(warm_Y[i], 1e-6)
            );
    }
};

/**
 * @test AdvancedVariables::MultidimensionalShapes
 * @brief Verifies multi-dimensional variable creation
 *
 * @scenario AdvancedBuilder creates 2D and 1D variable arrays
 * @given An AdvancedBuilder with I=4, J=3 dimensions
 * @when Calling addVariables() and accessing variable shapes
 * @then Variables have correct dimensions and sizes
 *
 * @covers VariableFactory::add() with multiple dimensions
 */
TEST_CASE("E1: AdvancedVariables::MultidimensionalShapes", "[ModelBuilder][advanced][variables]")
{
    AdvancedBuilder B;
    B.addVariables();

    auto& X = B.variables()(AdvVars::X);
    REQUIRE(X.asGroup().dimension() == 2);
    REQUIRE(X.asGroup().size(0) == B.I);
    REQUIRE(X.asGroup().size(1) == B.J);

    auto& Y = B.variables()(AdvVars::Y);
    REQUIRE(Y.asGroup().dimension() == 1);
    REQUIRE(Y.asGroup().size(0) == B.I);
}

/**
 * @test AdvancedConstraints::MultidimensionalShapes
 * @brief Verifies multi-dimensional constraint creation
 *
 * @scenario AdvancedBuilder creates 1D, 2D, and scalar constraints
 * @given An AdvancedBuilder with various constraint types
 * @when Calling addConstraints() and accessing constraint shapes
 * @then Constraints have correct dimensions and sizes
 *
 * @covers ConstraintFactory::add() with multiple dimensions
 */
TEST_CASE("E2: AdvancedConstraints::MultidimensionalShapes", "[ModelBuilder][advanced][constraints]")
{
    AdvancedBuilder B;
    B.addVariables();
    B.addConstraints();

    auto& Bal = B.constraints()(AdvCons::Bal);
    REQUIRE(Bal.asGroup().dimension() == 1);
    REQUIRE(Bal.asGroup().size(0) == B.I);

    auto& Cpl = B.constraints()(AdvCons::Coupling);
    REQUIRE(Cpl.asGroup().dimension() == 2);
    REQUIRE(Cpl.asGroup().size(0) == B.I);
    REQUIRE(Cpl.asGroup().size(1) == B.J);

    auto& Q = B.constraints()(AdvCons::Quad);
    REQUIRE(Q.asGroup().dimension() == 0);
}

/**
 * @test AdvancedConstraints::ForEachIteration
 * @brief Verifies constraint iteration via forEach
 *
 * @scenario Iterating through 2D constraint array
 * @given An AdvancedBuilder with Coupling constraints [I,J]
 * @when Calling forEach on constraint array
 * @then All elements are visited with correct indices
 *
 * @covers ConstraintArray::forEach()
 */
TEST_CASE("E3: AdvancedConstraints::ForEachIteration", "[ModelBuilder][advanced][iteration]")
{
    AdvancedBuilder B;
    B.addVariables();
    B.addConstraints();

    auto& Cpl = B.constraints()(AdvCons::Coupling);

    int count = 0;
    Cpl.forEach([&](GRBConstr& c, const std::vector<int>& idx) {
        REQUIRE(idx.size() == 2);
        REQUIRE(idx[0] < B.I);
        REQUIRE(idx[1] < B.J);
        count++;
        });

    REQUIRE(count == B.I * B.J);
}

/**
 * @test AdvancedExpressions::SumBuilding
 * @brief Verifies sum expression construction and evaluation
 *
 * @scenario Building linear expressions with sum() helper
 * @given An AdvancedBuilder with 2D variables
 * @when Creating sum expression and optimizing
 * @then Expression evaluates to expected value
 *
 * @covers sum() expression builder integration
 */
TEST_CASE("E4: AdvancedExpressions::SumBuilding", "[ModelBuilder][advanced][expressions]")
{
    AdvancedBuilder B;
    B.addVariables();

    auto& X = B.variables()(AdvVars::X);

    GRBLinExpr expr =
        sum(range(0, B.I) * range(0, B.J),
            [&](int i, int j) { return X(i, j); });

    // give them values
    for (int i = 0; i < B.I; i++)
        for (int j = 0; j < B.J; j++)
        {
			// set upper and lower to 1.0 to fix value
            X(i, j).set(GRB_DoubleAttr_LB, 1.0);
			X(i, j).set(GRB_DoubleAttr_UB, 1.0);
        }

    B.model().setObjective(expr, GRB_MINIMIZE);
    B.model().optimize();

    REQUIRE(B.model().get(GRB_DoubleAttr_ObjVal)
        == Catch::Approx(B.I * B.J));
}

// ============================================================================
// SECTION F: INTEGRATION AND COMPLEX SCENARIOS
// ============================================================================

/**
 * @test Integration::WarmStart
 * @brief Verifies warm-start functionality in beforeOptimize hook
 *
 * @scenario AdvancedBuilder sets start values for binary variables
 * @given An AdvancedBuilder with warm-start logic
 * @when Calling optimize()
 * @then Start values are applied and preserved in solution
 *
 * @covers ModelBuilder::beforeOptimize(), warm-start integration
 */
TEST_CASE("F1: Integration::WarmStart", "[ModelBuilder][integration][warmstart]")
{
    AdvancedBuilder B;
    B.optimize();

    auto& Y = B.variables()(AdvVars::Y);
    for (int i = 0; i < B.I; i++)
    {
        REQUIRE(Y(i).get(GRB_DoubleAttr_Start) == Catch::Approx(1.0));
        REQUIRE(Y(i).get(GRB_DoubleAttr_X) == Catch::Approx(1.0));
    }
}

/**
 * @test Integration::FullOptimization
 * @brief Verifies full optimization workflow produces valid status
 *
 * @scenario AdvancedBuilder performs complete optimization
 * @given An AdvancedBuilder with full model setup
 * @when Calling optimize()
 * @then Model status indicates feasible solution found
 *
 * @covers ModelBuilder::optimize() full workflow
 */
TEST_CASE("F2: Integration::FullOptimization", "[ModelBuilder][integration][solve]")
{
    AdvancedBuilder B;
    B.optimize();

    int st = B.model().get(GRB_IntAttr_Status);

    REQUIRE(st != GRB_INFEASIBLE);
    REQUIRE(st != GRB_INF_OR_UNBD);
    REQUIRE(st != GRB_UNBOUNDED);
}

/**
 * @test Integration::VariableIteration
 * @brief Verifies variable iteration via forEach
 *
 * @scenario Iterating through 2D variable array
 * @given An AdvancedBuilder with X variables [I,J]
 * @when Calling forEach on variable array
 * @then All elements are visited with correct indices
 *
 * @covers VariableArray::forEach()
 */
TEST_CASE("F3: Integration::VariableIteration", "[ModelBuilder][integration][iteration]")
{
    AdvancedBuilder B;
    B.addVariables();

    auto& X = B.variables()(AdvVars::X);

    int count = 0;
    X.forEach([&](GRBVar& v, const std::vector<int>& idx) {
        REQUIRE(idx.size() == 2);
        REQUIRE(idx[0] < B.I);
        REQUIRE(idx[1] < B.J);
        count++;
        });

    REQUIRE(count == B.I * B.J);
}

// ============================================================================
// SECTION G: LAZY INITIALIZATION AND MODEL ACCESS
// ============================================================================

/**
 * @test LazyInitialization::ModelAccessTriggersInit
 * @brief Verifies model() accessor triggers lazy initialization
 *
 * @scenario Accessing model() before explicit initialize()
 * @given A fresh ModelBuilder instance
 * @when Calling model() without prior initialization
 * @then Model is automatically initialized and accessible
 *
 * @covers ModelBuilder::model(), lazy initialization
 */
TEST_CASE("G1: LazyInitialization::ModelAccessTriggersInit", "[ModelBuilder][initialization]")
{
    EnvTestBuilder builder;

    // model() should trigger initialization
    GRBModel& m = builder.model();

    REQUIRE(builder.envConfigured == 1);

    // Model should be usable
    m.set(GRB_IntParam_OutputFlag, 0);
    int flag = m.get(GRB_IntParam_OutputFlag);
    REQUIRE(flag == 0);
}

/**
 * @test LazyInitialization::InitializeOnlyOnce
 * @brief Verifies initialize() is idempotent
 *
 * @scenario Calling initialize() multiple times
 * @given A ModelBuilder instance
 * @when Calling initialize() repeatedly
 * @then Environment configuration happens only once
 *
 * @covers ModelBuilder::initialize() idempotency
 */
TEST_CASE("G2: LazyInitialization::InitializeOnlyOnce", "[ModelBuilder][initialization]")
{
    EnvTestBuilder builder;

    builder.initialize();
    REQUIRE(builder.envConfigured == 1);

    builder.initialize();
    REQUIRE(builder.envConfigured == 1);

    builder.initialize();
    REQUIRE(builder.envConfigured == 1);
}

/**
 * @test LazyInitialization::OptimizeCallsInitialize
 * @brief Verifies optimize() calls initialize() if needed
 *
 * @scenario Calling optimize() without prior initialization
 * @given A fresh ModelBuilder instance
 * @when Calling optimize()
 * @then Initialization happens automatically before optimization
 *
 * @covers ModelBuilder::optimize() initialization integration
 */
TEST_CASE("G3: LazyInitialization::OptimizeCallsInitialize", "[ModelBuilder][initialization]")
{
    EnvTestBuilder builder;

    REQUIRE(builder.envConfigured == 0);

    builder.optimize();

    REQUIRE(builder.envConfigured == 1);
}

/**
 * @test LazyInitialization::ConstModelAccessor
 * @brief Verifies const model() accessor works correctly
 *
 * @scenario Accessing model via const reference
 * @given An initialized ModelBuilder
 * @when Accessing model() through const reference
 * @then Const model reference is returned
 *
 * @covers ModelBuilder::model() const
 */
TEST_CASE("G4: LazyInitialization::ConstModelAccessor", "[ModelBuilder][initialization][const]")
{
    BasicBuilder builder;
    builder.optimize();

    const BasicBuilder& constRef = builder;

    const GRBModel& m = constRef.model();

    // Should be able to query but not modify
    int status = m.get(GRB_IntAttr_Status);
    REQUIRE(status != GRB_LOADED);  // Model was optimized
}

/**
 * @test LazyInitialization::ConstTableAccessors
 * @brief Verifies const accessors for variables, constraints, store
 *
 * @scenario Accessing tables via const reference
 * @given An initialized ModelBuilder with data
 * @when Accessing tables through const reference
 * @then Const references are returned and accessible
 *
 * @covers ModelBuilder::variables() const, constraints() const, store() const
 */
TEST_CASE("G5: LazyInitialization::ConstTableAccessors", "[ModelBuilder][initialization][const]")
{
    BasicBuilder builder;
    builder.optimize();

    const BasicBuilder& constRef = builder;

    // Const variable access
    const auto& vars = constRef.variables();
    const auto& X = vars(TestVars::X);
    REQUIRE(X.asGroup().dimension() == 1);

    // Const constraint access
    const auto& cons = constRef.constraints();
    const auto& Cap = cons(TestCons::Cap);
    REQUIRE(Cap.asGroup().dimension() == 1);

    // Const store access
    const auto& s = constRef.store();
    (void)s;  // Just verify it compiles
}

// ============================================================================
// SECTION H: EXTERNAL MODEL SUPPORT
// ============================================================================

/**
 * @test ExternalModel::ConstructWithExternalModel
 * @brief Verifies construction with pre-existing GRBModel
 *
 * @scenario ModelBuilder constructed with external model
 * @given A pre-existing GRBModel and GRBEnv
 * @when Constructing ModelBuilder with external model
 * @then ModelBuilder uses external model, no new env created
 *
 * @covers ModelBuilder(GRBModel&) constructor
 */
TEST_CASE("H1: ExternalModel::ConstructWithExternalModel", "[ModelBuilder][external]")
{
    // Create external environment and model
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel externalModel(env);

    // Add a variable to external model
    GRBVar v = externalModel.addVar(0.0, 1.0, 1.0, GRB_BINARY, "ext_var");
    externalModel.update();

    // Create builder with external model
    EnvTestBuilder builder(externalModel);

    // configureEnvironment should NOT be called (already initialized)
    REQUIRE(builder.envConfigured == 0);

    // model() should return the external model
    GRBModel& m = builder.model();
    REQUIRE(m.get(GRB_IntAttr_NumVars) == 1);
}

/**
 * @test ExternalModel::OptimizeWithExternalModel
 * @brief Verifies optimize() works with external model
 *
 * @scenario Optimizing with externally-provided model
 * @given A ModelBuilder constructed with external model
 * @when Calling optimize()
 * @then Hooks are called and optimization uses external model
 *
 * @covers ModelBuilder::optimize() with external model
 */
TEST_CASE("H2: ExternalModel::OptimizeWithExternalModel", "[ModelBuilder][external]")
{
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel externalModel(env);

    BasicBuilder builder(externalModel);

    // Optimize should work
    builder.optimize();

    // Hooks should have been called
    REQUIRE(builder.calls_addVariables == 1);
    REQUIRE(builder.calls_addConstraints == 1);
    REQUIRE(builder.calls_addObjective == 1);

    // Variables should exist in external model
    REQUIRE(externalModel.get(GRB_IntAttr_NumVars) == builder.n);
}

/**
 * @test ExternalModel::NoEnvironmentCreated
 * @brief Verifies no internal environment is created for external model
 *
 * @scenario ModelBuilder with external model checks env ownership
 * @given A ModelBuilder constructed with external model
 * @when Checking internal state
 * @then No internal environment or model is created
 *
 * @covers ModelBuilder external model ownership semantics
 */
TEST_CASE("H3: ExternalModel::NoEnvironmentCreated", "[ModelBuilder][external]")
{
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel externalModel(env);

    EnvTestBuilder builder(externalModel);

    // Environment configuration hook should never be called
    builder.optimize();
    REQUIRE(builder.envConfigured == 0);

    // Model reference should be the external one
    GRBModel& m = builder.model();
    REQUIRE(&m == &externalModel);
}

// ============================================================================
// SECTION I: DATASTORE INTEGRATION
// ============================================================================

/**
 * @class StoreTestBuilder
 * @brief Test fixture for DataStore integration tests
 *
 * @details Builder that uses the store() for parameter passing
 *          and metadata storage across hook invocations.
 */
class StoreTestBuilder : public ModelBuilder<TestVars, TestCons>
{
public:
    using Base = ModelBuilder<TestVars, TestCons>;
    using Base::Base;

    void configureEnvironment(GRBEnv& env) override
    {
        env.set(GRB_IntParam_OutputFlag, 0);
    }

    void addVariables() override
    {
        int n = store()["n"].get<int>();

        auto X = VariableFactory::add(
            model(), GRB_BINARY, 0.0, 1.0, "x", n);

        variables().set(TestVars::X, std::move(X));

        // Store computed metadata
        store()["var_count"] = n;
    }

    void addConstraints() override
    {
        auto& X = variables()(TestVars::X);
        int n = store()["n"].get<int>();

        auto G = ConstraintFactory::add(
            model(), "cap",
            [&](const std::vector<int>& idx)
            {
                return X(idx[0]) <= 1.0;
            },
            n
        );

        constraints().set(TestCons::Cap, std::move(G));
    }

    void addObjective() override
    {
        auto& X = variables()(TestVars::X);
        int n = store()["n"].get<int>();

        GRBLinExpr obj = 0;
        for (int i = 0; i < n; i++)
            obj += X(i);

        model().setObjective(obj, GRB_MAXIMIZE);
    }

    void afterOptimize() override
    {
        store()["solved"] = true;
        store()["obj_value"] = model().get(GRB_DoubleAttr_ObjVal);
    }
};

/**
 * @test DataStoreIntegration::ParameterPassing
 * @brief Verifies DataStore can pass parameters to hooks
 *
 * @scenario Using store() to configure model size
 * @given A StoreTestBuilder with n stored in DataStore
 * @when Calling optimize()
 * @then Hooks read n from store and create correct model
 *
 * @covers ModelBuilder::store(), DataStore parameter passing
 */
TEST_CASE("I1: DataStoreIntegration::ParameterPassing", "[ModelBuilder][store]")
{
    StoreTestBuilder builder;

    // Set parameter via store
    builder.store()["n"] = 5;

    builder.optimize();

    // Verify model was built with correct size
    auto& X = builder.variables()(TestVars::X);
    REQUIRE(X.asGroup().size(0) == 5);
}

/**
 * @test DataStoreIntegration::MetadataStorage
 * @brief Verifies DataStore can store computed metadata
 *
 * @scenario Storing computed values in afterOptimize
 * @given A StoreTestBuilder that stores results
 * @when Calling optimize()
 * @then Computed values are retrievable from store
 *
 * @covers ModelBuilder::store(), DataStore result storage
 */
TEST_CASE("I2: DataStoreIntegration::MetadataStorage", "[ModelBuilder][store]")
{
    StoreTestBuilder builder;
    builder.store()["n"] = 3;

    builder.optimize();

    // Check metadata stored by hooks
    REQUIRE(builder.store()["var_count"].get<int>() == 3);
    REQUIRE(builder.store()["solved"].get<bool>() == true);
    REQUIRE(builder.store()["obj_value"].get<double>() == Catch::Approx(3.0));
}

/**
 * @test DataStoreIntegration::GetOrCompute
 * @brief Verifies DataStore lazy computation works in builder context
 *
 * @scenario Using getOrCompute for lazy initialization
 * @given A builder that uses getOrCompute for defaults
 * @when Accessing values with and without preset
 * @then Lazy computation or preset values are returned correctly
 *
 * @covers DataStore::getOrCompute() integration
 */
TEST_CASE("I3: DataStoreIntegration::GetOrCompute", "[ModelBuilder][store]")
{
    StoreTestBuilder builder;

    // Use getOrCompute for default value
    int n = builder.store()["n"].getOrCompute<int>([] { return 4; });
    REQUIRE(n == 4);

    // Value should now be stored
    REQUIRE(builder.store()["n"].get<int>() == 4);

    // Override and verify
    builder.store()["n"] = 7;
    int n2 = builder.store()["n"].getOrCompute<int>([] { return 99; });
    REQUIRE(n2 == 7);  // Existing value, not computed
}

/**
 * @test DataStoreIntegration::TypeSafety
 * @brief Verifies DataStore type safety in builder context
 *
 * @scenario Storing and retrieving various types
 * @given A ModelBuilder with store
 * @when Storing int, double, string, bool values
 * @then All types are stored and retrieved correctly
 *
 * @covers DataStore type-erased storage
 */
TEST_CASE("I4: DataStoreIntegration::TypeSafety", "[ModelBuilder][store]")
{
    BasicBuilder builder;

    builder.store()["int_val"] = 42;
    builder.store()["double_val"] = 3.14159;
    builder.store()["string_val"] = std::string("test");
    builder.store()["bool_val"] = true;

    REQUIRE(builder.store()["int_val"].get<int>() == 42);
    REQUIRE(builder.store()["double_val"].get<double>() == Catch::Approx(3.14159));
    REQUIRE(builder.store()["string_val"].get<std::string>() == "test");
    REQUIRE(builder.store()["bool_val"].get<bool>() == true);
}

// ============================================================================
// SECTION J: EDGE CASES AND ERROR CONDITIONS
// ============================================================================

/**
 * @test EdgeCases::MultipleOptimizeCalls
 * @brief Verifies behavior when optimize() is called multiple times
 *
 * @scenario Calling optimize() repeatedly on same builder
 * @given A BasicBuilder instance
 * @when Calling optimize() multiple times
 * @then Hooks are called each time but init only once
 *
 * @covers ModelBuilder::optimize() repeated calls
 */
TEST_CASE("J1: EdgeCases::MultipleOptimizeCalls", "[ModelBuilder][edge]")
{
    EnvTestBuilder builder;

    builder.optimize();
    REQUIRE(builder.envConfigured == 1);

    // Note: This will add duplicate variables/constraints
    // Real usage would need to handle this, but we test the lifecycle
    builder.optimize();
    REQUIRE(builder.envConfigured == 1);  // Still only configured once
}

/**
 * @test EdgeCases::EmptyHooks
 * @brief Verifies builder works with no-op hooks
 *
 * @scenario ModelBuilder with default (empty) hook implementations
 * @given A minimal builder with no overrides
 * @when Calling optimize()
 * @then Optimization completes without error
 *
 * @covers ModelBuilder default hook behavior
 */
TEST_CASE("J2: EdgeCases::EmptyHooks", "[ModelBuilder][edge]")
{
    // Minimal builder with only env config (to silence output)
    class MinimalBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override
        {
            env.set(GRB_IntParam_OutputFlag, 0);
        }
    };

    MinimalBuilder builder;

    // Should not throw - empty model is valid
    REQUIRE_NOTHROW(builder.optimize());
}

/**
 * @test EdgeCases::InitializeThenOptimize
 * @brief Verifies explicit initialize() followed by optimize()
 *
 * @scenario Calling initialize() before optimize()
 * @given A ModelBuilder instance
 * @when Calling initialize() then optimize()
 * @then Both complete successfully with correct hook counts
 *
 * @covers ModelBuilder::initialize() then optimize() sequence
 */
TEST_CASE("J3: EdgeCases::InitializeThenOptimize", "[ModelBuilder][edge]")
{
    BasicBuilder builder;

    builder.initialize();
    REQUIRE(builder.calls_addVariables == 0);

    builder.optimize();
    REQUIRE(builder.calls_addVariables == 1);
    REQUIRE(builder.calls_afterOptimize == 1);
}

/**
 * @test EdgeCases::ModelAccessBeforeVariables
 * @brief Verifies model can be accessed before hooks run
 *
 * @scenario Accessing model() before optimize()
 * @given A fresh ModelBuilder
 * @when Accessing model() and setting parameters manually
 * @then Model is accessible and configurable
 *
 * @covers ModelBuilder::model() early access
 */
TEST_CASE("J4: EdgeCases::ModelAccessBeforeVariables", "[ModelBuilder][edge]")
{
    BasicBuilder builder;

    // Access model before optimize
    GRBModel& m = builder.model();

    // Can set parameters directly
    m.set(GRB_DoubleParam_TimeLimit, 100.0);
    REQUIRE(m.get(GRB_DoubleParam_TimeLimit) == Catch::Approx(100.0));

    // Then optimize (which may override TimeLimit in addParameters)
    builder.optimize();

    // addParameters sets TimeLimit to 10.0
    REQUIRE(m.get(GRB_DoubleParam_TimeLimit) == Catch::Approx(10.0));
}

/**
 * @test EdgeCases::LargeModelCreation
 * @brief Verifies builder handles larger models
 *
 * @scenario Creating a model with many variables and constraints
 * @given A builder configured for larger dimensions
 * @when Building and optimizing
 * @then Model is created correctly without issues
 *
 * @covers ModelBuilder scalability
 */
TEST_CASE("J5: EdgeCases::LargeModelCreation", "[ModelBuilder][edge][.slow]")
{
    AdvancedBuilder builder;
    builder.I = 50;
    builder.J = 50;

    builder.optimize();

    auto& X = builder.variables()(AdvVars::X);
    REQUIRE(X.asGroup().dimension() == 2);
    REQUIRE(X.asGroup().size(0) == 50);
    REQUIRE(X.asGroup().size(1) == 50);

    // Total variables: 50*50 (X) + 50 (Y) = 2550
    REQUIRE(builder.model().get(GRB_IntAttr_NumVars) == 2550);
}

// ============================================================================
// SECTION K: SOLUTION DIAGNOSTICS UTILITIES
// ============================================================================

/**
 * @test Diagnostics::StatusAfterOptimal
 * @brief Verifies status() and isOptimal() for optimal solution
 *
 * @scenario A feasible model is optimized to optimality
 * @given A BasicBuilder that produces an optimal solution
 * @when Querying status diagnostics
 * @then status() returns GRB_OPTIMAL and isOptimal() returns true
 *
 * @covers ModelBuilder::status(), isOptimal()
 */
TEST_CASE("K1: Diagnostics::StatusAfterOptimal", "[ModelBuilder][diagnostics]")
{
    BasicBuilder builder;
    builder.optimize();

    REQUIRE(builder.status() == GRB_OPTIMAL);
    REQUIRE(builder.isOptimal() == true);
    REQUIRE(builder.hasSolution() == true);
    REQUIRE(builder.isInfeasible() == false);
    REQUIRE(builder.isUnbounded() == false);
}

/**
 * @test Diagnostics::ObjValAfterOptimization
 * @brief Verifies objVal() returns correct objective value
 *
 * @scenario A model is optimized and objective queried
 * @given A BasicBuilder maximizing sum of binary variables
 * @when Querying objVal() after optimization
 * @then Correct objective value is returned
 *
 * @covers ModelBuilder::objVal()
 */
TEST_CASE("K2: Diagnostics::ObjValAfterOptimization", "[ModelBuilder][diagnostics]")
{
    BasicBuilder builder;
    builder.n = 5;
    builder.optimize();

    // Maximizing sum of 5 binary variables with x <= 1 constraints
    // Optimal: all x = 1, objective = 5
    REQUIRE(builder.objVal() == Catch::Approx(5.0));
}

/**
 * @test Diagnostics::ObjBoundForLP
 * @brief Verifies objBound() equals objVal() for LP at optimality
 *
 * @scenario An LP model is solved to optimality
 * @given A builder with continuous variables
 * @when Querying objBound() after optimization
 * @then objBound equals objVal for optimal LP
 *
 * @covers ModelBuilder::objBound()
 */
TEST_CASE("K3: Diagnostics::ObjBoundForLP", "[ModelBuilder][diagnostics]")
{
    class LPBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addVariables() override {
            auto X = VariableFactory::add(model(), GRB_CONTINUOUS, 0, 10, "x", 3);
            variables().set(TestVars::X, std::move(X));
        }

        void addConstraints() override {
            auto& X = variables()(TestVars::X);
            // x0 + x1 + x2 <= 15
            model().addConstr(X(0) + X(1) + X(2) <= 15);
        }

        void addObjective() override {
            auto& X = variables()(TestVars::X);
            model().setObjective(X(0) + X(1) + X(2), GRB_MAXIMIZE);
        }
    };

    LPBuilder builder;
    builder.optimize();

    REQUIRE(builder.isOptimal());
    REQUIRE(builder.objVal() == Catch::Approx(15.0));
    REQUIRE(builder.objBound() == Catch::Approx(builder.objVal()));
}

/**
 * @test Diagnostics::RuntimeIsPositive
 * @brief Verifies runtime() returns positive value after optimization
 *
 * @scenario A model is optimized
 * @given Any valid ModelBuilder
 * @when Querying runtime() after optimization
 * @then A non-negative runtime value is returned
 *
 * @covers ModelBuilder::runtime()
 */
TEST_CASE("K4: Diagnostics::RuntimeIsPositive", "[ModelBuilder][diagnostics]")
{
    BasicBuilder builder;
    builder.optimize();

    double rt = builder.runtime();
    REQUIRE(rt >= 0.0);
}

/**
 * @test Diagnostics::SolutionCountForMIP
 * @brief Verifies solutionCount() for MIP model
 *
 * @scenario A MIP model is solved
 * @given A BasicBuilder with binary variables
 * @when Querying solutionCount() after optimization
 * @then At least one solution is found
 *
 * @covers ModelBuilder::solutionCount()
 */
TEST_CASE("K5: Diagnostics::SolutionCountForMIP", "[ModelBuilder][diagnostics]")
{
    BasicBuilder builder;
    builder.optimize();

    int solCount = builder.solutionCount();
    REQUIRE(solCount >= 1);
}

/**
 * @test Diagnostics::IterCountForLP
 * @brief Verifies iterCount() returns iteration count
 *
 * @scenario An LP is solved
 * @given A model with continuous variables
 * @when Querying iterCount() after optimization
 * @then A non-negative iteration count is returned
 *
 * @covers ModelBuilder::iterCount()
 */
TEST_CASE("K6: Diagnostics::IterCountForLP", "[ModelBuilder][diagnostics]")
{
    class SimpleLPBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addVariables() override {
            auto X = VariableFactory::add(model(), GRB_CONTINUOUS, 0, 100, "x", 2);
            variables().set(TestVars::X, std::move(X));
        }

        void addConstraints() override {
            auto& X = variables()(TestVars::X);
            model().addConstr(X(0) + X(1) <= 50);
        }

        void addObjective() override {
            auto& X = variables()(TestVars::X);
            model().setObjective(X(0) + 2 * X(1), GRB_MAXIMIZE);
        }
    };

    SimpleLPBuilder builder;
    builder.optimize();

    double iters = builder.iterCount();
    REQUIRE(iters >= 0.0);
}

/**
 * @test Diagnostics::NodeCountForMIP
 * @brief Verifies nodeCount() for MIP model
 *
 * @scenario A MIP model is solved
 * @given A BasicBuilder with binary variables
 * @when Querying nodeCount() after optimization
 * @then A non-negative node count is returned
 *
 * @covers ModelBuilder::nodeCount()
 */
TEST_CASE("K7: Diagnostics::NodeCountForMIP", "[ModelBuilder][diagnostics]")
{
    BasicBuilder builder;
    builder.optimize();

    double nodes = builder.nodeCount();
    REQUIRE(nodes >= 0.0);
}

/**
 * @test Diagnostics::MipGapAtOptimality
 * @brief Verifies mipGap() is zero or near-zero at optimality
 *
 * @scenario A MIP is solved to optimality
 * @given A BasicBuilder solved to optimality
 * @when Querying mipGap()
 * @then Gap is zero (or within tolerance)
 *
 * @covers ModelBuilder::mipGap()
 */
TEST_CASE("K8: Diagnostics::MipGapAtOptimality", "[ModelBuilder][diagnostics]")
{
    BasicBuilder builder;
    builder.optimize();

    REQUIRE(builder.isOptimal());
    REQUIRE(builder.mipGap() == Catch::Approx(0.0).margin(1e-6));
}

/**
 * @test Diagnostics::InfeasibleModel
 * @brief Verifies diagnostics for infeasible model
 *
 * @scenario An infeasible model is optimized
 * @given A builder that creates contradictory constraints
 * @when Querying diagnostics after optimization
 * @then isInfeasible() returns true, hasSolution() returns false
 *
 * @covers ModelBuilder::isInfeasible(), hasSolution()
 */
TEST_CASE("K9: Diagnostics::InfeasibleModel", "[ModelBuilder][diagnostics]")
{
    class InfeasibleBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addVariables() override {
            auto X = VariableFactory::add(model(), GRB_CONTINUOUS, 0, 10, "x", 1);
            variables().set(TestVars::X, std::move(X));
        }

        void addConstraints() override {
            auto& X = variables()(TestVars::X);
            // Contradictory: x >= 5 AND x <= 3
            model().addConstr(X(0) >= 5);
            model().addConstr(X(0) <= 3);
        }

        void addObjective() override {
            auto& X = variables()(TestVars::X);
            model().setObjective(GRBLinExpr(X(0)), GRB_MINIMIZE);
        }
    };

    InfeasibleBuilder builder;
    builder.optimize();

    REQUIRE(builder.isInfeasible() == true);
    REQUIRE(builder.isOptimal() == false);
    REQUIRE(builder.hasSolution() == false);
}

/**
 * @test Diagnostics::HasSolutionWithTimeLimit
 * @brief Verifies hasSolution() correctly handles time-limited solves
 *
 * @scenario A model is solved with very short time limit
 * @given A builder that may or may not find solution in time
 * @when Checking hasSolution() after time limit
 * @then hasSolution() returns true only if solution was found
 *
 * @covers ModelBuilder::hasSolution() with TIME_LIMIT status
 */
TEST_CASE("K10: Diagnostics::HasSolutionWithTimeLimit", "[ModelBuilder][diagnostics]")
{
    // This test uses a model that should find a solution quickly
    // even with a short time limit
    BasicBuilder builder;
    builder.n = 2;  // Very small model
    builder.optimize();

    // Should have found optimal solution
    if (builder.status() == GRB_OPTIMAL) {
        REQUIRE(builder.hasSolution() == true);
    }
    else if (builder.status() == GRB_TIME_LIMIT) {
        // hasSolution should check solutionCount
        REQUIRE(builder.hasSolution() == (builder.solutionCount() > 0));
    }
}

// ============================================================================
// SECTION L: OBJECTIVE HELPERS (MINIMIZE/MAXIMIZE)
// ============================================================================

/**
 * @test ObjectiveHelpers::MinimizeBasic
 * @brief Verifies minimize() sets correct objective sense
 *
 * @scenario A minimization objective is set using minimize()
 * @given A builder with continuous variables
 * @when Using minimize() to set objective
 * @then Model minimizes the expression correctly
 *
 * @covers ModelBuilder::minimize()
 */
TEST_CASE("L1: ObjectiveHelpers::MinimizeBasic", "[ModelBuilder][objective]")
{
    class MinBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addVariables() override {
            auto X = VariableFactory::add(model(), GRB_CONTINUOUS, 0, 10, "x", 2);
            variables().set(TestVars::X, std::move(X));
        }

        void addConstraints() override {
            auto& X = variables()(TestVars::X);
            model().addConstr(X(0) + X(1) >= 5);
        }

        void addObjective() override {
            auto& X = variables()(TestVars::X);
            minimize(X(0) + X(1));
        }
    };

    MinBuilder builder;
    builder.optimize();

    REQUIRE(builder.isOptimal());
    REQUIRE(builder.objVal() == Catch::Approx(5.0));
}

/**
 * @test ObjectiveHelpers::MaximizeBasic
 * @brief Verifies maximize() sets correct objective sense
 *
 * @scenario A maximization objective is set using maximize()
 * @given A builder with continuous variables
 * @when Using maximize() to set objective
 * @then Model maximizes the expression correctly
 *
 * @covers ModelBuilder::maximize()
 */
TEST_CASE("L2: ObjectiveHelpers::MaximizeBasic", "[ModelBuilder][objective]")
{
    class MaxBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addVariables() override {
            auto X = VariableFactory::add(model(), GRB_CONTINUOUS, 0, 10, "x", 2);
            variables().set(TestVars::X, std::move(X));
        }

        void addConstraints() override {
            auto& X = variables()(TestVars::X);
            model().addConstr(X(0) + X(1) <= 15);
        }

        void addObjective() override {
            auto& X = variables()(TestVars::X);
            maximize(X(0) + X(1));
        }
    };

    MaxBuilder builder;
    builder.optimize();

    REQUIRE(builder.isOptimal());
    REQUIRE(builder.objVal() == Catch::Approx(15.0));
}

/**
 * @test ObjectiveHelpers::MinimizeWithSum
 * @brief Verifies minimize() works with sum() expressions
 *
 * @scenario A minimization objective is set using minimize() with sum()
 * @given A builder with array variables
 * @when Using minimize(sum(...)) pattern
 * @then Model minimizes the sum correctly
 *
 * @covers ModelBuilder::minimize() with sum() expression
 */
TEST_CASE("L3: ObjectiveHelpers::MinimizeWithSum", "[ModelBuilder][objective]")
{
    class SumMinBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        int n = 4;

        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addVariables() override {
            auto X = VariableFactory::add(model(), GRB_CONTINUOUS, 1, 10, "x", n);
            variables().set(TestVars::X, std::move(X));
        }

        void addObjective() override {
            auto& X = variables()(TestVars::X);
            // Minimize sum of all variables (each has lb=1)
            minimize(sum(range(0, n), [&](int i) { return X(i); }));
        }
    };

    SumMinBuilder builder;
    builder.optimize();

    REQUIRE(builder.isOptimal());
    // All variables at lower bound = 1, sum = 4
    REQUIRE(builder.objVal() == Catch::Approx(4.0));
}

/**
 * @test ObjectiveHelpers::MaximizeWithSum
 * @brief Verifies maximize() works with sum() expressions
 *
 * @scenario A maximization objective is set using maximize() with sum()
 * @given A builder with array variables
 * @when Using maximize(sum(...)) pattern
 * @then Model maximizes the sum correctly
 *
 * @covers ModelBuilder::maximize() with sum() expression
 */
TEST_CASE("L4: ObjectiveHelpers::MaximizeWithSum", "[ModelBuilder][objective]")
{
    class SumMaxBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        int n = 4;

        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addVariables() override {
            auto X = VariableFactory::add(model(), GRB_CONTINUOUS, 0, 5, "x", n);
            variables().set(TestVars::X, std::move(X));
        }

        void addObjective() override {
            auto& X = variables()(TestVars::X);
            // Maximize sum of all variables (each has ub=5)
            maximize(sum(range(0, n), [&](int i) { return X(i); }));
        }
    };

    SumMaxBuilder builder;
    builder.optimize();

    REQUIRE(builder.isOptimal());
    // All variables at upper bound = 5, sum = 20
    REQUIRE(builder.objVal() == Catch::Approx(20.0));
}

/**
 * @test ObjectiveHelpers::MinimizeWithCoefficients
 * @brief Verifies minimize() works with weighted expressions
 *
 * @scenario A weighted minimization objective
 * @given A builder with cost coefficients
 * @when Using minimize() with coefficient-weighted variables
 * @then Correct weighted minimum is found
 *
 * @covers ModelBuilder::minimize() with coefficients
 */
TEST_CASE("L5: ObjectiveHelpers::MinimizeWithCoefficients", "[ModelBuilder][objective]")
{
    class WeightedMinBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addVariables() override {
            auto X = VariableFactory::add(model(), GRB_CONTINUOUS, 0, 10, "x", 3);
            variables().set(TestVars::X, std::move(X));
        }

        void addConstraints() override {
            auto& X = variables()(TestVars::X);
            // Need at least 10 units total
            model().addConstr(X(0) + X(1) + X(2) >= 10);
        }

        void addObjective() override {
            auto& X = variables()(TestVars::X);
            // Costs: 1, 2, 3 per unit
            minimize(1.0 * X(0) + 2.0 * X(1) + 3.0 * X(2));
        }
    };

    WeightedMinBuilder builder;
    builder.optimize();

    REQUIRE(builder.isOptimal());
    // Cheapest to use X(0) at cost 1, so obj = 10
    REQUIRE(builder.objVal() == Catch::Approx(10.0));
}

// ============================================================================
// SECTION M: PARAMETER PRESETS AND CONVENIENCE SETTERS
// ============================================================================

/**
 * @test ParameterConvenience::TimeLimitSetter
 * @brief Verifies timeLimit() sets parameter and tracks in store
 *
 * @scenario Setting time limit via convenience method
 * @given A ModelBuilder instance
 * @when Calling timeLimit(30.0)
 * @then Parameter is set on model and tracked in store
 *
 * @covers ModelBuilder::timeLimit()
 */
TEST_CASE("M1: ParameterConvenience::TimeLimitSetter", "[ModelBuilder][params]")
{
    class ParamBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addParameters() override {
            timeLimit(30.0);
        }
    };

    ParamBuilder builder;
    builder.optimize();

    // Check model parameter
    REQUIRE(builder.model().get(GRB_DoubleParam_TimeLimit) == Catch::Approx(30.0));

    // Check store tracking
    REQUIRE(builder.store()["param:TimeLimit"].get<double>() == Catch::Approx(30.0));
}

/**
 * @test ParameterConvenience::MipGapLimitSetter
 * @brief Verifies mipGapLimit() sets parameter and tracks in store
 *
 * @scenario Setting MIP gap via convenience method
 * @given A ModelBuilder instance
 * @when Calling mipGapLimit(0.02)
 * @then Parameter is set on model and tracked in store
 *
 * @covers ModelBuilder::mipGapLimit()
 */
TEST_CASE("M2: ParameterConvenience::MipGapLimitSetter", "[ModelBuilder][params]")
{
    class ParamBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addParameters() override {
            mipGapLimit(0.02);
        }
    };

    ParamBuilder builder;
    builder.optimize();

    REQUIRE(builder.model().get(GRB_DoubleParam_MIPGap) == Catch::Approx(0.02));
    REQUIRE(builder.store()["param:MIPGap"].get<double>() == Catch::Approx(0.02));
}

/**
 * @test ParameterConvenience::ThreadsSetter
 * @brief Verifies threads() sets parameter and tracks in store
 *
 * @scenario Setting thread count via convenience method
 * @given A ModelBuilder instance
 * @when Calling threads(2)
 * @then Parameter is set on model and tracked in store
 *
 * @covers ModelBuilder::threads()
 */
TEST_CASE("M3: ParameterConvenience::ThreadsSetter", "[ModelBuilder][params]")
{
    class ParamBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addParameters() override {
            threads(2);
        }
    };

    ParamBuilder builder;
    builder.optimize();

    REQUIRE(builder.model().get(GRB_IntParam_Threads) == 2);
    REQUIRE(builder.store()["param:Threads"].get<int>() == 2);
}

/**
 * @test ParameterConvenience::QuietSetter
 * @brief Verifies quiet() suppresses output and tracks in store
 *
 * @scenario Suppressing output via convenience method
 * @given A ModelBuilder instance
 * @when Calling quiet()
 * @then OutputFlag is set to 0 and tracked in store
 *
 * @covers ModelBuilder::quiet()
 */
TEST_CASE("M4: ParameterConvenience::QuietSetter", "[ModelBuilder][params]")
{
    class ParamBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void addParameters() override {
            quiet();
        }
    };

    ParamBuilder builder;
    builder.optimize();

    REQUIRE(builder.model().get(GRB_IntParam_OutputFlag) == 0);
    REQUIRE(builder.store()["param:OutputFlag"].get<int>() == 0);
}

/**
 * @test ParameterConvenience::VerboseSetter
 * @brief Verifies verbose() enables output and tracks in store
 *
 * @scenario Enabling output via convenience method
 * @given A ModelBuilder instance
 * @when Calling verbose()
 * @then OutputFlag is set to 1 and tracked in store
 *
 * @covers ModelBuilder::verbose()
 */
TEST_CASE("M5: ParameterConvenience::VerboseSetter", "[ModelBuilder][params]")
{
    class ParamBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);  // Start quiet
        }

        void addParameters() override {
            verbose();
        }
    };

    ParamBuilder builder;
    builder.initialize();
    builder.addParameters();

    REQUIRE(builder.model().get(GRB_IntParam_OutputFlag) == 1);
    REQUIRE(builder.store()["param:OutputFlag"].get<int>() == 1);
}

/**
 * @test ParameterPresets::FastPreset
 * @brief Verifies Preset::Fast applies correct configuration
 *
 * @scenario Applying Fast preset
 * @given A ModelBuilder instance
 * @when Calling applyPreset(Preset::Fast)
 * @then TimeLimit=60, MIPGap=5%, and preset name is tracked
 *
 * @covers ModelBuilder::applyPreset(Preset::Fast)
 */
TEST_CASE("M6: ParameterPresets::FastPreset", "[ModelBuilder][params][presets]")
{
    class PresetBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addParameters() override {
            applyPreset(Preset::Fast);
        }
    };

    PresetBuilder builder;
    builder.optimize();

    REQUIRE(builder.model().get(GRB_DoubleParam_TimeLimit) == Catch::Approx(60.0));
    REQUIRE(builder.model().get(GRB_DoubleParam_MIPGap) == Catch::Approx(0.05));
    REQUIRE(builder.store()["param:Preset"].get<std::string>() == "Fast");
}

/**
 * @test ParameterPresets::AccuratePreset
 * @brief Verifies Preset::Accurate applies correct configuration
 *
 * @scenario Applying Accurate preset
 * @given A ModelBuilder instance
 * @when Calling applyPreset(Preset::Accurate)
 * @then TimeLimit=3600, MIPGap=0.01%, and preset name is tracked
 *
 * @covers ModelBuilder::applyPreset(Preset::Accurate)
 */
TEST_CASE("M7: ParameterPresets::AccuratePreset", "[ModelBuilder][params][presets]")
{
    class PresetBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addParameters() override {
            applyPreset(Preset::Accurate);
        }
    };

    PresetBuilder builder;
    builder.optimize();

    REQUIRE(builder.model().get(GRB_DoubleParam_TimeLimit) == Catch::Approx(3600.0));
    REQUIRE(builder.model().get(GRB_DoubleParam_MIPGap) == Catch::Approx(0.0001));
    REQUIRE(builder.store()["param:Preset"].get<std::string>() == "Accurate");
}

/**
 * @test ParameterPresets::FeasibilityPreset
 * @brief Verifies Preset::Feasibility sets MIPFocus=1
 *
 * @scenario Applying Feasibility preset
 * @given A ModelBuilder instance
 * @when Calling applyPreset(Preset::Feasibility)
 * @then MIPFocus=1 and preset name is tracked
 *
 * @covers ModelBuilder::applyPreset(Preset::Feasibility)
 */
TEST_CASE("M8: ParameterPresets::FeasibilityPreset", "[ModelBuilder][params][presets]")
{
    class PresetBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addParameters() override {
            applyPreset(Preset::Feasibility);
        }
    };

    PresetBuilder builder;
    builder.optimize();

    REQUIRE(builder.model().get(GRB_IntParam_MIPFocus) == 1);
    REQUIRE(builder.store()["param:Preset"].get<std::string>() == "Feasibility");
}

/**
 * @test ParameterPresets::DebugPreset
 * @brief Verifies Preset::Debug enables verbose output and disables presolve
 *
 * @scenario Applying Debug preset
 * @given A ModelBuilder instance
 * @when Calling applyPreset(Preset::Debug)
 * @then OutputFlag=1, Presolve=0, and preset name is tracked
 *
 * @covers ModelBuilder::applyPreset(Preset::Debug)
 */
TEST_CASE("M9: ParameterPresets::DebugPreset", "[ModelBuilder][params][presets]")
{
    class PresetBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void addParameters() override {
            applyPreset(Preset::Debug);
        }
    };

    PresetBuilder builder;
    builder.initialize();
    builder.addParameters();

    REQUIRE(builder.model().get(GRB_IntParam_OutputFlag) == 1);
    REQUIRE(builder.model().get(GRB_IntParam_Presolve) == 0);
    REQUIRE(builder.store()["param:Preset"].get<std::string>() == "Debug");
}

/**
 * @test ParameterTracking::MultipleParamsTracked
 * @brief Verifies multiple convenience setters track all values
 *
 * @scenario Setting multiple parameters via convenience methods
 * @given A ModelBuilder instance
 * @when Calling multiple parameter setters
 * @then All values are tracked in store for diagnostics
 *
 * @covers Parameter tracking in store()
 */
TEST_CASE("M10: ParameterTracking::MultipleParamsTracked", "[ModelBuilder][params]")
{
    class ParamBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addParameters() override {
            timeLimit(120.0);
            mipGapLimit(0.03);
            threads(4);
            mipFocus(2);
        }
    };

    ParamBuilder builder;
    builder.optimize();

    // All parameters should be tracked
    REQUIRE(builder.store()["param:TimeLimit"].get<double>() == Catch::Approx(120.0));
    REQUIRE(builder.store()["param:MIPGap"].get<double>() == Catch::Approx(0.03));
    REQUIRE(builder.store()["param:Threads"].get<int>() == 4);
    REQUIRE(builder.store()["param:MIPFocus"].get<int>() == 2);
}

/**
 * @test ParameterPresets::PresetThenOverride
 * @brief Verifies parameters can be overridden after preset
 *
 * @scenario Applying preset then overriding specific values
 * @given A ModelBuilder that applies preset then customizes
 * @when Calling applyPreset(Fast) then timeLimit(300)
 * @then Override takes effect, tracking reflects final value
 *
 * @covers Preset + override workflow
 */
TEST_CASE("M11: ParameterPresets::PresetThenOverride", "[ModelBuilder][params][presets]")
{
    class PresetBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addParameters() override {
            applyPreset(Preset::Fast);  // TimeLimit=60
            timeLimit(300.0);           // Override to 300
        }
    };

    PresetBuilder builder;
    builder.optimize();

    // Override should take effect
    REQUIRE(builder.model().get(GRB_DoubleParam_TimeLimit) == Catch::Approx(300.0));
    REQUIRE(builder.store()["param:TimeLimit"].get<double>() == Catch::Approx(300.0));

    // Preset name still tracked
    REQUIRE(builder.store()["param:Preset"].get<std::string>() == "Fast");
}

/**
 * @test ParameterTracking::SetParamWithExplicitName
 * @brief Verifies 3-argument setParam() tracks with custom name
 *
 * @scenario Setting uncommon parameter with explicit tracking name
 * @given A ModelBuilder instance
 * @when Calling setParam(param, value, "Name")
 * @then Parameter is set and tracked under "param:Name"
 *
 * @covers ModelBuilder::setParam(p, value, name) overload
 */
TEST_CASE("M12: ParameterTracking::SetParamWithExplicitName", "[ModelBuilder][params]")
{
    class ParamBuilder : public ModelBuilder<TestVars, TestCons>
    {
    public:
        void configureEnvironment(GRBEnv& env) override {
            env.set(GRB_IntParam_OutputFlag, 0);
        }

        void addParameters() override {
            // Use 3-arg setParam for less common parameters
            setParam(GRB_DoubleParam_Heuristics, 0.25, "Heuristics");
            setParam(GRB_IntParam_Cuts, 2, "Cuts");
        }
    };

    ParamBuilder builder;
    builder.optimize();

    // Check model parameters
    REQUIRE(builder.model().get(GRB_DoubleParam_Heuristics) == Catch::Approx(0.25));
    REQUIRE(builder.model().get(GRB_IntParam_Cuts) == 2);

    // Check store tracking
    REQUIRE(builder.store()["param:Heuristics"].get<double>() == Catch::Approx(0.25));
    REQUIRE(builder.store()["param:Cuts"].get<int>() == 2);
}
