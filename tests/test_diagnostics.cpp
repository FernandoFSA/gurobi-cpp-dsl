/*
===============================================================================
TEST DIAGNOSTICS — Comprehensive tests for diagnostics.h
===============================================================================

OVERVIEW
--------
Validates the diagnostic utilities for Gurobi model analysis including:
- Status string conversion
- Model statistics computation
- IIS (Irreducible Inconsistent Subsystem) extraction
- Solution quality metrics

TEST ORGANIZATION
-----------------
• Section A: Status string conversion
• Section B: Model statistics
• Section C: IIS computation
• Section D: Solution quality metrics
• Section E: Convenience functions

DEPENDENCIES
------------
• Catch2 v3.0+ - Test framework
• diagnostics.h - System under test
• model_builder.h - For creating test models
• Gurobi C++ API - Solver backend

===============================================================================
*/

#include "catch_amalgamated.hpp"

#include <gurobi_dsl/diagnostics.h>
#include <gurobi_dsl/model_builder.h>
#include <gurobi_dsl/variables.h>
#include <gurobi_dsl/expressions.h>
#include <gurobi_dsl/indexing.h>

using namespace dsl;

// ============================================================================
// TEST UTILITIES
// ============================================================================

DECLARE_ENUM_WITH_COUNT(DiagVars, X, Y);
DECLARE_ENUM_WITH_COUNT(DiagCons, C1, C2);

/**
 * @brief Simple builder for diagnostic tests
 */
class DiagBuilder : public ModelBuilder<DiagVars, DiagCons>
{
public:
    int numBinary = 3;
    int numContinuous = 2;
    bool makeInfeasible = false;

    void configureEnvironment(GRBEnv& env) override {
        env.set(GRB_IntParam_OutputFlag, 0);
    }

    void addVariables() override {
        auto X = VariableFactory::add(model(), GRB_BINARY, 0, 1, "x", numBinary);
        auto Y = VariableFactory::add(model(), GRB_CONTINUOUS, 0, 10, "y", numContinuous);
        
        variables().set(DiagVars::X, std::move(X));
        variables().set(DiagVars::Y, std::move(Y));
    }

    void addConstraints() override {
        auto& X = variables()(DiagVars::X);
        auto& Y = variables()(DiagVars::Y);

        // Simple constraint
        model().addConstr(X(0) + X(1) + X(2) <= 2, "sum_x");
        model().addConstr(Y(0) + Y(1) <= 15, "sum_y");

        if (makeInfeasible) {
            // Contradictory constraints
            model().addConstr(X(0) >= 1, "x0_ge_1");
            model().addConstr(X(0) <= 0, "x0_le_0");
        }
    }

    void addObjective() override {
        auto& X = variables()(DiagVars::X);
        auto& Y = variables()(DiagVars::Y);
        
        maximize(X(0) + X(1) + X(2) + Y(0) + Y(1));
    }
};

// ============================================================================
// SECTION A: STATUS STRING CONVERSION
// ============================================================================

/**
 * @test StatusString::OptimalStatus
 * @brief Verifies statusString returns correct string for OPTIMAL
 */
TEST_CASE("A1: StatusString::OptimalStatus", "[diagnostics][status]")
{
    REQUIRE(statusString(GRB_OPTIMAL) == "OPTIMAL");
}

/**
 * @test StatusString::InfeasibleStatus
 * @brief Verifies statusString returns correct string for INFEASIBLE
 */
TEST_CASE("A2: StatusString::InfeasibleStatus", "[diagnostics][status]")
{
    REQUIRE(statusString(GRB_INFEASIBLE) == "INFEASIBLE");
}

/**
 * @test StatusString::AllKnownStatuses
 * @brief Verifies statusString handles all known Gurobi statuses
 */
TEST_CASE("A3: StatusString::AllKnownStatuses", "[diagnostics][status]")
{
    REQUIRE(statusString(GRB_LOADED) == "LOADED");
    REQUIRE(statusString(GRB_OPTIMAL) == "OPTIMAL");
    REQUIRE(statusString(GRB_INFEASIBLE) == "INFEASIBLE");
    REQUIRE(statusString(GRB_INF_OR_UNBD) == "INF_OR_UNBD");
    REQUIRE(statusString(GRB_UNBOUNDED) == "UNBOUNDED");
    REQUIRE(statusString(GRB_CUTOFF) == "CUTOFF");
    REQUIRE(statusString(GRB_ITERATION_LIMIT) == "ITERATION_LIMIT");
    REQUIRE(statusString(GRB_NODE_LIMIT) == "NODE_LIMIT");
    REQUIRE(statusString(GRB_TIME_LIMIT) == "TIME_LIMIT");
    REQUIRE(statusString(GRB_SOLUTION_LIMIT) == "SOLUTION_LIMIT");
    REQUIRE(statusString(GRB_INTERRUPTED) == "INTERRUPTED");
    REQUIRE(statusString(GRB_NUMERIC) == "NUMERIC");
    REQUIRE(statusString(GRB_SUBOPTIMAL) == "SUBOPTIMAL");
    REQUIRE(statusString(GRB_INPROGRESS) == "INPROGRESS");
    REQUIRE(statusString(GRB_USER_OBJ_LIMIT) == "USER_OBJ_LIMIT");
}

/**
 * @test StatusString::UnknownStatus
 * @brief Verifies statusString handles unknown status codes gracefully
 */
TEST_CASE("A4: StatusString::UnknownStatus", "[diagnostics][status]")
{
    std::string result = statusString(999);
    REQUIRE(result.find("UNKNOWN") != std::string::npos);
    REQUIRE(result.find("999") != std::string::npos);
}

/**
 * @test StatusString::IntegrationWithBuilder
 * @brief Verifies statusString works with ModelBuilder status
 */
TEST_CASE("A5: StatusString::IntegrationWithBuilder", "[diagnostics][status]")
{
    DiagBuilder builder;
    builder.optimize();
    
    std::string status = statusString(builder.status());
    REQUIRE(status == "OPTIMAL");
}

// ============================================================================
// SECTION B: MODEL STATISTICS
// ============================================================================

/**
 * @test ModelStatistics::BasicCounts
 * @brief Verifies computeStatistics returns correct variable counts
 */
TEST_CASE("B1: ModelStatistics::BasicCounts", "[diagnostics][statistics]")
{
    DiagBuilder builder;
    builder.numBinary = 5;
    builder.numContinuous = 3;
    builder.addVariables();
    builder.addConstraints();
    builder.model().update();  // Ensure attributes are available
    
    auto stats = computeStatistics(builder.model());
    
    REQUIRE(stats.numVars == 8);  // 5 binary + 3 continuous
    REQUIRE(stats.numBinary == 5);
    REQUIRE(stats.numContinuous == 3);
    REQUIRE(stats.numInteger == 0);
}

/**
 * @test ModelStatistics::ConstraintCount
 * @brief Verifies computeStatistics returns correct constraint counts
 */
TEST_CASE("B2: ModelStatistics::ConstraintCount", "[diagnostics][statistics]")
{
    DiagBuilder builder;
    builder.addVariables();
    builder.addConstraints();
    builder.model().update();  // Ensure attributes are available
    
    auto stats = computeStatistics(builder.model());
    
    REQUIRE(stats.numConstrs == 2);  // sum_x and sum_y
}

/**
 * @test ModelStatistics::NonZeroCount
 * @brief Verifies computeStatistics returns non-zero coefficient count
 */
TEST_CASE("B3: ModelStatistics::NonZeroCount", "[diagnostics][statistics]")
{
    DiagBuilder builder;
    builder.addVariables();
    builder.addConstraints();
    builder.model().update();  // Ensure attributes are available
    
    auto stats = computeStatistics(builder.model());
    
    // sum_x has 3 coefficients, sum_y has 2 coefficients
    REQUIRE(stats.numNonZeros == 5);
}

/**
 * @test ModelStatistics::EmptyModel
 * @brief Verifies computeStatistics handles empty model
 */
TEST_CASE("B4: ModelStatistics::EmptyModel", "[diagnostics][statistics]")
{
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model(env);
    
    auto stats = computeStatistics(model);
    
    REQUIRE(stats.numVars == 0);
    REQUIRE(stats.numConstrs == 0);
    REQUIRE(stats.numBinary == 0);
    REQUIRE(stats.numInteger == 0);
    REQUIRE(stats.numContinuous == 0);
}

/**
 * @test ModelStatistics::MixedIntegerModel
 * @brief Verifies statistics for model with binary and general integer vars
 */
TEST_CASE("B5: ModelStatistics::MixedIntegerModel", "[diagnostics][statistics]")
{
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model(env);
    
    // Add mixed variable types
    model.addVar(0, 1, 0, GRB_BINARY, "b1");
    model.addVar(0, 1, 0, GRB_BINARY, "b2");
    model.addVar(0, 10, 0, GRB_INTEGER, "i1");
    model.addVar(0, 100, 0, GRB_CONTINUOUS, "c1");
    model.update();
    
    auto stats = computeStatistics(model);
    
    REQUIRE(stats.numVars == 4);
    REQUIRE(stats.numBinary == 2);
    REQUIRE(stats.numInteger == 1);
    REQUIRE(stats.numContinuous == 1);
}

// ============================================================================
// SECTION C: IIS COMPUTATION
// ============================================================================

/**
 * @test IISComputation::InfeasibleModel
 * @brief Verifies IIS is computed for infeasible model
 */
TEST_CASE("C1: IISComputation::InfeasibleModel", "[diagnostics][iis]")
{
    DiagBuilder builder;
    builder.makeInfeasible = true;
    builder.optimize();
    
    REQUIRE(builder.isInfeasible());
    
    auto iis = computeIIS(builder.model());
    
    // IIS should not be empty
    REQUIRE_FALSE(iis.empty());
    REQUIRE(iis.size() > 0);
}

/**
 * @test IISComputation::ContainsConflictingConstraints
 * @brief Verifies IIS contains the conflicting constraints
 */
TEST_CASE("C2: IISComputation::ContainsConflictingConstraints", "[diagnostics][iis]")
{
    DiagBuilder builder;
    builder.makeInfeasible = true;
    builder.optimize();
    
    auto iis = computeIIS(builder.model());
    
    // Should have constraints or bounds in IIS
    bool hasConstraints = !iis.constraints.empty();
    bool hasBounds = !iis.lowerBounds.empty() || !iis.upperBounds.empty();
    
    REQUIRE((hasConstraints || hasBounds));
}

/**
 * @test IISComputation::ConstraintNames
 * @brief Verifies IIS constraint names are captured
 */
TEST_CASE("C3: IISComputation::ConstraintNames", "[diagnostics][iis]")
{
    DiagBuilder builder;
    builder.makeInfeasible = true;
    builder.optimize();
    
    auto iis = computeIIS(builder.model());
    
    // Check that constraint names are non-empty strings
    for (const auto& [name, constr] : iis.constraints) {
        REQUIRE_FALSE(name.empty());
    }
}

/**
 * @test IISResult::EmptyCheck
 * @brief Verifies IISResult::empty() works correctly
 */
TEST_CASE("C4: IISResult::EmptyCheck", "[diagnostics][iis]")
{
    IISResult empty;
    REQUIRE(empty.empty());
    REQUIRE(empty.size() == 0);
    
    IISResult nonEmpty;
    nonEmpty.constraints.emplace_back("test", GRBConstr());
    REQUIRE_FALSE(nonEmpty.empty());
    REQUIRE(nonEmpty.size() == 1);
}

/**
 * @test IISResult::SizeCalculation
 * @brief Verifies IISResult::size() sums all components
 */
TEST_CASE("C5: IISResult::SizeCalculation", "[diagnostics][iis]")
{
    IISResult result;
    result.constraints.emplace_back("c1", GRBConstr());
    result.constraints.emplace_back("c2", GRBConstr());
    result.lowerBounds.emplace_back("lb1", GRBVar());
    result.upperBounds.emplace_back("ub1", GRBVar());
    result.upperBounds.emplace_back("ub2", GRBVar());
    
    REQUIRE(result.size() == 5);  // 2 + 1 + 2
}

// ============================================================================
// SECTION D: SOLUTION QUALITY METRICS
// ============================================================================

/**
 * @test SolutionQuality::OptimalSolution
 * @brief Verifies solution quality for optimal solution has minimal violations
 */
TEST_CASE("D1: SolutionQuality::OptimalSolution", "[diagnostics][quality]")
{
    DiagBuilder builder;
    builder.optimize();
    
    REQUIRE(builder.isOptimal());
    
    auto quality = computeSolutionQuality(builder.model());
    
    // Optimal solution should have negligible violations
    REQUIRE(quality.maxConstrViolation < 1e-6);
    REQUIRE(quality.maxBoundViolation < 1e-6);
}

/**
 * @test SolutionQuality::IntegralityViolation
 * @brief Verifies integrality violation is reported for MIP
 */
TEST_CASE("D2: SolutionQuality::IntegralityViolation", "[diagnostics][quality]")
{
    DiagBuilder builder;
    builder.optimize();
    
    auto quality = computeSolutionQuality(builder.model());
    
    // For optimal MIP, integrality violation should be minimal
    REQUIRE(quality.maxIntViolation < 1e-6);
}

// ============================================================================
// SECTION E: CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * @test ConvenienceFunctions::IsLP
 * @brief Verifies isLP correctly identifies LP models
 */
TEST_CASE("E1: ConvenienceFunctions::IsLP", "[diagnostics][convenience]")
{
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    
    // Pure LP
    GRBModel lpModel(env);
    lpModel.addVar(0, 10, 0, GRB_CONTINUOUS, "x");
    lpModel.update();
    REQUIRE(isLP(lpModel));
    REQUIRE_FALSE(isMIP(lpModel));
    
    // MIP with binary
    GRBModel mipModel(env);
    mipModel.addVar(0, 1, 0, GRB_BINARY, "y");
    mipModel.update();
    REQUIRE_FALSE(isLP(mipModel));
    REQUIRE(isMIP(mipModel));
}

/**
 * @test ConvenienceFunctions::IsMIP
 * @brief Verifies isMIP correctly identifies MIP models
 */
TEST_CASE("E2: ConvenienceFunctions::IsMIP", "[diagnostics][convenience]")
{
    DiagBuilder builder;
    builder.addVariables();
    builder.model().update();  // Ensure attributes are available
    
    // Has binary variables, so it's a MIP
    REQUIRE(isMIP(builder.model()));
    REQUIRE_FALSE(isLP(builder.model()));
}

/**
 * @test ConvenienceFunctions::ModelSummaryBasic
 * @brief Verifies modelSummary produces readable output
 */
TEST_CASE("E3: ConvenienceFunctions::ModelSummaryBasic", "[diagnostics][convenience]")
{
    DiagBuilder builder;
    builder.numBinary = 5;
    builder.numContinuous = 3;
    builder.addVariables();
    builder.addConstraints();
    builder.model().update();  // Ensure attributes are available
    
    std::string summary = modelSummary(builder.model());
    
    // Should mention vars and constrs
    REQUIRE(summary.find("vars") != std::string::npos);
    REQUIRE(summary.find("constrs") != std::string::npos);
    // Should mention binary count
    REQUIRE(summary.find("bin") != std::string::npos);
}

/**
 * @test ConvenienceFunctions::ModelSummaryPureLP
 * @brief Verifies modelSummary for pure LP doesn't mention integer types
 */
TEST_CASE("E4: ConvenienceFunctions::ModelSummaryPureLP", "[diagnostics][convenience]")
{
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model(env);
    
    model.addVar(0, 10, 0, GRB_CONTINUOUS, "x1");
    model.addVar(0, 10, 0, GRB_CONTINUOUS, "x2");
    model.addConstr(GRBLinExpr(), GRB_LESS_EQUAL, 5, "c1");
    model.update();
    
    std::string summary = modelSummary(model);
    
    // Should mention 2 vars and 1 constr
    REQUIRE(summary.find("2 vars") != std::string::npos);
    REQUIRE(summary.find("1 constrs") != std::string::npos);
    // Should NOT mention bin or int for pure LP
    REQUIRE(summary.find("bin") == std::string::npos);
    REQUIRE(summary.find("int") == std::string::npos);
}

/**
 * @test ConvenienceFunctions::ModelSummaryMixedInteger
 * @brief Verifies modelSummary for mixed integer model
 */
TEST_CASE("E5: ConvenienceFunctions::ModelSummaryMixedInteger", "[diagnostics][convenience]")
{
    GRBEnv env(true);
    env.set(GRB_IntParam_OutputFlag, 0);
    env.start();
    GRBModel model(env);
    
    model.addVar(0, 1, 0, GRB_BINARY, "b1");
    model.addVar(0, 1, 0, GRB_BINARY, "b2");
    model.addVar(0, 10, 0, GRB_INTEGER, "i1");
    model.addVar(0, 100, 0, GRB_CONTINUOUS, "c1");
    model.update();
    
    std::string summary = modelSummary(model);
    
    REQUIRE(summary.find("4 vars") != std::string::npos);
    REQUIRE(summary.find("2 bin") != std::string::npos);
    REQUIRE(summary.find("1 int") != std::string::npos);
}
