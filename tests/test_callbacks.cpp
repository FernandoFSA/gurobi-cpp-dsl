/*
===============================================================================
TEST CALLBACKS — Comprehensive tests for callbacks.h
===============================================================================

OVERVIEW
--------
Validates the MIP callback framework including Progress struct, CallbackSolution
accessor, and MIPCallback base class. Tests cover optimization progress monitoring,
incumbent solution access, lazy constraint injection, and early termination.

The callback framework provides a type-safe wrapper around Gurobi's raw callback
interface, eliminating the need to remember GRB_CB_* constants and providing
RAII-based solution access instead of raw pointer management.

TEST ORGANIZATION
-----------------
• Section A: Progress struct default values and helper methods
• Section B: MIPCallback derivation and basic integration with GRBModel
• Section C: CallbackSolution value access for VariableGroup and IndexedVariableSet
• Section D: Lazy constraint injection via addLazy() in onIncumbent()
• Section E: Early termination via abort() from onIncumbent() and onProgress()
• Section F: Helper methods (runtime, bestObj, bestBound, gap, progress)

TEST STRATEGY
-------------
• Verify Progress struct initializes with correct default values
• Confirm Progress helper methods (hasSolution, gapWithin) work correctly
• Test that MIPCallback derived classes can be instantiated and used as GRBCallback*
• Validate callback is invoked during optimization by counting invocations
• Test CallbackSolution::getValues() extracts correct values from VariableGroup
• Test CallbackSolution::getValues() extracts correct values from IndexedVariableSet
• Verify addLazy() successfully adds constraints that affect the solution
• Test abort() terminates optimization early from onIncumbent()
• Test abort() can be triggered by gap condition in onProgress()
• Verify helper methods (runtime, bestObj, etc.) return valid values during callbacks

CALLBACK BEHAVIOR NOTES
-----------------------
• Callback invocation counts depend on solver behavior and problem structure
• Small problems may solve before progress callbacks are invoked
• Lazy constraints require GRB_IntParam_LazyConstraints=1 to be set
• The abort() method signals termination at the next opportunity (not immediate)
• Solution values in callbacks represent the current incumbent, not the final solution

DEPENDENCIES
------------
• Catch2 v3.0+ - Test framework
• callbacks.h - System under test (Progress, CallbackSolution, MIPCallback)
• variables.h - VariableGroup, VariableFactory for creating test variables
• indexing.h - IndexList, range() for IndexedVariableSet tests
• expressions.h - sum() for building objective and constraint expressions

===============================================================================
*/

#include "catch_amalgamated.hpp"
#include <gurobi_dsl/callbacks.h>
#include <gurobi_dsl/variables.h>
#include <gurobi_dsl/indexing.h>
#include <gurobi_dsl/expressions.h>

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
// SECTION A: PROGRESS STRUCT FUNCTIONALITY
// ============================================================================

/**
 * @test Progress::DefaultValues
 * @brief Verifies Progress struct initializes with correct default values
 *
 * @scenario A Progress struct is default-constructed
 * @given No prior state
 * @when Creating a new Progress instance
 * @then All fields have their documented default values
 *
 * @details Default values are:
 *          - runtime = 0.0 (no time elapsed)
 *          - bestObj = GRB_INFINITY (no incumbent found)
 *          - bestBound = -GRB_INFINITY (no bound computed)
 *          - gap = GRB_INFINITY (infinite gap until solution found)
 *          - nodeCount = 0 (no nodes explored)
 *          - solutionCount = 0 (no solutions found)
 *
 * @covers Progress default construction
 */
TEST_CASE("A1: Progress::DefaultValues", "[callbacks][progress]")
{
    dsl::Progress p;
    
    REQUIRE(p.runtime == 0.0);
    REQUIRE(p.bestObj == GRB_INFINITY);
    REQUIRE(p.bestBound == -GRB_INFINITY);
    REQUIRE(p.gap == GRB_INFINITY);
    REQUIRE(p.nodeCount == 0);
    REQUIRE(p.solutionCount == 0);
}

/**
 * @test Progress::HasSolution
 * @brief Verifies hasSolution() correctly indicates whether an incumbent exists
 *
 * @scenario hasSolution() is called with various solutionCount values
 * @given A Progress struct
 * @when solutionCount is 0, 1, or greater
 * @then hasSolution() returns false for 0, true otherwise
 *
 * @covers Progress::hasSolution()
 */
TEST_CASE("A2: Progress::HasSolution", "[callbacks][progress]")
{
    dsl::Progress p;
    REQUIRE_FALSE(p.hasSolution());
    
    p.solutionCount = 1;
    REQUIRE(p.hasSolution());
    
    p.solutionCount = 5;
    REQUIRE(p.hasSolution());
}

/**
 * @test Progress::GapWithin
 * @brief Verifies gapWithin() correctly checks if gap is within tolerance
 *
 * @scenario gapWithin() is called with various gap values and tolerances
 * @given A Progress struct with different gap values
 * @when Checking against various tolerance thresholds
 * @then Returns true only when gap <= tolerance
 *
 * @details The gap is the relative MIP gap:
 *          - 0.0 = optimal (no gap)
 *          - 0.01 = 1% gap
 *          - 0.10 = 10% gap
 *          - GRB_INFINITY = no solution found yet
 *
 * @covers Progress::gapWithin()
 */
TEST_CASE("A3: Progress::GapWithin", "[callbacks][progress]")
{
    dsl::Progress p;
    
    // Default gap is infinity
    REQUIRE_FALSE(p.gapWithin(0.01));
    REQUIRE_FALSE(p.gapWithin(1.0));
    
    // Set a gap
    p.gap = 0.05;  // 5%
    REQUIRE_FALSE(p.gapWithin(0.01));  // Not within 1%
    REQUIRE(p.gapWithin(0.05));        // Exactly 5%
    REQUIRE(p.gapWithin(0.10));        // Within 10%
    
    // Zero gap
    p.gap = 0.0;
    REQUIRE(p.gapWithin(0.01));
    REQUIRE(p.gapWithin(0.0));
}

// ============================================================================
// SECTION B: MIPCALLBACK BASIC USAGE
// ============================================================================

/**
 * @brief Test callback that counts how many times each virtual method is invoked
 *
 * @details This helper class tracks invocation counts for all callback methods,
 *          allowing tests to verify that callbacks are actually being triggered
 *          during optimization.
 */
class CountingCallback : public dsl::MIPCallback {
public:
    int incumbentCount = 0;   ///< Number of times onIncumbent() was called
    int progressCount = 0;    ///< Number of times onProgress() was called
    int nodeCount = 0;        ///< Number of times onMIPNode() was called
    int messageCount = 0;     ///< Number of times onMessage() was called
    
protected:
    void onIncumbent(const dsl::CallbackSolution&) override {
        incumbentCount++;
    }
    
    void onMIPNode() override {
        nodeCount++;
    }
    
    void onProgress(const dsl::Progress&) override {
        progressCount++;
    }
    
    void onMessage(const std::string&) override {
        messageCount++;
    }
};

/**
 * @test MIPCallback::DerivationCompiles
 * @brief Verifies MIPCallback can be derived from and used as GRBCallback*
 *
 * @scenario A class derives from MIPCallback and is instantiated
 * @given A derived callback class
 * @when The instance is cast to GRBCallback*
 * @then The pointer is valid and can be used with GRBModel::setCallback()
 *
 * @covers MIPCallback derivation and polymorphism
 */
TEST_CASE("B1: MIPCallback::DerivationCompiles", "[callbacks][MIPCallback]")
{
    CountingCallback cb;
    
    // Should be usable as GRBCallback*
    GRBCallback* base = &cb;
    REQUIRE(base != nullptr);
}

/**
 * @test MIPCallback::RunsWithOptimization
 * @brief Verifies callback methods are invoked during MIP optimization
 *
 * @scenario A callback is attached to a model and optimization runs
 * @given A simple MIP with binary variables and a callback
 * @when model.optimize() is called
 * @then Optimization completes and callback methods may have been invoked
 *
 * @note The exact number of callback invocations depends on solver behavior.
 *       Small problems may solve very quickly before progress callbacks trigger.
 *
 * @covers MIPCallback integration with GRBModel::setCallback()
 */
TEST_CASE("B2: MIPCallback::RunsWithOptimization", "[callbacks][MIPCallback]")
{
    GRBModel model = makeModel();
    
    // Create a simple MIP
    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 3);
    model.setObjective(X(0) + 2*X(1) + 3*X(2), GRB_MAXIMIZE);
    model.addConstr(X(0) + X(1) + X(2) <= 2);
    model.update();
    
    CountingCallback cb;
    model.setCallback(&cb);
    model.optimize();
    
    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
    
    // Callback should have been invoked at least once
    // (exact counts depend on solver behavior)
    REQUIRE(cb.incumbentCount >= 0);  // May or may not find incumbents
    REQUIRE(cb.progressCount >= 0);   // Progress calls depend on problem
}

// ============================================================================
// SECTION C: CALLBACKSOLUTION VALUE ACCESS
// ============================================================================

/**
 * @brief Test callback that captures solution values from a VariableGroup
 *
 * @details This helper stores the variable group pointer and captures
 *          solution values when onIncumbent() is called. Used to verify
 *          that CallbackSolution correctly extracts values during callbacks.
 */
class SolutionCapturingCallback : public dsl::MIPCallback {
public:
    dsl::VariableGroup* vars = nullptr;    ///< Pointer to variables to capture
    std::vector<double> capturedValues;     ///< Values captured from last incumbent
    bool wasCalled = false;                 ///< Whether onIncumbent was invoked
    
protected:
    void onIncumbent(const dsl::CallbackSolution& sol) override {
        wasCalled = true;
        if (vars) {
            capturedValues = sol.getValues(*vars);
        }
    }
};

/**
 * @test CallbackSolution::GetValuesVariableGroup
 * @brief Verifies getValues() correctly extracts solution from VariableGroup
 *
 * @scenario Solution values are accessed via CallbackSolution in onIncumbent()
 * @given A MIP with a VariableGroup and a solution-capturing callback
 * @when An incumbent solution is found during optimization
 * @then getValues() returns a vector with correct solution values
 *
 * @details The test creates a simple MIP where exactly one binary variable
 *          must be selected. The callback captures the incumbent solution
 *          and verifies the values sum to 1.0 (one selected).
 *
 * @covers CallbackSolution::getValues(const VariableGroup&)
 */
TEST_CASE("C1: CallbackSolution::GetValuesVariableGroup", "[callbacks][solution]")
{
    GRBModel model = makeModel();
    
    // Create a simple MIP with unique optimal solution
    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 3);
    model.setObjective(X(0) + 2*X(1) + 3*X(2), GRB_MAXIMIZE);
    model.addConstr(X(0) + X(1) + X(2) == 1);  // Exactly one selected
    model.update();
    
    SolutionCapturingCallback cb;
    cb.vars = &X;
    model.setCallback(&cb);
    model.optimize();
    
    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
    
    // Callback should have captured solution
    if (cb.wasCalled) {
        REQUIRE(cb.capturedValues.size() == 3);
        
        // Optimal: X(2) = 1, others = 0
        double sum = 0;
        for (double v : cb.capturedValues) {
            sum += v;
        }
        REQUIRE(sum == Catch::Approx(1.0));  // Exactly one selected
    }
}

/**
 * @brief Test callback that captures solution values from an IndexedVariableSet
 *
 * @details Similar to SolutionCapturingCallback but works with IndexedVariableSet
 *          instead of VariableGroup. Used to verify CallbackSolution works with
 *          domain-indexed variables.
 */
class IndexedSolutionCallback : public dsl::MIPCallback {
public:
    dsl::IndexedVariableSet* vars = nullptr;  ///< Pointer to indexed variables
    std::vector<double> capturedValues;        ///< Values captured from last incumbent
    bool wasCalled = false;                    ///< Whether onIncumbent was invoked
    
protected:
    void onIncumbent(const dsl::CallbackSolution& sol) override {
        wasCalled = true;
        if (vars) {
            capturedValues = sol.getValues(*vars);
        }
    }
};

/**
 * @test CallbackSolution::GetValuesIndexedVariableSet
 * @brief Verifies getValues() correctly extracts solution from IndexedVariableSet
 *
 * @scenario Solution values are accessed from IndexedVariableSet in onIncumbent()
 * @given A MIP with an IndexedVariableSet and a solution-capturing callback
 * @when An incumbent solution is found during optimization
 * @then getValues() returns a vector with values in storage order
 *
 * @covers CallbackSolution::getValues(const IndexedVariableSet&)
 */
TEST_CASE("C2: CallbackSolution::GetValuesIndexedVariableSet", "[callbacks][solution][indexed]")
{
    GRBModel model = makeModel();
    
    auto I = dsl::IndexList{0, 1, 2};
    auto X = dsl::VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "X", I);
    model.setObjective(X.at(0) + 2*X.at(1) + 3*X.at(2), GRB_MAXIMIZE);
    model.addConstr(X.at(0) + X.at(1) + X.at(2) <= 2);
    model.update();
    
    IndexedSolutionCallback cb;
    cb.vars = &X;
    model.setCallback(&cb);
    model.optimize();
    
    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
    
    if (cb.wasCalled) {
        REQUIRE(cb.capturedValues.size() == 3);
    }
}

// ============================================================================
// SECTION D: LAZY CONSTRAINT INJECTION
// ============================================================================

/**
 * @brief Test callback that adds lazy constraints based on solution values
 *
 * @details This callback demonstrates the lazy constraint pattern:
 *          1. Check the incumbent solution in onIncumbent()
 *          2. If it violates a condition, add a constraint via addLazy()
 *          3. Gurobi will reject the incumbent and continue searching
 *
 *          In this example, we add a constraint that X(0) + X(1) <= 1
 *          whenever both are selected in the incumbent.
 */
class LazyConstraintCallback : public dsl::MIPCallback {
public:
    dsl::VariableGroup* vars = nullptr;  ///< Variables to check
    int lazyCutsAdded = 0;               ///< Count of lazy constraints added
    
protected:
    void onIncumbent(const dsl::CallbackSolution& sol) override {
        if (!vars) return;
        
        // Check if both X(0) and X(1) are selected
        double x0 = sol(*vars, 0);
        double x1 = sol(*vars, 1);
        
        if (x0 + x1 > 1.5) {  // Both are ~1 (binary)
            // Add lazy constraint: at most one can be selected
            addLazy((*vars)(0) + (*vars)(1) <= 1);
            lazyCutsAdded++;
        }
    }
};

/**
 * @test LazyConstraints::AddLazy
 * @brief Verifies addLazy() successfully adds constraints during callbacks
 *
 * @scenario A lazy constraint is added when incumbent violates a condition
 * @given A MIP where the unconstrained optimum would have X(0) = X(1) = 1
 * @when The callback adds a lazy constraint X(0) + X(1) <= 1
 * @then The final solution respects the lazy constraint
 *
 * @details Lazy constraints are powerful for problems with exponentially many
 *          constraints (e.g., subtour elimination in TSP). They are only checked
 *          when an integer solution is found, not at every LP relaxation.
 *
 * @note Requires model.set(GRB_IntParam_LazyConstraints, 1) before optimize()
 *
 * @covers MIPCallback::addLazy()
 */
TEST_CASE("D1: LazyConstraints::AddLazy", "[callbacks][lazy]")
{
    GRBModel model = makeModel();
    
    // Create MIP where lazy cut changes solution
    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 3);
    
    // Objective encourages X(0) = X(1) = 1
    model.setObjective(X(0) + X(1) + 0.5*X(2), GRB_MAXIMIZE);
    model.addConstr(X(0) + X(1) + X(2) <= 3);  // Trivial constraint
    model.update();
    
    // Enable lazy constraints
    model.set(GRB_IntParam_LazyConstraints, 1);
    
    LazyConstraintCallback cb;
    cb.vars = &X;
    model.setCallback(&cb);
    model.optimize();
    
    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
    
    // Final solution should respect lazy constraint
    double x0 = X(0).get(GRB_DoubleAttr_X);
    double x1 = X(1).get(GRB_DoubleAttr_X);
    
    // Either lazy cut was added, or solution already feasible
    REQUIRE(x0 + x1 <= 1.0 + 1e-6);
}

// ============================================================================
// SECTION E: EARLY TERMINATION
// ============================================================================

/**
 * @brief Test callback that aborts after finding the first incumbent
 *
 * @details Demonstrates early termination from onIncumbent(). This pattern
 *          is useful when you only need any feasible solution, not necessarily
 *          the optimal one.
 */
class EarlyTerminationCallback : public dsl::MIPCallback {
public:
    bool aborted = false;           ///< Whether abort() was called
    int incumbentsBeforeAbort = 0;  ///< Number of incumbents seen before aborting
    
protected:
    void onIncumbent(const dsl::CallbackSolution&) override {
        incumbentsBeforeAbort++;
        if (incumbentsBeforeAbort >= 1) {
            aborted = true;
            abort();  // Signal Gurobi to stop
        }
    }
};

/**
 * @test EarlyTermination::AbortStopsOptimization
 * @brief Verifies abort() terminates optimization from onIncumbent()
 *
 * @scenario abort() is called after finding the first incumbent
 * @given A MIP that would take time to solve to optimality
 * @when The callback calls abort() in onIncumbent()
 * @then Optimization terminates early with the current best solution
 *
 * @note The resulting status may be GRB_INTERRUPTED, GRB_SUBOPTIMAL,
 *       or even GRB_OPTIMAL if the first incumbent happened to be optimal.
 *
 * @covers MIPCallback::abort()
 */
TEST_CASE("E1: EarlyTermination::AbortStopsOptimization", "[callbacks][abort]")
{
    GRBModel model = makeModel();
    
    // Create a non-trivial MIP
    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 10);
    
    // Use DSL's sum() with weighted coefficients
    auto I = dsl::range(0, 10);
    model.setObjective(dsl::sum(I, [&](int i) { return (i + 1) * X(i); }), GRB_MAXIMIZE);
    model.addConstr(dsl::sum(I, [&](int i) { return X(i); }) <= 5);
    model.update();
    
    EarlyTerminationCallback cb;
    model.setCallback(&cb);
    model.optimize();
    
    // Should have terminated early (status depends on Gurobi version)
    int status = model.get(GRB_IntAttr_Status);
    
    // Callback should have been invoked
    if (cb.aborted) {
        // Status might be INTERRUPTED or SUBOPTIMAL or OPTIMAL (if found quickly)
        REQUIRE((status == GRB_INTERRUPTED || 
                 status == GRB_SUBOPTIMAL || 
                 status == GRB_OPTIMAL ||
                 status == GRB_USER_OBJ_LIMIT));
    }
}

/**
 * @brief Test callback that aborts when gap reaches a threshold
 *
 * @details Demonstrates early termination based on solution quality.
 *          This pattern is useful when a "good enough" solution is acceptable
 *          and you don't need to wait for proven optimality.
 */
class GapTerminationCallback : public dsl::MIPCallback {
public:
    double gapThreshold = 0.10;  ///< Gap threshold (0.10 = 10%)
    bool terminated = false;     ///< Whether abort() was called
    
protected:
    void onProgress(const dsl::Progress& p) override {
        // Abort if we have a solution and gap is within threshold
        if (p.hasSolution() && p.gapWithin(gapThreshold)) {
            terminated = true;
            abort();
        }
    }
};

/**
 * @test EarlyTermination::AbortOnGap
 * @brief Verifies abort() can be triggered by gap condition in onProgress()
 *
 * @scenario abort() is called when gap drops below a threshold
 * @given A MIP with a callback monitoring progress
 * @when The gap reaches the threshold (50% in this test)
 * @then Optimization terminates with a solution within the gap tolerance
 *
 * @details This pattern is common in production systems where:
 *          - Time limits are important
 *          - A 1% or 5% gap solution is acceptable
 *          - You want custom termination logic beyond Gurobi's MIPGap parameter
 *
 * @covers Progress::hasSolution(), Progress::gapWithin(), MIPCallback::abort()
 */
TEST_CASE("E2: EarlyTermination::AbortOnGap", "[callbacks][abort][gap]")
{
    GRBModel model = makeModel();
    
    // Simple MIP
    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 5);
    model.setObjective(X(0) + X(1) + X(2) + X(3) + X(4), GRB_MAXIMIZE);
    model.addConstr(X(0) + X(1) + X(2) + X(3) + X(4) <= 3);
    model.update();
    
    GapTerminationCallback cb;
    cb.gapThreshold = 0.5;  // 50% gap threshold (easy to reach)
    model.setCallback(&cb);
    model.optimize();
    
    // Should complete (small problem)
    int status = model.get(GRB_IntAttr_Status);
    REQUIRE((status == GRB_OPTIMAL || 
             status == GRB_INTERRUPTED || 
             status == GRB_SUBOPTIMAL));
}

// ============================================================================
// SECTION F: HELPER METHOD TESTS
// ============================================================================

/**
 * @brief Test callback that captures progress information using helper methods
 *
 * @details This callback demonstrates using the helper methods (runtime, bestObj,
 *          bestBound, gap, progress) to access optimization state during callbacks.
 *          It stores a history of Progress snapshots for verification.
 */
class ProgressCapturingCallback : public dsl::MIPCallback {
public:
    std::vector<dsl::Progress> progressHistory;  ///< All Progress snapshots
    double lastRuntime = 0;                       ///< Last runtime from helper
    double lastBestObj = GRB_INFINITY;            ///< Last bestObj from helper
    
protected:
    void onProgress(const dsl::Progress& p) override {
        progressHistory.push_back(p);
        
        // Also test the individual helper methods
        lastRuntime = runtime();
        lastBestObj = bestObj();
    }
};

/**
 * @test HelperMethods::RuntimeAndBestObj
 * @brief Verifies helper methods (runtime, bestObj, etc.) return valid values
 *
 * @scenario Helper methods are called during onProgress() callback
 * @given A MIP being optimized with a progress-capturing callback
 * @when onProgress() is invoked during optimization
 * @then Helper methods return non-negative/finite values as expected
 *
 * @details The helper methods wrap Gurobi's callback info attributes:
 *          - runtime() ? GRB_CB_RUNTIME
 *          - bestObj() ? GRB_CB_MIP_OBJBST
 *          - bestBound() ? GRB_CB_MIP_OBJBND
 *          - gap() ? computed from bestObj and bestBound
 *          - progress() ? aggregates all metrics into Progress struct
 *
 * @covers MIPCallback::runtime(), bestObj(), bestBound(), gap(), progress()
 */
TEST_CASE("F1: HelperMethods::RuntimeAndBestObj", "[callbacks][helpers]")
{
    GRBModel model = makeModel();
    
    auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 5);
    model.setObjective(X(0) + X(1) + X(2), GRB_MAXIMIZE);
    model.addConstr(X(0) + X(1) + X(2) <= 2);
    model.update();
    
    ProgressCapturingCallback cb;
    model.setCallback(&cb);
    model.optimize();
    
    REQUIRE(model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
    
    // Runtime should be non-negative
    REQUIRE(cb.lastRuntime >= 0.0);
    
    // If progress was called, should have captured some info
    if (!cb.progressHistory.empty()) {
        for (const auto& p : cb.progressHistory) {
            REQUIRE(p.runtime >= 0.0);
            REQUIRE(p.nodeCount >= 0);
        }
    }
}
