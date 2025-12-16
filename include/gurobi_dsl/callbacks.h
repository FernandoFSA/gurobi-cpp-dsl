#pragma once
/*
===============================================================================
CALLBACKS — MIP Callback Framework for Gurobi C++ DSL
===============================================================================

Overview
--------
Provides a type-safe callback framework for intercepting the MIP optimization
process. Simplifies common callback patterns:

    * Progress monitoring (gap, bound, runtime)
    * Incumbent solution access
    * Lazy constraint injection
    * Early termination

This wrapper eliminates the need to remember GRB_CB_* constants and provides
RAII-based solution access instead of raw pointer management.

Key Components
--------------
• Progress         — Struct containing optimization progress metrics
• CallbackSolution — RAII wrapper for accessing solution values in callbacks
• MIPCallback      — Base class with named virtual methods for callback events

Design Philosophy
-----------------
1. Named virtual methods instead of where-based dispatch
2. RAII solution access — no manual memory management
3. Progress struct for convenient metric access
4. Compatible with VariableGroup and IndexedVariableSet

Typical Usage
-------------
    class MyCallback : public dsl::MIPCallback {
    protected:
        void onIncumbent(const CallbackSolution& sol) override {
            // Access solution values
            double x0 = sol(X, 0);
            auto xvals = sol.getValues(X);
            
            // Add lazy constraint if needed
            if (needsCut(xvals)) {
                addLazy(X(0) + X(1) <= 5);
            }
        }
        
        void onProgress(const Progress& p) override {
            std::cout << "Gap: " << p.gap * 100 << "%\n";
            
            // Early termination
            if (p.gap < 0.01) abort();
        }
        
    private:
        VariableGroup& X;
    };
    
    // Usage
    MyCallback cb;
    model.setCallback(&cb);
    model.set(GRB_IntParam_LazyConstraints, 1);  // If using lazy constraints
    model.optimize();

Callback Points
---------------
| Method         | When Called                | Common Use                    |
|----------------|----------------------------|-------------------------------|
| onIncumbent()  | New incumbent found        | Lazy constraints, logging     |
| onMIPNode()    | At each B&B node           | User cuts (advanced)          |
| onProgress()   | Periodically during MIP    | Monitoring, early termination |
| onMessage()    | Gurobi log message         | Custom logging                |

Dependencies
------------
• <string>, <vector>, <functional>
• "gurobi_c++.h" — Gurobi C++ API
• "variables.h" — VariableGroup, IndexedVariableSet (optional integration)

Thread Safety
-------------
• Callbacks are invoked from Gurobi's internal threads
• Do not modify shared state without synchronization
• Solution access is only valid within the callback scope

Exception Safety
----------------
• Exceptions in callbacks are caught by Gurobi and abort optimization
• Use try/catch within callbacks if you need custom error handling

===============================================================================
*/

#include <string>
#include <vector>
#include <functional>
#include <stdexcept>

#include "gurobi_c++.h"

namespace dsl {

// Forward declarations
class VariableGroup;
class IndexedVariableSet;

// =============================================================================
// PROGRESS STRUCT
// =============================================================================

/**
 * @brief Optimization progress metrics
 *
 * @details Contains key metrics available during MIP optimization.
 *          Values are populated from Gurobi callback attributes.
 *
 * @note Not all values are available at all callback points.
 *       Unavailable values are set to their default (0 or infinity).
 */
struct Progress {
    double runtime = 0.0;           ///< Elapsed time in seconds
    double bestObj = GRB_INFINITY;  ///< Best incumbent objective value
    double bestBound = -GRB_INFINITY; ///< Best relaxation bound
    double gap = GRB_INFINITY;      ///< Relative MIP gap (0.0 = optimal)
    int nodeCount = 0;              ///< Number of B&B nodes explored
    int solutionCount = 0;          ///< Number of incumbent solutions found
    
    /**
     * @brief Check if a feasible solution has been found
     * @return true if at least one incumbent exists
     */
    bool hasSolution() const noexcept {
        return solutionCount > 0;
    }
    
    /**
     * @brief Check if gap is within tolerance
     * @param tolerance Gap threshold (default 1% = 0.01)
     * @return true if gap <= tolerance
     */
    bool gapWithin(double tolerance = 0.01) const noexcept {
        return gap <= tolerance;
    }
};

// =============================================================================
// CALLBACK SOLUTION ACCESSOR
// =============================================================================

// Forward declare MIPCallback
class MIPCallback;

/**
 * @brief RAII wrapper for accessing solution values within callbacks
 *
 * @details Provides safe access to incumbent solution values during
 *          onIncumbent() callbacks. Handles memory management internally.
 *
 * @note Only valid within the scope of a callback invocation.
 *       Do not store references to CallbackSolution.
 */
class CallbackSolution {
public:
    /**
     * @brief Get solution value for a single variable
     * @param v The GRBVar to query
     * @return Solution value for v
     */
    double operator()(const GRBVar& v) const;
    
    /**
     * @brief Get solution value from a VariableGroup at index
     * @param vg The VariableGroup
     * @param i First index
     * @return Solution value at vg(i)
     */
    double operator()(const VariableGroup& vg, int i) const;
    
    /**
     * @brief Get solution value from a VariableGroup at 2D index
     * @param vg The VariableGroup
     * @param i First index
     * @param j Second index
     * @return Solution value at vg(i, j)
     */
    double operator()(const VariableGroup& vg, int i, int j) const;
    
    /**
     * @brief Get solution value from a VariableGroup at 3D index
     * @param vg The VariableGroup
     * @param i First index
     * @param j Second index
     * @param k Third index
     * @return Solution value at vg(i, j, k)
     */
    double operator()(const VariableGroup& vg, int i, int j, int k) const;
    
    /**
     * @brief Get solution value from an IndexedVariableSet
     * @param vs The IndexedVariableSet
     * @param i First index
     * @return Solution value at vs(i)
     */
    double operator()(const IndexedVariableSet& vs, int i) const;
    
    /**
     * @brief Get solution value from an IndexedVariableSet at 2D index
     * @param vs The IndexedVariableSet
     * @param i First index
     * @param j Second index
     * @return Solution value at vs(i, j)
     */
    double operator()(const IndexedVariableSet& vs, int i, int j) const;
    
    /**
     * @brief Get all solution values from a VariableGroup
     * @param vg The VariableGroup to query
     * @return Vector of solution values in forEach order
     */
    std::vector<double> getValues(const VariableGroup& vg) const;
    
    /**
     * @brief Get all solution values from an IndexedVariableSet
     * @param vs The IndexedVariableSet to query
     * @return Vector of solution values in storage order
     */
    std::vector<double> getValues(const IndexedVariableSet& vs) const;

private:
    friend class MIPCallback;
    
    // Only MIPCallback can construct CallbackSolution
    explicit CallbackSolution(MIPCallback* cb) : callback_(cb) {}
    
    MIPCallback* callback_;
};

// =============================================================================
// MIP CALLBACK BASE CLASS
// =============================================================================

/**
 * @brief Base class for MIP optimization callbacks
 *
 * @details Inherit from this class and override the virtual methods
 *          to intercept optimization events. The base class handles
 *          dispatch from Gurobi's where-based callback to named methods.
 *
 * @example
 *     class SubtourCallback : public dsl::MIPCallback {
 *         VariableGroup& X;
 *         int n;
 *     public:
 *         SubtourCallback(VariableGroup& x, int nodes) : X(x), n(nodes) {}
 *         
 *     protected:
 *         void onIncumbent(const CallbackSolution& sol) override {
 *             auto xvals = sol.getValues(X);
 *             auto tour = findSubtour(xvals, n);
 *             if (tour.size() < n) {
 *                 GRBLinExpr cut = buildSubtourCut(tour);
 *                 addLazy(cut <= tour.size() - 1);
 *             }
 *         }
 *     };
 */
class MIPCallback : public GRBCallback {
public:
    virtual ~MIPCallback() = default;
    
    /**
     * @brief Get solution value for a variable (for use by CallbackSolution)
     * @param v The GRBVar to query
     * @return Solution value
     * @note This wraps the protected GRBCallback::getSolution()
     */
    double getSolutionValue(const GRBVar& v) {
        return getSolution(v);
    }

protected:
    // =========================================================================
    // OVERRIDE THESE IN YOUR DERIVED CLASS
    // =========================================================================
    
    /**
     * @brief Called when a new incumbent solution is found
     *
     * @param sol Solution accessor for querying variable values
     *
     * @details Use this to:
     *          - Log incumbent solutions
     *          - Add lazy constraints (call addLazy())
     *          - Collect solutions for a solution pool
     *
     * @note Only called when a new best solution is found.
     *       Must enable GRB_IntParam_LazyConstraints=1 if adding lazy cuts.
     */
    virtual void onIncumbent(const CallbackSolution& sol) {
        (void)sol; // Suppress unused parameter warning
    }
    
    /**
     * @brief Called at each branch-and-bound node
     *
     * @details Use this to add user cuts (strengthening inequalities).
     *          More advanced than lazy constraints.
     *
     * @note Requires understanding of cutting plane theory.
     *       For most use cases, onIncumbent() with lazy constraints suffices.
     */
    virtual void onMIPNode() {}
    
    /**
     * @brief Called periodically during MIP optimization
     *
     * @param p Current progress metrics
     *
     * @details Use this to:
     *          - Log progress (gap, bound, runtime)
     *          - Terminate early (call abort())
     *          - Save checkpoints
     *
     * @note Called frequently; keep implementation lightweight.
     */
    virtual void onProgress(const Progress& p) {
        (void)p; // Suppress unused parameter warning
    }
    
    /**
     * @brief Called for each Gurobi log message
     *
     * @param msg The log message
     *
     * @details Use this to redirect or filter Gurobi output.
     */
    virtual void onMessage(const std::string& msg) {
        (void)msg; // Suppress unused parameter warning
    }

    // =========================================================================
    // HELPER METHODS AVAILABLE IN CALLBACKS
    // =========================================================================
    
    /**
     * @brief Get current optimization progress
     * @return Progress struct with current metrics
     *
     * @note Available in onProgress(), onIncumbent(), onMIPNode()
     */
    Progress progress() {
        Progress p;
        try {
            p.runtime = getDoubleInfo(GRB_CB_RUNTIME);
        } catch (...) {}
        
        try {
            if (where == GRB_CB_MIP || where == GRB_CB_MIPSOL || where == GRB_CB_MIPNODE) {
                p.bestObj = getDoubleInfo(GRB_CB_MIP_OBJBST);
                p.bestBound = getDoubleInfo(GRB_CB_MIP_OBJBND);
                p.nodeCount = static_cast<int>(getDoubleInfo(GRB_CB_MIP_NODCNT));
                p.solutionCount = getIntInfo(GRB_CB_MIP_SOLCNT);
                
                // Compute gap
                if (p.solutionCount > 0 && std::abs(p.bestObj) > 1e-10) {
                    p.gap = std::abs(p.bestObj - p.bestBound) / std::abs(p.bestObj);
                }
            }
        } catch (...) {}
        
        return p;
    }
    
    /**
     * @brief Get elapsed runtime in seconds
     * @return Seconds since optimization started
     */
    double runtime() {
        try {
            return getDoubleInfo(GRB_CB_RUNTIME);
        } catch (...) {
            return 0.0;
        }
    }
    
    /**
     * @brief Get current MIP gap
     * @return Relative gap (0.0 = optimal, 1.0 = 100% gap)
     */
    double gap() {
        return progress().gap;
    }
    
    /**
     * @brief Get best relaxation bound
     * @return Current best bound
     */
    double bestBound() {
        try {
            return getDoubleInfo(GRB_CB_MIP_OBJBND);
        } catch (...) {
            return -GRB_INFINITY;
        }
    }
    
    /**
     * @brief Get best incumbent objective
     * @return Current best objective value
     */
    double bestObj() {
        try {
            return getDoubleInfo(GRB_CB_MIP_OBJBST);
        } catch (...) {
            return GRB_INFINITY;
        }
    }
    
    /**
     * @brief Add a lazy constraint
     *
     * @param constr The constraint to add (GRBTempConstr)
     *
     * @details Lazy constraints are only checked when an incumbent is found.
     *          Use for constraints that are too numerous to add upfront
     *          (e.g., subtour elimination in TSP).
     *
     * @note Must set GRB_IntParam_LazyConstraints=1 before optimize().
     * @note Only valid in onIncumbent() callback.
     *
     * @example
     *     addLazy(X(0) + X(1) + X(2) <= 2);
     */
    void addLazy(const GRBTempConstr& constr) {
        GRBCallback::addLazy(constr);
    }
    
    /**
     * @brief Add a user cut
     *
     * @param constr The cut to add (GRBTempConstr)
     *
     * @details User cuts strengthen the LP relaxation without removing
     *          any integer solutions. Advanced usage.
     *
     * @note Only valid in onMIPNode() callback.
     */
    void addCut(const GRBTempConstr& constr) {
        GRBCallback::addCut(constr);
    }
    
    /**
     * @brief Terminate optimization early
     *
     * @details Signals Gurobi to stop optimization at the next opportunity.
     *          The current best solution (if any) will be available.
     *
     * @example
     *     void onProgress(const Progress& p) override {
     *         if (p.gap < 0.01) {
     *             std::cout << "Good enough, stopping.\n";
     *             abort();
     *         }
     *     }
     */
    void abort() {
        GRBCallback::abort();
    }

private:
    // =========================================================================
    // GUROBI CALLBACK DISPATCH
    // =========================================================================
    
    /**
     * @brief Main callback entry point (called by Gurobi)
     *
     * @details Dispatches to the appropriate virtual method based on
     *          the callback location (where).
     */
    void callback() override {
        try {
            switch (where) {
                case GRB_CB_MIPSOL: {
                    // New incumbent solution found
                    CallbackSolution sol(this);
                    onIncumbent(sol);
                    break;
                }
                
                case GRB_CB_MIPNODE: {
                    // At a B&B node (for user cuts)
                    // Only call if node is optimal (has valid relaxation)
                    if (getIntInfo(GRB_CB_MIPNODE_STATUS) == GRB_OPTIMAL) {
                        onMIPNode();
                    }
                    break;
                }
                
                case GRB_CB_MIP: {
                    // Periodic MIP progress
                    onProgress(progress());
                    break;
                }
                
                case GRB_CB_MESSAGE: {
                    // Log message
                    std::string msg = getStringInfo(GRB_CB_MSG_STRING);
                    onMessage(msg);
                    break;
                }
                
                default:
                    // Other callback points (PRESOLVE, SIMPLEX, BARRIER, etc.)
                    // Not dispatched in this minimal implementation
                    break;
            }
        } catch (GRBException&) {
            // Re-throw Gurobi exceptions (they abort optimization)
            throw;
        } catch (std::exception& e) {
            // Convert standard exceptions to GRBException
            throw GRBException(e.what(), GRB_ERROR_CALLBACK);
        }
    }
};

// =============================================================================
// CALLBACKSOLUTION IMPLEMENTATION (requires VariableGroup/IndexedVariableSet)
// =============================================================================

} // namespace dsl

// Include after class definitions to resolve forward declarations
#include "variables.h"

namespace dsl {

inline double CallbackSolution::operator()(const GRBVar& v) const {
    return callback_->getSolutionValue(v);
}

inline double CallbackSolution::operator()(const VariableGroup& vg, int i) const {
    return callback_->getSolutionValue(vg.at(i));
}

inline double CallbackSolution::operator()(const VariableGroup& vg, int i, int j) const {
    return callback_->getSolutionValue(vg.at(i, j));
}

inline double CallbackSolution::operator()(const VariableGroup& vg, int i, int j, int k) const {
    return callback_->getSolutionValue(vg.at(i, j, k));
}

inline double CallbackSolution::operator()(const IndexedVariableSet& vs, int i) const {
    return callback_->getSolutionValue(vs.at(i));
}

inline double CallbackSolution::operator()(const IndexedVariableSet& vs, int i, int j) const {
    return callback_->getSolutionValue(vs.at(i, j));
}

inline std::vector<double> CallbackSolution::getValues(const VariableGroup& vg) const {
    std::vector<double> result;
    const_cast<VariableGroup&>(vg).forEach([&](GRBVar& v, const std::vector<int>&) {
        result.push_back(callback_->getSolutionValue(v));
    });
    return result;
}

inline std::vector<double> CallbackSolution::getValues(const IndexedVariableSet& vs) const {
    std::vector<double> result;
    result.reserve(vs.size());
    for (const auto& entry : vs.all()) {
        result.push_back(callback_->getSolutionValue(entry.var));
    }
    return result;
}

} // namespace dsl
