#pragma once
/*
===============================================================================
DIAGNOSTICS — Advanced model analysis and debugging utilities
===============================================================================

Overview
--------
This header provides diagnostic utilities for analyzing Gurobi optimization
models. Unlike the lightweight solution accessors in ModelBuilder (status(),
objVal(), etc.), these utilities perform deeper analysis:

    * Model statistics (variable/constraint counts by type)
    * IIS computation for infeasible models
    * Solution quality metrics (constraint violations)
    * Human-readable status string conversion

Design Philosophy
-----------------
1. Free functions operating on GRBModel — not tied to ModelBuilder template
2. Lightweight result structs for returning diagnostic data
3. Non-templated — avoids code bloat, works with any model
4. Optional include — users who don't need IIS don't pay for it

Typical Usage
-------------
    #include "model_builder.h"
    #include "diagnostics.h"

    MyBuilder builder;
    builder.optimize();

    // Status as string
    std::cout << dsl::statusString(builder.status()) << "\n";

    // Model statistics
    auto stats = dsl::computeStatistics(builder.model());
    std::cout << "Variables: " << stats.numVars << "\n";
    std::cout << "Binary: " << stats.numBinary << "\n";

    // IIS for infeasible models
    if (builder.isInfeasible()) {
        auto iis = dsl::computeIIS(builder.model());
        for (const auto& [name, constr] : iis.constraints) {
            std::cout << "IIS constraint: " << name << "\n";
        }
    }

    // Solution quality check
    if (builder.hasSolution()) {
        auto quality = dsl::computeSolutionQuality(builder.model());
        std::cout << "Max violation: " << quality.maxConstrViolation << "\n";
    }

===============================================================================
*/

#include <string>
#include <vector>
#include <utility>
#include "gurobi_c++.h"

namespace dsl {

// =============================================================================
// STATUS STRING CONVERSION
// =============================================================================

/**
 * @brief Convert Gurobi status code to human-readable string
 * @param status Gurobi status code (GRB_OPTIMAL, GRB_INFEASIBLE, etc.)
 * @return Human-readable status name
 *
 * @example
 *     std::cout << statusString(model.get(GRB_IntAttr_Status)) << "\n";
 *     // Output: "OPTIMAL" or "INFEASIBLE" etc.
 */
inline std::string statusString(int status) {
    switch (status) {
        case GRB_LOADED:          return "LOADED";
        case GRB_OPTIMAL:         return "OPTIMAL";
        case GRB_INFEASIBLE:      return "INFEASIBLE";
        case GRB_INF_OR_UNBD:     return "INF_OR_UNBD";
        case GRB_UNBOUNDED:       return "UNBOUNDED";
        case GRB_CUTOFF:          return "CUTOFF";
        case GRB_ITERATION_LIMIT: return "ITERATION_LIMIT";
        case GRB_NODE_LIMIT:      return "NODE_LIMIT";
        case GRB_TIME_LIMIT:      return "TIME_LIMIT";
        case GRB_SOLUTION_LIMIT:  return "SOLUTION_LIMIT";
        case GRB_INTERRUPTED:     return "INTERRUPTED";
        case GRB_NUMERIC:         return "NUMERIC";
        case GRB_SUBOPTIMAL:      return "SUBOPTIMAL";
        case GRB_INPROGRESS:      return "INPROGRESS";
        case GRB_USER_OBJ_LIMIT:  return "USER_OBJ_LIMIT";
        default:                  return "UNKNOWN(" + std::to_string(status) + ")";
    }
}

// =============================================================================
// MODEL STATISTICS
// =============================================================================

/**
 * @brief Snapshot of model size and composition
 *
 * @details Provides counts of variables by type, constraints, and matrix
 *          non-zeros. Useful for logging, debugging, and model validation.
 */
struct ModelStatistics {
    int numVars = 0;        ///< Total number of variables
    int numConstrs = 0;     ///< Total number of linear constraints
    int numBinary = 0;      ///< Number of binary variables
    int numInteger = 0;     ///< Number of general integer variables
    int numContinuous = 0;  ///< Number of continuous variables
    int numNonZeros = 0;    ///< Number of non-zero coefficients in constraint matrix
    int numSOS = 0;         ///< Number of SOS constraints
    int numQConstrs = 0;    ///< Number of quadratic constraints
    int numGenConstrs = 0;  ///< Number of general constraints
};

/**
 * @brief Compute statistics for a Gurobi model
 * @param model The model to analyze (const reference)
 * @return ModelStatistics struct with counts
 *
 * @note Call after model is built (variables/constraints added)
 *
 * @example
 *     auto stats = computeStatistics(builder.model());
 *     std::cout << "Vars: " << stats.numVars 
 *               << ", Binary: " << stats.numBinary << "\n";
 */
inline ModelStatistics computeStatistics(const GRBModel& model) {
    ModelStatistics stats;
    
    stats.numVars = model.get(GRB_IntAttr_NumVars);
    stats.numConstrs = model.get(GRB_IntAttr_NumConstrs);
    stats.numBinary = model.get(GRB_IntAttr_NumBinVars);
    // NumIntVars in Gurobi includes binary vars, so subtract to get general integers only
    int totalIntVars = model.get(GRB_IntAttr_NumIntVars);
    stats.numInteger = totalIntVars - stats.numBinary;
    stats.numNonZeros = model.get(GRB_IntAttr_NumNZs);
    stats.numSOS = model.get(GRB_IntAttr_NumSOS);
    stats.numQConstrs = model.get(GRB_IntAttr_NumQConstrs);
    stats.numGenConstrs = model.get(GRB_IntAttr_NumGenConstrs);
    
    // Continuous = Total - Binary - General Integer
    stats.numContinuous = stats.numVars - stats.numBinary - stats.numInteger;
    
    return stats;
}

// =============================================================================
// IIS (IRREDUCIBLE INCONSISTENT SUBSYSTEM)
// =============================================================================

/**
 * @brief Result of IIS computation for an infeasible model
 *
 * @details Contains the constraints and variable bounds that form a minimal
 *          infeasible subsystem. Removing any single element would make the
 *          remaining system feasible.
 */
struct IISResult {
    /// Constraints in the IIS (name, constraint object)
    std::vector<std::pair<std::string, GRBConstr>> constraints;
    
    /// Variables with lower bounds in the IIS
    std::vector<std::pair<std::string, GRBVar>> lowerBounds;
    
    /// Variables with upper bounds in the IIS
    std::vector<std::pair<std::string, GRBVar>> upperBounds;
    
    /// @brief Check if IIS is empty (no conflicts found)
    bool empty() const {
        return constraints.empty() && lowerBounds.empty() && upperBounds.empty();
    }
    
    /// @brief Total number of elements in the IIS
    size_t size() const {
        return constraints.size() + lowerBounds.size() + upperBounds.size();
    }
};

/**
 * @brief Compute IIS for an infeasible model
 * @param model The infeasible model (non-const: IIS computation modifies state)
 * @return IISResult containing conflicting constraints and bounds
 *
 * @note Only call when model status is INFEASIBLE or INF_OR_UNBD
 * @note This is a potentially expensive operation
 *
 * @example
 *     if (builder.isInfeasible()) {
 *         auto iis = computeIIS(builder.model());
 *         std::cout << "IIS has " << iis.size() << " elements:\n";
 *         for (const auto& [name, c] : iis.constraints) {
 *             std::cout << "  Constraint: " << name << "\n";
 *         }
 *         for (const auto& [name, v] : iis.lowerBounds) {
 *             std::cout << "  Lower bound: " << name << "\n";
 *         }
 *     }
 */
inline IISResult computeIIS(GRBModel& model) {
    IISResult result;
    
    // Compute IIS (modifies model attributes)
    model.computeIIS();
    
    // Collect constraints in IIS
    GRBConstr* constrs = model.getConstrs();
    int numConstrs = model.get(GRB_IntAttr_NumConstrs);
    
    for (int i = 0; i < numConstrs; ++i) {
        if (constrs[i].get(GRB_IntAttr_IISConstr) > 0) {
            std::string name = constrs[i].get(GRB_StringAttr_ConstrName);
            result.constraints.emplace_back(name, constrs[i]);
        }
    }
    
    // Collect variable bounds in IIS
    GRBVar* vars = model.getVars();
    int numVars = model.get(GRB_IntAttr_NumVars);
    
    for (int i = 0; i < numVars; ++i) {
        std::string name = vars[i].get(GRB_StringAttr_VarName);
        
        if (vars[i].get(GRB_IntAttr_IISLB) > 0) {
            result.lowerBounds.emplace_back(name, vars[i]);
        }
        if (vars[i].get(GRB_IntAttr_IISUB) > 0) {
            result.upperBounds.emplace_back(name, vars[i]);
        }
    }
    
    return result;
}

// =============================================================================
// SOLUTION QUALITY
// =============================================================================

/**
 * @brief Quality metrics for a solution
 *
 * @details Measures constraint and bound violations. Useful for validating
 *          solution quality, especially after time-limited solves or when
 *          using tolerance parameters.
 */
struct SolutionQuality {
    double maxConstrViolation = 0.0;  ///< Maximum constraint violation
    double sumConstrViolation = 0.0;  ///< Sum of all constraint violations
    double maxBoundViolation = 0.0;   ///< Maximum variable bound violation
    double maxIntViolation = 0.0;     ///< Maximum integrality violation
};

/**
 * @brief Compute solution quality metrics
 * @param model The model with a solution (const reference)
 * @return SolutionQuality struct with violation metrics
 *
 * @note Only call when model has a solution (hasSolution() == true)
 *
 * @example
 *     if (builder.hasSolution()) {
 *         auto quality = computeSolutionQuality(builder.model());
 *         if (quality.maxConstrViolation > 1e-6) {
 *             std::cout << "Warning: constraint violation detected\n";
 *         }
 *     }
 */
inline SolutionQuality computeSolutionQuality(const GRBModel& model) {
    SolutionQuality quality;
    
    // These attributes are available after optimization with a solution
    quality.maxConstrViolation = model.get(GRB_DoubleAttr_MaxVio);
    quality.maxBoundViolation = model.get(GRB_DoubleAttr_BoundVio);
    quality.maxIntViolation = model.get(GRB_DoubleAttr_IntVio);
    
    // Compute sum of constraint violations manually
    GRBConstr* constrs = model.getConstrs();
    int numConstrs = model.get(GRB_IntAttr_NumConstrs);
    
    for (int i = 0; i < numConstrs; ++i) {
        double slack = constrs[i].get(GRB_DoubleAttr_Slack);
        char sense = constrs[i].get(GRB_CharAttr_Sense);
        
        // Violation is negative slack for <= and >=, or abs(slack) for =
        double violation = 0.0;
        if (sense == GRB_LESS_EQUAL && slack < 0) {
            violation = -slack;
        } else if (sense == GRB_GREATER_EQUAL && slack < 0) {
            violation = -slack;
        } else if (sense == GRB_EQUAL) {
            violation = std::abs(slack);
        }
        
        quality.sumConstrViolation += violation;
    }
    
    return quality;
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * @brief Check if model is a pure LP (no integer variables)
 * @param model The model to check
 * @return true if model has no binary or integer variables
 */
inline bool isLP(const GRBModel& model) {
    return model.get(GRB_IntAttr_NumBinVars) == 0 &&
           model.get(GRB_IntAttr_NumIntVars) == 0;
}

/**
 * @brief Check if model is a MIP (has integer variables)
 * @param model The model to check
 * @return true if model has binary or integer variables
 */
inline bool isMIP(const GRBModel& model) {
    return !isLP(model);
}

/**
 * @brief Get a brief summary string of model statistics
 * @param model The model to summarize
 * @return Summary string like "100 vars (50 bin, 10 int), 200 constrs"
 */
inline std::string modelSummary(const GRBModel& model) {
    auto stats = computeStatistics(model);
    
    std::string result = std::to_string(stats.numVars) + " vars";
    
    if (stats.numBinary > 0 || stats.numInteger > 0) {
        result += " (";
        if (stats.numBinary > 0) {
            result += std::to_string(stats.numBinary) + " bin";
            if (stats.numInteger > 0) result += ", ";
        }
        if (stats.numInteger > 0) {
            result += std::to_string(stats.numInteger) + " int";
        }
        result += ")";
    }
    
    result += ", " + std::to_string(stats.numConstrs) + " constrs";
    
    if (stats.numQConstrs > 0) {
        result += ", " + std::to_string(stats.numQConstrs) + " qconstrs";
    }
    
    return result;
}

} // namespace dsl
