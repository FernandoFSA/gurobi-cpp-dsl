#pragma once
/*
===============================================================================
GUROBI C++ DSL — Unified Include Header
===============================================================================

OVERVIEW
--------
This is the single-include header for the Gurobi C++ DSL. Including this file
provides access to all DSL components for building optimization models with
a clean, expressive syntax.

WHAT'S INCLUDED
---------------
• indexing.h     — Index domains, Cartesian products, filtering
• naming.h       — Debug/release variable naming utilities
• enum_utils.h   — Compile-time enum helpers (DECLARE_ENUM_WITH_COUNT)
• data_store.h   — Type-erased key-value storage
• variables.h    — Variable groups, indexed sets, VariableTable
• constraints.h  — Constraint groups, indexed sets, ConstraintTable
• expressions.h  — Expression building helpers (sum, etc.)
• model_builder.h— High-level model construction template
• callbacks.h    — MIP callback framework
• diagnostics.h  — Model analysis and debugging utilities

QUICK START
-----------
    #include <dsl.h>

    int main() {
        GRBEnv env;
        GRBModel model(env);

        // Create index domains
        auto I = dsl::range(0, 5);
        auto J = dsl::range(0, 3);

        // Create variables
        auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 5, 3);

        // Build constraints using sum()
        for (int i : I) {
            model.addConstr(dsl::sum(J, [&](int j) { return X(i, j); }) <= 1);
        }

        // Set objective
        model.setObjective(dsl::sum(I * J, [&](int i, int j) {
            return (i + j) * X(i, j);
        }), GRB_MAXIMIZE);

        model.optimize();

        // Extract solution
        for (auto [idx, val] : dsl::valuesWithIndex(X)) {
            if (val > 0.5) {
                std::cout << "X[" << idx[0] << "," << idx[1] << "] = 1\n";
            }
        }

        return 0;
    }

REQUIREMENTS
------------
• C++20 compiler (MSVC 19.29+, GCC 10+, Clang 12+)
• Gurobi Optimizer 10.0+ with C++ API
• Standard library headers: <format>, <ranges>, <concepts>

NAMESPACE
---------
All DSL components are in the `dsl::` namespace. Some utilities like
`make_name::` and `force_name::` are in the global namespace for convenience.

CONFIGURATION
-------------
Build configuration affects naming behavior:
• Debug builds (DSL_DEBUG or _DEBUG defined): Human-readable variable names
• Release builds: No symbolic names (zero overhead)

LICENSE
-------
See LICENSE file in repository root.

===============================================================================
*/

// ============================================================================
// CORE COMPONENTS (order matters for dependencies)
// ============================================================================

// Naming utilities (no dependencies, used by variables/constraints)
#include "naming.h"

// Enum utilities (no dependencies, used by tables)
#include "enum_utils.h"

// Data store (no dependencies, used by model_builder)
#include "data_store.h"

// Index domains (no dependencies)
#include "indexing.h"

// Variables (depends on naming, enum_utils, indexing concepts)
#include "variables.h"

// Constraints (depends on naming, enum_utils)
#include "constraints.h"

// Expressions (depends on variables, indexing)
#include "expressions.h"

// ============================================================================
// HIGH-LEVEL COMPONENTS
// ============================================================================

// Model builder (depends on variables, constraints, data_store)
#include "model_builder.h"

// Callbacks (depends on variables)
#include "callbacks.h"

// Diagnostics (standalone, operates on GRBModel)
#include "diagnostics.h"

// ============================================================================
// CONVENIENCE NAMESPACE ALIASES (optional usage)
// ============================================================================

/**
 * @namespace dsl
 * @brief Main namespace for all DSL components
 *
 * Core types:
 * - dsl::IndexList, dsl::RangeView, dsl::Cartesian, dsl::Filtered
 * - dsl::VariableGroup, dsl::IndexedVariableSet, dsl::VariableTable
 * - dsl::ConstraintGroup, dsl::IndexedConstraintSet, dsl::ConstraintTable
 * - dsl::VariableFactory, dsl::ConstraintFactory
 * - dsl::ModelBuilder<VarEnum, ConEnum>
 * - dsl::MIPCallback, dsl::CallbackSolution, dsl::Progress
 *
 * Free functions:
 * - dsl::range(), dsl::range_view(), dsl::filter()
 * - dsl::sum()
 * - dsl::value(), dsl::values(), dsl::valueAt(), dsl::valuesWithIndex()
 * - dsl::fix(), dsl::unfix(), dsl::setStart(), dsl::fixAll(), dsl::setStartAll()
 * - dsl::lb(), dsl::ub(), dsl::setLB(), dsl::setUB()
 * - dsl::rhs(), dsl::setRHS(), dsl::sense(), dsl::slack(), dsl::dual()
 * - dsl::statusString(), dsl::computeStatistics(), dsl::computeIIS()
 * - dsl::isLP(), dsl::isMIP(), dsl::modelSummary()
 */
