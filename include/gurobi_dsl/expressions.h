#pragma once
/*
===============================================================================
EXPRESSIONS � Lightweight helpers for constructing GRBLinExpr and GRBQuadExpr
===============================================================================

OVERVIEW
--------
Provides a minimal, flexible expression-building layer for constructing Gurobi
linear and quadratic expressions. The primary interface is dsl::sum(...) for
linear expressions and dsl::quadSum(...) for quadratic expressions, both of
which iterate over DSL index domains and evaluate user-provided lambdas.

KEY COMPONENTS
--------------
- sum(Range, Func): Domain-based linear summation with user-provided lambda
- sum(IndexedVariableSet): Full summation over sparse variable sets
- sum(Range, IndexedVariableSet): Domain-filtered sparse variable summation
- sum(VariableGroup): Full summation over dense variable groups
- sum(Range, VariableGroup): Domain-filtered dense variable summation
- sum(VariableContainer): Unified summation over dense or sparse containers
- sum(Range, VariableContainer): Domain-filtered unified container summation
- quadSum(Range, Func): Domain-based quadratic summation for QP objectives
- expr_detail::invoke_on_index(): Internal tuple unpacking helper
- expr_detail::add_term(): Internal term accumulation helper

DESIGN PHILOSOPHY
-----------------
� Thin layer: Defers directly to Gurobi's GRBLinExpr type
� Mathematical alignment: Models sum_{i in I} f(i) notation
� Automatic unpacking: Handles both scalar (int) and tuple indices
� Zero ownership: Builds new GRBLinExpr on each call; no caching
� Future-proof: Central point for expression type upgrades

CONCEPTUAL MODEL
----------------
The DSL treats index domains as mathematical sets:

    sum_{i in I} f(i)              -> sum(I, f)
    sum_{(i,j) in I x J} f(i,j)    -> sum(I * J, f)
    sum_{(i,j) in I x J | P} f     -> sum(I * J | filter(P), f)

USAGE EXAMPLES
--------------
    // 1D domain with lambda
    auto expr = dsl::sum(I, [&](int i) { return cost[i] * X(i); });

    // Cartesian product with lambda
    auto expr = dsl::sum(I * J, [&](int i, int j) { return A[i][j] * X(i,j); });

    // Filtered domain
    auto F = (I * J) | dsl::filter([](int i, int j) { return i < j; });
    auto expr = dsl::sum(F, [&](int i, int j) { return X(i,j); });

    // Direct variable set summation
    auto expr = dsl::sum(XV);               // All variables in XV
    auto expr = dsl::sum(I, XV);            // Variables for indices in I

DEPENDENCIES
------------
� <type_traits>, <utility>, <tuple>, <functional> - Metaprogramming
� gurobi_c++.h - GRBLinExpr, GRBVar types
� indexing.h - Index domain types, is_tuple_like_v trait

PERFORMANCE NOTES
-----------------
� sum(Range, Func): O(n) where n = number of elements in range
� sum(IndexedVariableSet): O(n) where n = number of stored variables
� sum(VariableGroup): O(n) where n = total variables in dense structure
� Memory: No dynamic allocation beyond GRBLinExpr internal growth
� GRBLinExpr: Amortized O(1) per += operation

THREAD SAFETY
-------------
� All functions are thread-safe for concurrent calls with distinct outputs
� No shared mutable state between invocations
� GRBLinExpr operations follow Gurobi's threading guarantees
� Lambda captures are user's responsibility for thread safety

EXCEPTION SAFETY
----------------
� sum(Range, Func): Strong guarantee; propagates lambda exceptions
� sum(IndexedVariableSet): Strong guarantee; no throwing operations
� sum(Range, IndexedVariableSet): Throws std::out_of_range for invalid indices
� sum(VariableGroup): Strong guarantee; no throwing operations
� sum(Range, VariableGroup): Throws std::out_of_range for invalid indices

===============================================================================
*/

#include <type_traits>
#include <utility>
#include <tuple>
#include <functional>

#include "gurobi_c++.h"
#include "indexing.h"
#include "variables.h"

namespace dsl {

    // ========================================================================
    // INTERNAL IMPLEMENTATION DETAILS
    // ========================================================================
    namespace expr_detail {

        /**
         * @brief Invokes a callable with unpacked index arguments
         *
         * @details Examines the index type and calls the function appropriately:
         *          - If idx is tuple-like: unpacks via std::apply to call f(i,j,k,...)
         *          - If idx is scalar: calls f(idx) directly
         *
         * @tparam Func Callable type; must accept either single int or unpacked tuple
         * @tparam Idx Index type; either integral or tuple-like
         * @param f Callable to invoke
         * @param idx Index value (scalar or tuple)
         *
         * @return Result of invoking f with the (unpacked) index
         *
         * @complexity O(1) plus cost of f
         *
         * @note Tuple-like detection uses detail::is_tuple_like_v from indexing.h
         * @note Perfect forwarding preserves value categories
         */
        template<typename Func, typename Idx>
        auto invoke_on_index(Func&& f, Idx&& idx) {
            using RawIdx = std::remove_cvref_t<Idx>;

            if constexpr (detail::is_tuple_like_v<RawIdx>) {
                // idx is (i,j,...). Expand into f(i,j,...)
                return std::apply(
                    [&](auto&&... args) {
                        return std::invoke(
                            std::forward<Func>(f),
                            std::forward<decltype(args)>(args)...);
                    },
                    std::forward<Idx>(idx)
                );
            }
            else {
                // idx is scalar (int)
                return std::invoke(
                    std::forward<Func>(f),
                    std::forward<Idx>(idx)
                );
            }
        }

        /**
         * @brief Adds a term to a GRBLinExpr accumulator
         *
         * @tparam Term Type of term; must be compatible with GRBLinExpr::operator+=
         * @param expr Target expression to accumulate into
         * @param t Term to add (GRBVar, GRBLinExpr, double, or compatible type)
         *
         * @complexity O(1) amortized (GRBLinExpr internal growth)
         *
         * @note Uses perfect forwarding to avoid unnecessary copies
         * @note Accepted term types: GRBVar, GRBLinExpr, double, scaled expressions
         */
        template<typename Term>
        void add_term(GRBLinExpr& expr, Term&& t) {
            expr += std::forward<Term>(t);
        }

    } // namespace expr_detail

    // ========================================================================
    // SUM FUNCTIONS
    // ========================================================================

    /**
     * @brief Builds a GRBLinExpr by summing lambda results over an index domain
     *
     * @details Iterates over any DSL index domain (IndexList, RangeView, Cartesian,
     *          Filtered, etc.) and accumulates the results of applying the lambda
     *          to each index element.
     *
     *          Semantically implements: sum_{idx in Range} func(idx...)
     *
     * @tparam Range Iterable type with begin()/end() yielding int or tuple<int,...>
     * @tparam Func Callable with signature f(int) or f(int, int, ...) as appropriate
     * @param rng Index domain to iterate over
     * @param func Lambda returning GRBVar, GRBLinExpr, double, or compatible type
     *
     * @return GRBLinExpr containing all accumulated terms
     *
     * @throws Propagates any exceptions thrown by func
     * @complexity O(n) where n = number of elements in rng
     *
     * @note Tuple indices are automatically unpacked via invoke_on_index
     * @note No assumptions about Range structure beyond iteration support
     *
     * @example
     *     // 1D summation with coefficients
     *     auto expr = dsl::sum(I, [&](int i) {
     *         return weight[i] * X(i);
     *     });
     *
     *     // 2D Cartesian product
     *     auto expr = dsl::sum(I * J, [&](int i, int j) {
     *         return A[i][j] * X(i, j);
     *     });
     *
     * @see IndexList, RangeView, Cartesian, Filtered
     */
    template<typename Range, typename Func>
    GRBLinExpr sum(const Range& rng, Func&& func) {
        GRBLinExpr expr = 0.0;

        for (const auto& idx : rng) {
            auto term = expr_detail::invoke_on_index(func, idx);
            expr_detail::add_term(expr, std::move(term));
        }

        return expr;
    }

    /**
     * @brief Sums all variables stored in an IndexedVariableSet
     *
     * @details Iterates over all entries in the sparse variable set and accumulates
     *          their GRBVar values. Order follows the internal storage order.
     *
     *          Semantically implements: sum_{k in keys(vSet)} vSet[k].var
     *
     * @param vSet IndexedVariableSet containing sparse variable entries
     *
     * @return GRBLinExpr containing sum of all stored variables
     *
     * @complexity O(n) where n = number of entries in vSet
     * @noexcept
     *
     * @note No ambiguity with sum(Range, Func) due to single argument
     * @note Does not inspect index dimensions; purely structural summation
     *
     * @example
     *     IndexedVariableSet XV = ...;
     *     GRBLinExpr total = dsl::sum(XV);  // Sum all variables
     *
     * @see IndexedVariableSet
     */
    inline GRBLinExpr sum(const IndexedVariableSet& vSet)
    {
        GRBLinExpr expr = 0.0;

        for (const auto& entry : vSet.all()) {
            expr += entry.var;
        }

        return expr;
    }

    /**
     * @brief Sums IndexedVariableSet variables over a specified index domain
     *
     * @details Iterates over each element in the range and looks up the corresponding
     *          variable in the sparse set. Tuple indices are automatically unpacked.
     *
     *          Semantically implements: sum_{idx in Range} vSet(idx...)
     *
     * @tparam Range Iterable type with begin()/end() yielding int or tuple<int,...>
     * @param rng Index domain specifying which variables to sum
     * @param vSet IndexedVariableSet to access
     *
     * @return GRBLinExpr containing sum of variables for specified indices
     *
     * @throws std::out_of_range if any index in rng is not present in vSet
     * @complexity O(n) where n = number of elements in rng
     *
     * @note Avoids trivial lambdas like [&](int i) { return vSet(i); }
     * @note Domain iteration order defines expression order
     *
     * @example
     *     auto expr = dsl::sum(I, XV);      // 1D domain
     *     auto expr = dsl::sum(I * J, XV);  // Cartesian product
     *
     * @see IndexedVariableSet, sum(Range, Func)
     */
    template<typename Range>
    GRBLinExpr sum(const Range& rng, const IndexedVariableSet& vSet)
    {
        auto fn = [&](auto&&... args) -> const GRBVar& {
            return vSet(args...);
        };
        return sum(rng, fn);
    }

    /**
     * @brief Sums all variables in a dense VariableGroup
     *
     * @details Iterates over all leaf nodes in the VariableGroup tree structure
     *          and accumulates their GRBVar values. Traversal order is dim0-major.
     *
     *          Semantically implements: sum over all indices in the dense structure
     *
     * @param vSet VariableGroup containing dense variable structure
     *
     * @return GRBLinExpr containing sum of all stored variables
     *
     * @complexity O(n) where n = total number of variables in vSet
     * @noexcept
     *
     * @note Safe for empty groups (returns 0.0)
     * @note Works for scalars, vectors, matrices, and higher-order tensors
     *
     * @example
     *     VariableGroup X = ...;  // 3x4 matrix of variables
     *     GRBLinExpr total = dsl::sum(X);  // Sum all 12 variables
     *
     * @see VariableGroup
     */
    inline GRBLinExpr sum(const VariableGroup& vSet)
    {
        GRBLinExpr expr = 0.0;

        vSet.forEach([&](const GRBVar& v, const std::vector<int>&) {
            expr += v;
        });

        return expr;
    }

    /**
     * @brief Sums VariableGroup variables over a specified index domain
     *
     * @details Iterates over each element in the range and accesses the corresponding
     *          variable in the dense group. Tuple indices are automatically unpacked.
     *
     *          Semantically implements: sum_{idx in Range} vSet(idx...)
     *
     * @tparam Range Iterable type with begin()/end() yielding int or tuple<int,...>
     * @param rng Index domain specifying which variables to sum
     * @param vSet VariableGroup to access
     *
     * @return GRBLinExpr containing sum of variables for specified indices
     *
     * @throws std::out_of_range if any index is out of bounds
     * @throws std::invalid_argument if index tuple dimension mismatches VariableGroup
     * @complexity O(n) where n = number of elements in rng
     *
     * @note Uses invoke_on_index for automatic tuple unpacking
     *
     * @example
     *     auto expr = dsl::sum(I, X);        // 1D domain over vector
     *     auto expr = dsl::sum(I * J, X);    // 2D domain over matrix
     *
     * @see VariableGroup, sum(Range, Func)
     */
    template<typename Range>
    GRBLinExpr sum(const Range& rng, const VariableGroup& vSet)
    {
        auto fn = [&](auto&&... args) -> const GRBVar& {
            return vSet(args...);
        };

        return sum(rng, fn);
    }

    /**
     * @brief Sums all variables in a VariableContainer (dense or sparse)
     *
     * @details Unified summation that works for both VariableGroup (dense) and
     *          IndexedVariableSet (sparse) storage modes. Iterates over all
     *          variables in the container regardless of storage type.
     *
     *          Semantically implements: sum over all variables in container
     *
     * @param vc VariableContainer holding either dense or sparse variables
     *
     * @return GRBLinExpr containing sum of all stored variables
     *
     * @throws std::runtime_error if container is empty
     * @complexity O(n) where n = total number of variables in container
     *
     * @note Useful when working with VariableTable entries
     * @note Safe for both dense and sparse storage modes
     *
     * @example
     *     // With VariableTable
     *     VariableTable<Vars> vt;
     *     vt.set(Vars::X, std::move(denseVars));
     *     vt.set(Vars::Y, std::move(sparseVars));
     *     
     *     GRBLinExpr sumX = dsl::sum(vt.get(Vars::X));  // Works for dense
     *     GRBLinExpr sumY = dsl::sum(vt.get(Vars::Y));  // Works for sparse
     *
     * @see VariableContainer, VariableGroup, IndexedVariableSet
     */
    inline GRBLinExpr sum(const VariableContainer& vc)
    {
        GRBLinExpr expr = 0.0;

        vc.forEach([&](const GRBVar& v, const std::vector<int>&) {
            expr += v;
        });

        return expr;
    }

    /**
     * @brief Sums VariableContainer variables over a specified index domain
     *
     * @details Iterates over each element in the range and accesses the corresponding
     *          variable in the container. Works for both dense and sparse storage.
     *          Tuple indices are automatically unpacked.
     *
     *          Semantically implements: sum_{idx in Range} vc(idx...)
     *
     * @tparam Range Iterable type with begin()/end() yielding int or tuple<int,...>
     * @param rng Index domain specifying which variables to sum
     * @param vc VariableContainer to access
     *
     * @return GRBLinExpr containing sum of variables for specified indices
     *
     * @throws std::out_of_range if any index is out of bounds or not found
     * @throws std::runtime_error if container is empty
     * @complexity O(n) where n = number of elements in rng
     *
     * @note Uses invoke_on_index for automatic tuple unpacking
     * @note Works with both dense and sparse storage modes
     *
     * @example
     *     auto expr = dsl::sum(I, vc);        // 1D domain
     *     auto expr = dsl::sum(I * J, vc);    // 2D domain
     *
     * @see VariableContainer, sum(Range, Func)
     */
    template<typename Range>
    GRBLinExpr sum(const Range& rng, const VariableContainer& vc)
    {
        auto fn = [&](auto&&... args) -> const GRBVar& {
            return vc.at(args...);
        };

        return sum(rng, fn);
    }

    // ========================================================================
    // QUADRATIC SUM FUNCTIONS
    // ========================================================================

    /**
     * @brief Builds a GRBQuadExpr by summing lambda results over an index domain
     *
     * @details Iterates over any DSL index domain and accumulates the results of
     *          applying the lambda to each index element into a quadratic expression.
     *          Useful for constructing QP objectives like variance minimization.
     *
     *          Semantically implements: sum_{idx in Range} func(idx...)
     *
     * @tparam Range Iterable type with begin()/end() yielding int or tuple<int,...>
     * @tparam Func Callable with signature f(int) or f(int, int, ...) as appropriate
     * @param rng Index domain to iterate over
     * @param func Lambda returning GRBQuadExpr, GRBVar*GRBVar, double, or compatible type
     *
     * @return GRBQuadExpr containing all accumulated terms
     *
     * @throws Propagates any exceptions thrown by func
     * @complexity O(n) where n = number of elements in rng
     *
     * @note Tuple indices are automatically unpacked via invoke_on_index
     * @note Use for QP objectives; for linear expressions, prefer sum()
     *
     * @example
     *     // Portfolio variance: sum_{a,b} sigma[a][b] * x[a] * x[b]
     *     auto variance = dsl::quadSum(A * A, [&](int a, int b) {
     *         return covariance[a][b] * X(a) * X(b);
     *     });
     *
     *     // Quadratic regularization: sum_i x[i]^2
     *     auto regularization = dsl::quadSum(I, [&](int i) {
     *         return X(i) * X(i);
     *     });
     *
     * @see sum, GRBQuadExpr
     */
    template<typename Range, typename Func>
    GRBQuadExpr quadSum(const Range& rng, Func&& func) {
        GRBQuadExpr expr = 0.0;

        for (const auto& idx : rng) {
            expr += expr_detail::invoke_on_index(std::forward<Func>(func), idx);
        }

        return expr;
    }

} // namespace dsl
