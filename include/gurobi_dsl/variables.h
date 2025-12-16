#pragma once
/*
===============================================================================
VARIABLE MANAGEMENT SYSTEM — Gurobi C++ DSL
===============================================================================

OVERVIEW
--------
Implements the variable management layer of the DSL. Provides typed containers
for Gurobi decision variables with safe access patterns, domain-based indexing,
and factory methods for common variable creation patterns.

KEY COMPONENTS
--------------
• VariableGroup — Dense N-dimensional container of GRBVar (scalars, vectors, matrices, tensors)
• IndexedVariableSet — Variables indexed by arbitrary domains (Cartesian products, filtered sets)
• VariableFactory — Unified backend for creating rectangular and domain-based variables
• VariableTable — Enum-keyed registry for organizing variable collections
• Solution Extraction — value(), values() for retrieving optimization results
• Variable Modification — fix(), unfix(), setStart() for bounds and warm starts

DESIGN PHILOSOPHY
-----------------
• Type-safe access with clear failure modes (exceptions for out-of-range)
• Support for both dense rectangular layouts and sparse domain-based indexing
• Minimal overhead; no dynamic allocation during indexed access
• Natural mathematical notation via operator() and variadic at()
• Seamless integration with DSL indexing primitives (IndexList, RangeView, Cartesian)

USAGE EXAMPLES
--------------
    // Rectangular variables
    auto X = VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 10, 20);
    X(3, 5) = ...;  // Access element at (3, 5)

    // Domain-based variables
    auto I = dsl::range(0, 5);
    auto J = dsl::range(0, 3);
    auto Y = VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 10, "Y",
        (I * J) | dsl::filter([](int i, int j) { return i < j; }));

    // Iteration
    X.forEach([](const GRBVar& v, const std::vector<int>& idx) {
        std::cout << v.get(GRB_StringAttr_VarName) << "\n";
    });

    // Solution extraction (after optimization)
    double val = dsl::value(x);              // Single variable
    auto vals = dsl::values(X);              // All variables in group

    // Variable modification
    dsl::fix(x, 1.0);                        // Fix to specific value
    dsl::unfix(x, 0.0, 1.0);                 // Restore bounds
    dsl::setStart(X, initialSolution);       // Warm start

NAMING BEHAVIOR
---------------
Variable names are built via naming.h:
• Rectangular groups: "X_3_5" (using make_name::index(baseName, indices))
• Indexed sets: "X_3_5" (same underscore-based style)
• In release mode, naming can be disabled via naming_enabled()

DEPENDENCIES
------------
• <string>, <vector>, <array>, <stdexcept>, <type_traits>, <format>
• <unordered_map>, <sstream>
• "gurobi_c++.h" — Gurobi C++ API
• "naming.h" — Variable naming utilities
• "enum_utils.h" — Enum introspection for VariableTable

PERFORMANCE NOTES
-----------------
• VariableGroup: O(dims) tree traversal for indexed access
• IndexedVariableSet: O(1) average lookup via hash map
• Memory: Tree nodes for VariableGroup; flat vector + hash map for IndexedVariableSet
• forEach: Linear in number of variables, no allocation per iteration

THREAD SAFETY
-------------
• All containers are value types; concurrent const access is safe
• Modifications to GRBVar require Gurobi model synchronization
• External locking required for concurrent modifications

EXCEPTION SAFETY
----------------
• Construction: Strong guarantee (all or nothing)
• at() / operator(): Throws std::out_of_range or std::runtime_error on invalid access
• try_get(): No-throw guarantee (returns nullptr on missing index)
• forEach: Propagates exceptions from user-provided callback
• value() / values(): Propagates GRBException if model not optimized
• fix() / unfix() / setStart(): Propagates GRBException on invalid operations

===============================================================================
*/

#include <string>
#include <vector>
#include <array>
#include <stdexcept>
#include <type_traits>
#include <format>
#include <unordered_map>
#include <sstream>
#include <variant>

#include "gurobi_c++.h"
#include "naming.h"
#include "enum_utils.h"

namespace dsl {

class VariableFactory; // forward declaration

// ============================================================================
// VARIABLE GROUP
// ============================================================================
/**
 * @class VariableGroup
 * @brief Dense N-dimensional container of GRBVar (scalars, vectors, matrices, tensors)
 *
 * @details Represents decision variables in rectangular layouts:
 *          • Scalar variable (dims == 0)
 *          • 1D array (dims == 1)
 *          • 2D matrix (dims == 2)
 *          • N-dimensional tensor (dims == N)
 *
 *          Internal representation uses a tree of Node objects:
 *          • Leaf nodes (children.empty()): contain a GRBVar
 *          • Container nodes: have children.size() sub-nodes
 *
 * @note Supports both variadic at(i,j,k,...) and vector-based at(vec) access.
 *
 * @example
 *     auto X = VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 10, 20);
 *     X(3, 5);  // Access element at (3, 5)
 *     X.forEach([](const GRBVar& v, const std::vector<int>& idx) {
 *         // Process each variable with its index
 *     });
 *
 * @see VariableFactory
 * @see IndexedVariableSet
 */
class VariableGroup {
    public:
        // ========================================================================
        // NODE STRUCTURE
        // ========================================================================

        /**
         * @struct Node
         * @brief Internal tree node for N-dimensional variable storage
         *
         * @details Leaf nodes store a GRBVar in `scalar`; container nodes
         *          have non-empty `children` vector.
         */
        struct Node {
            GRBVar scalar;                ///< Stored variable (valid if children.empty())
            std::vector<Node> children;   ///< Child nodes (empty for leaf nodes)

            /// @brief Default constructor
            Node() = default;

            /// @brief Construct leaf node from GRBVar (copy)
            explicit Node(const GRBVar& v) : scalar(v) {}

            /// @brief Construct leaf node from GRBVar (move)
            explicit Node(GRBVar&& v) : scalar(std::move(v)) {}

            /// @brief Construct container node with n children
            explicit Node(std::size_t n) : children(n) {}
        };

    private:
        Node root;      ///< Root node of the variable tree
        int  dims = 0;  ///< Number of dimensions (0 == scalar)

    public:
        // ========================================================================
        // CONSTRUCTORS
        // ========================================================================

        /// @brief Default constructor; creates an empty VariableGroup
        VariableGroup() = default;

        /**
         * @brief Construct scalar VariableGroup from GRBVar (copy)
         * @param v Variable to store
         * @post dimension() == 0, isScalar() == true
         */
        explicit VariableGroup(const GRBVar& v)
            : root(Node(v)), dims(0) {
        }

        /**
         * @brief Construct scalar VariableGroup from GRBVar (move)
         * @param v Variable to move
         * @post dimension() == 0, isScalar() == true
         */
        explicit VariableGroup(GRBVar&& v)
            : root(Node(std::move(v))), dims(0) {
        }

        /**
         * @brief Construct from pre-built node tree
         * @param r Root node (moved)
         * @param d Number of dimensions
         * @throws std::invalid_argument if d < 0
         */
        VariableGroup(Node&& r, int d)
            : root(std::move(r)), dims(d) {
            if (d < 0) {
                throw std::invalid_argument(
                    std::format("VariableGroup: negative dimension {}", d));
            }
        }

        // ========================================================================
        // INTROSPECTION
        // ========================================================================

        /// @brief Returns the number of dimensions (0 for scalar)
        /// @noexcept
        [[nodiscard]] int dimension() const noexcept { return dims; }

        /// @brief Returns true if this is a scalar (0-dimensional)
        /// @noexcept
        [[nodiscard]] bool isScalar() const noexcept { return dims == 0; }

        /// @brief Returns true if this has 1 or more dimensions
        /// @noexcept
        [[nodiscard]] bool isMultiDimensional() const noexcept { return dims > 0; }

        /**
         * @brief Returns the shape as a vector of dimension sizes
         * @return Vector of sizes for each dimension
         * @complexity O(dims)
         *
         * @example
         *     auto X = VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 3, 4, 5);
         *     auto shp = X.shape();  // {3, 4, 5}
         */
        [[nodiscard]] std::vector<std::size_t> shape() const {
            std::vector<std::size_t> shp;
            shp.reserve(dims);

            const Node* n = &root;
            for (int d = 0; d < dims; ++d) {
                if (n->children.empty()) break;
                shp.push_back(n->children.size());
                n = &n->children[0];
            }
            return shp;
        }

        /**
         * @brief Returns the size of a specific dimension
         * @param dim Dimension index (0-based)
         * @return Number of elements in that dimension
         * @throws std::out_of_range if dim < 0 or dim >= dimension()
         * @complexity O(dim)
         */
        [[nodiscard]] std::size_t size(int dim) const {
            if (dim < 0 || dim >= dims) {
                throw std::out_of_range(
                    std::format("VariableGroup::size: dim {} out of range [0, {})",
                        dim, dims));
            }

            const Node* n = &root;
            for (int d = 0; d < dim; ++d) {
                n = &n->children[0];
            }
            return n->children.size();
        }

        /**
         * @brief Returns the total number of variables in the group
         * @return Total count of variables (product of all dimension sizes)
         * @complexity O(n) where n = total number of variables
         *
         * @note For scalars, returns 1.
         * @note Useful for validation in fixAll/setStartAll.
         *
         * @example
         *     auto X = VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 3, 4);
         *     std::size_t n = X.count();  // 12
         */
        [[nodiscard]] std::size_t count() const {
            std::size_t total = 0;
            forEach([&](const GRBVar&, const std::vector<int>&) { ++total; });
            return total;
        }

        // ========================================================================
        // ACCESS (VARIADIC INDICES)
        // ========================================================================

        /**
         * @brief Access variable by variadic indices
         * @tparam Indices Integral index types
         * @param idx Index values for each dimension
         * @return Mutable reference to the GRBVar at the specified position
         * @throws std::runtime_error if number of indices != dimension()
         * @throws std::out_of_range if any index is out of bounds
         * @complexity O(dims)
         *
         * @example
         *     auto X = VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 10, 20);
         *     GRBVar& v = X.at(3, 5);
         */
        template<typename... Indices>
        GRBVar& at(Indices... idx) {
            static_assert((std::is_integral_v<Indices> && ...),
                "VariableGroup::at: indices must be integral");

            constexpr std::size_t N = sizeof...(idx);
            if (static_cast<int>(N) != dims) {
                throw std::runtime_error(
                    std::format("VariableGroup::at: expected {} indices, got {}",
                        dims, N));
            }
            return atRec(root, idx...);
        }

        /// @brief Const version of at()
        template<typename... Indices>
        const GRBVar& at(Indices... idx) const {
            return const_cast<VariableGroup*>(this)->at(idx...);
        }

        /// @brief Alias for at() using function call syntax
        template<typename... Indices>
        GRBVar& operator()(Indices... idx) { return at(idx...); }

        /// @brief Const alias for at() using function call syntax
        template<typename... Indices>
        const GRBVar& operator()(Indices... idx) const { return at(idx...); }

        /**
         * @brief Access the scalar value (only valid for 0-dimensional groups)
         * @return Mutable reference to the stored GRBVar
         * @throws std::runtime_error if dimension() != 0
         * @complexity Constant time
         */
        GRBVar& scalar() {
            if (dims != 0) {
                throw std::runtime_error(
                    std::format("VariableGroup::scalar: group is {}-dimensional",
                        dims));
            }
            return root.scalar;
        }

        /// @brief Const version of scalar()
        const GRBVar& scalar() const {
            if (dims != 0) {
                throw std::runtime_error(
                    std::format("VariableGroup::scalar: group is {}-dimensional",
                        dims));
            }
            return root.scalar;
        }

        // ========================================================================
        // ACCESS (VECTOR-BASED INDICES)
        // ========================================================================

        /**
         * @brief Access variable by vector of indices (N-dimensional)
         * @param idxVec Vector of index values
         * @return Mutable reference to the GRBVar at the specified position
         * @throws std::runtime_error if idxVec.size() != dimension()
         * @throws std::out_of_range if any index is out of bounds
         * @complexity O(dims)
         *
         * @note For scalars (dims == 0), idxVec must be empty ({}).
         * @note Especially useful for generic algorithms and domain-based operations.
         *
         * @example
         *     for (auto&& elem : domain) {
         *         std::vector<int> idx = ...;
         *         expr += X.at(idx);
         *     }
         */
        GRBVar& at(const std::vector<int>& idxVec) {
            if (static_cast<int>(idxVec.size()) != dims) {
                throw std::runtime_error(
                    std::format("VariableGroup::at(vec): expected {} indices, got {}",
                        dims, idxVec.size()));
            }
            return atVecRec(root, idxVec, 0);
        }

        /// @brief Const version of at(vector)
        const GRBVar& at(const std::vector<int>& idxVec) const {
            return const_cast<VariableGroup*>(this)->at(idxVec);
        }

        // ========================================================================
        // ITERATION
        // ========================================================================

        /**
         * @brief Iterate over all GRBVar entries with their indices
         * @tparam Fn Callable with signature (GRBVar&, const std::vector<int>&)
         * @param fn Function to call for each variable
         * @complexity O(total number of variables)
         *
         * @example
         *     X.forEach([](GRBVar& v, const std::vector<int>& idx) {
         *         std::cout << "X";
         *         for (int i : idx) std::cout << "_" << i;
         *         std::cout << " = " << v.get(GRB_DoubleAttr_X) << "\n";
         *     });
         */
        template<typename Fn>
        void forEach(Fn&& fn) {
            std::vector<int> idx;
            idx.reserve(dims);
            forEachRec(root, idx, fn);
        }

        /// @brief Const version of forEach()
        template<typename Fn>
        void forEach(Fn&& fn) const {
            std::vector<int> idx;
            idx.reserve(dims);
            forEachRecConst(root, idx, fn);
        }

    private:
        // ========================================================================
        // PRIVATE HELPERS
        // ========================================================================

        /// @brief Leaf case for variadic atRec
        GRBVar& atRec(Node& n) {
            return n.scalar;
        }

        /// @brief Recursive descent for variadic indices
        template<typename First, typename... Rest>
        GRBVar& atRec(Node& n, First i, Rest... rest) {
            if (i < 0) {
                throw std::out_of_range(
                    std::format("VariableGroup::atRec: negative index {}", i));
            }

            std::size_t idx = static_cast<std::size_t>(i);
            if (idx >= n.children.size()) {
                throw std::out_of_range(
                    std::format("VariableGroup::atRec: index {} out of range [0, {})",
                        i, n.children.size()));
            }

            return atRec(n.children[idx], rest...);
        }

        /// @brief Recursive descent for vector-based indices
        GRBVar& atVecRec(Node& n, const std::vector<int>& idx, std::size_t d) {
            if (d == idx.size()) {
                return n.scalar;
            }

            int i = idx[d];
            if (i < 0) {
                throw std::out_of_range(
                    std::format("VariableGroup::atVecRec: negative index {} at dim {}",
                        i, d));
            }

            std::size_t pos = static_cast<std::size_t>(i);
            if (pos >= n.children.size()) {
                throw std::out_of_range(
                    std::format("VariableGroup::atVecRec: index {} out of range [0, {}) at dim {}",
                        i, n.children.size(), d));
            }

            return atVecRec(n.children[pos], idx, d + 1);
        }

        /// @brief Recursive forEach implementation (mutable)
        template<typename Fn>
        static void forEachRec(Node& n,
            std::vector<int>& idx,
            Fn& fn) {
            if (n.children.empty()) {
                fn(n.scalar, idx);
                return;
            }

            for (std::size_t i = 0; i < n.children.size(); ++i) {
                idx.push_back(static_cast<int>(i));
                forEachRec(n.children[i], idx, fn);
                idx.pop_back();
            }
        }

        /// @brief Recursive forEach implementation (const)
        template<typename Fn>
        static void forEachRecConst(const Node& n,
            std::vector<int>& idx,
            Fn& fn)
        {
            if (n.children.empty()) {
                fn(n.scalar, idx);
                return;
            }

            for (std::size_t i = 0; i < n.children.size(); ++i) {
                idx.push_back(static_cast<int>(i));
                forEachRecConst(n.children[i], idx, fn);
                idx.pop_back();
            }
        }

        friend class VariableFactory;
    };


    // ============================================================================
    // INDEXED VARIABLE SET
    // ============================================================================
    /**
     * @class IndexedVariableSet
     * @brief Variables indexed by arbitrary domains (Cartesian products, filtered sets)
     *
     * @details Stores a flat list of entries where each entry contains a GRBVar
     *          and its associated index vector. Provides O(1) lookup via hash map.
     *
     *          The domain can be any iterable whose elements are either:
     *          • An int (1-dimensional)
     *          • A tuple-like object (i, j, k, ...) for N-dimensional
     *
     * @note Unlike VariableGroup, does not require rectangular domains.
     *
     * @example
     *     auto I = dsl::range(0, 5);
     *     auto J = dsl::range(0, 3);
     *
     *     // Full Cartesian
     *     auto X = VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "X", I * J);
     *
     *     // Filtered
     *     auto Y = VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 10, "Y",
     *         (I * J) | dsl::filter([](int i, int j) { return i < j; }));
     *
     *     // Access
     *     X(1, 2);              // By variadic indices
     *     X.at(1, 2);           // Same as above
     *     X.try_get(1, 2);      // Returns nullptr if not found
     *
     * @see VariableFactory::addIndexed
     * @see VariableGroup
     */
    class IndexedVariableSet {
    public:
        // ========================================================================
        // ENTRY STRUCTURE
        // ========================================================================

        /**
         * @struct Entry
         * @brief Stores a GRBVar with its associated index vector
         */
        struct Entry {
            GRBVar var;               ///< The decision variable
            std::vector<int> index;   ///< Index tuple for this variable
        };

    private:
        std::vector<Entry> entries;                        ///< Flat list of all entries
        std::unordered_map<std::string, std::size_t> indexMap;  ///< Hash map for O(1) lookup

        /// @brief Create lookup key from index vector
        static std::string makeKeyFromVector(const std::vector<int>& idx) {
            std::ostringstream oss;
            for (std::size_t k = 0; k < idx.size(); ++k) {
                if (k > 0) {
                    oss << '_';
                }
                oss << static_cast<long long>(idx[k]);
            }
            return oss.str();
        }

        /// @brief Create lookup key from variadic indices
        template<typename... I>
        static std::string makeKey(I... idx) {
            static_assert((std::is_integral_v<I> && ...),
                "IndexedVariableSet::makeKey: indices must be integral");
            std::ostringstream oss;
            bool first = true;
            ((oss << (first ? (first = false, "") : "_")
                << static_cast<long long>(idx)), ...);
            return oss.str();
        }

        /// @brief Add an entry to the set
        void addEntry(GRBVar&& v, std::vector<int>&& idx) {
            std::size_t pos = entries.size();
            std::string key = makeKeyFromVector(idx);
            indexMap.emplace(std::move(key), pos);
            entries.push_back(Entry{ std::move(v), std::move(idx) });
        }

    public:
        // ========================================================================
        // CONSTRUCTORS
        // ========================================================================

        /// @brief Default constructor; creates an empty IndexedVariableSet
        IndexedVariableSet() = default;

        // ========================================================================
        // INTROSPECTION
        // ========================================================================

        /// @brief Returns the number of variables in the set
        /// @noexcept
        [[nodiscard]] std::size_t size() const noexcept { return entries.size(); }

        /// @brief Returns true if the set is empty
        /// @noexcept
        [[nodiscard]] bool empty() const noexcept { return entries.empty(); }

        // ========================================================================
        // ITERATION
        // ========================================================================

        using iterator = std::vector<Entry>::iterator;
        using const_iterator = std::vector<Entry>::const_iterator;

        /// @brief Begin iterator
        /// @noexcept
        iterator begin() noexcept { return entries.begin(); }

        /// @brief End iterator
        /// @noexcept
        iterator end() noexcept { return entries.end(); }

        /// @brief Begin iterator (const)
        /// @noexcept
        const_iterator begin() const noexcept { return entries.begin(); }

        /// @brief End iterator (const)
        /// @noexcept
        const_iterator end() const noexcept { return entries.end(); }

        /// @brief Access the underlying entry vector
        /// @noexcept
        const std::vector<Entry>& all() const noexcept { return entries; }

        // ========================================================================
        // ACCESS (VARIADIC INDICES)
        // ========================================================================

        /**
         * @brief Access variable by variadic indices
         * @tparam I Integral index types
         * @param idx Index values
         * @return Mutable reference to the GRBVar
         * @throws std::out_of_range if index not found
         * @complexity O(1) average (hash lookup)
         */
        template<typename... I>
        GRBVar& at(I... idx) {
            std::string key = makeKey(idx...);
            auto it = indexMap.find(key);
            if (it == indexMap.end()) {
                throw std::out_of_range(
                    std::format("IndexedVariableSet::at: index {} not found", key));
            }
            return entries[it->second].var;
        }

        /// @brief Const version of at()
        template<typename... I>
        const GRBVar& at(I... idx) const {
            return const_cast<IndexedVariableSet*>(this)->at(idx...);
        }

        /// @brief Alias for at() using function call syntax
        template<typename... I>
        GRBVar& operator()(I... idx) { return at(idx...); }

        /// @brief Const alias for at() using function call syntax
        template<typename... I>
        const GRBVar& operator()(I... idx) const { return at(idx...); }

        /**
         * @brief Try to access variable by variadic indices
         * @tparam I Integral index types
         * @param idx Index values
         * @return Pointer to GRBVar, or nullptr if not found
         * @complexity O(1) average (hash lookup)
         * @noexcept
         */
        template<typename... I>
        GRBVar* try_get(I... idx) noexcept {
            try {
                std::string key = makeKey(idx...);
                auto it = indexMap.find(key);
                if (it == indexMap.end()) {
                    return nullptr;
                }
                return &entries[it->second].var;
            }
            catch (...) {
                return nullptr;
            }
        }

        /// @brief Const version of try_get()
        template<typename... I>
        const GRBVar* try_get(I... idx) const noexcept {
            return const_cast<IndexedVariableSet*>(this)->try_get(idx...);
        }

        // ========================================================================
        // ACCESS (VECTOR-BASED INDICES)
        // ========================================================================

        /**
         * @brief Try to access variable by index vector
         * @param idxVec Vector of index values
         * @return Pointer to GRBVar, or nullptr if not found
         * @complexity O(1) average (hash lookup)
         * @noexcept
         *
         * @note idxVec can have any length (matches how the domain was built).
         */
        GRBVar* try_get(const std::vector<int>& idxVec) noexcept {
            try {
                std::string key = makeKeyFromVector(idxVec);
                auto it = indexMap.find(key);
                if (it == indexMap.end()) {
                    return nullptr;
                }
                return &entries[it->second].var;
            }
            catch (...) {
                return nullptr;
            }
        }

        /// @brief Const version of try_get(vector)
        const GRBVar* try_get(const std::vector<int>& idxVec) const noexcept {
            return const_cast<IndexedVariableSet*>(this)->try_get(idxVec);
        }

        /**
         * @brief Access variable by index vector
         * @param idxVec Vector of index values
         * @return Reference to the GRBVar
         * @throws std::runtime_error if index not found
         * @complexity O(1) average (hash lookup)
         *
         * @note Useful when domain membership is guaranteed by construction.
         */
        GRBVar& at(const std::vector<int>& idxVec) {
            GRBVar* v = try_get(idxVec);
            if (!v) {
                std::ostringstream oss;
                oss << "IndexedVariableSet::at(vec): index [";
                for (std::size_t i = 0; i < idxVec.size(); ++i) {
                    if (i > 0) oss << ",";
                    oss << idxVec[i];
                }
                oss << "] not found";
                throw std::runtime_error(oss.str());
            }
            return *v;
        }

        /// @brief Const version of at(vector)
        const GRBVar& at(const std::vector<int>& idxVec) const {
            return const_cast<IndexedVariableSet*>(this)->at(idxVec);
        }

        // ========================================================================
        // FOREACH ITERATION
        // ========================================================================

        /**
         * @brief Iterate over all entries with their indices
         * @tparam Fn Callable with signature (GRBVar&, const std::vector<int>&)
         * @param fn Function to call for each entry
         * @complexity O(size())
         *
         * @example
         *     X.forEach([](GRBVar& v, const std::vector<int>& idx) {
         *         std::cout << "X";
         *         for (int i : idx) std::cout << "_" << i;
         *         std::cout << " = " << v.get(GRB_DoubleAttr_X) << "\n";
         *     });
         */
        template<typename Fn>
        void forEach(Fn&& fn) {
            for (auto& e : entries) {
                fn(e.var, e.index);
            }
        }

        /// @brief Const version of forEach()
        template<typename Fn>
        void forEach(Fn&& fn) const
        {
            for (const auto& e : entries) {
                fn(e.var, e.index);
            }
        }

    private:
        friend class VariableFactory;
    };


    // ============================================================================
    // VARIABLE CONTAINER (UNIFIED DENSE/SPARSE)
    // ============================================================================
    /**
     * @class VariableContainer
     * @brief Unified container that can hold either dense (VariableGroup) or sparse (IndexedVariableSet) variables
     *
     * @details Provides a single type that can operate in two modes:
     *          • Dense mode: Wraps a VariableGroup for rectangular variable arrays
     *          • Sparse mode: Wraps an IndexedVariableSet for domain-based variables
     *
     *          This enables VariableTable to hold mixed collections without requiring
     *          separate tables for dense and sparse variables.
     *
     * @note Mode is determined at construction and cannot be changed afterward.
     *
     * @example
     *     VariableContainer dense(VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 10, 20));
     *     VariableContainer sparse(VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "Y", I * J));
     *
     *     // Unified access works for both
     *     dense(3, 5);   // Access dense variable
     *     sparse(1, 2);  // Access sparse variable
     *
     *     // Mode-specific access
     *     if (dense.isDense()) {
     *         auto& group = dense.asGroup();
     *         std::cout << "Shape: " << group.shape()[0] << "\n";
     *     }
     *
     * @see VariableGroup
     * @see IndexedVariableSet
     * @see VariableTable
     */
    class VariableContainer {
    public:
        /// @brief Storage mode enumeration
        enum class Mode { Empty, Dense, Sparse };

    private:
        std::variant<std::monostate, VariableGroup, IndexedVariableSet> storage_;

    public:
        // ========================================================================
        // CONSTRUCTORS
        // ========================================================================

        /// @brief Default constructor; creates an empty container
        VariableContainer() = default;

        /**
         * @brief Construct from a VariableGroup (dense mode)
         * @param group Dense variable collection to store
         * @post mode() == Mode::Dense
         */
        VariableContainer(VariableGroup&& group)
            : storage_(std::move(group)) {}

        /**
         * @brief Construct from a VariableGroup (copy)
         * @param group Dense variable collection to copy
         * @post mode() == Mode::Dense
         */
        VariableContainer(const VariableGroup& group)
            : storage_(group) {}

        /**
         * @brief Construct from an IndexedVariableSet (sparse mode)
         * @param set Sparse variable collection to store
         * @post mode() == Mode::Sparse
         */
        VariableContainer(IndexedVariableSet&& set)
            : storage_(std::move(set)) {}

        /**
         * @brief Construct from an IndexedVariableSet (copy)
         * @param set Sparse variable collection to copy
         * @post mode() == Mode::Sparse
         */
        VariableContainer(const IndexedVariableSet& set)
            : storage_(set) {}

        /**
         * @brief Construct scalar from GRBVar (dense mode, 0-D)
         * @param v Scalar variable to store
         * @post mode() == Mode::Dense, asGroup().isScalar() == true
         */
        VariableContainer(const GRBVar& v)
            : storage_(VariableGroup(v)) {}

        /**
         * @brief Construct scalar from GRBVar (move)
         * @param v Scalar variable to move
         * @post mode() == Mode::Dense, asGroup().isScalar() == true
         */
        VariableContainer(GRBVar&& v)
            : storage_(VariableGroup(std::move(v))) {}

        // ========================================================================
        // MODE INTROSPECTION
        // ========================================================================

        /// @brief Returns the current storage mode
        /// @noexcept
        [[nodiscard]] Mode mode() const noexcept {
            if (std::holds_alternative<std::monostate>(storage_)) return Mode::Empty;
            if (std::holds_alternative<VariableGroup>(storage_)) return Mode::Dense;
            return Mode::Sparse;
        }

        /// @brief Returns true if container is empty
        /// @noexcept
        [[nodiscard]] bool isEmpty() const noexcept { return mode() == Mode::Empty; }

        /// @brief Returns true if container holds a VariableGroup (dense)
        /// @noexcept
        [[nodiscard]] bool isDense() const noexcept { return mode() == Mode::Dense; }

        /// @brief Returns true if container holds an IndexedVariableSet (sparse)
        /// @noexcept
        [[nodiscard]] bool isSparse() const noexcept { return mode() == Mode::Sparse; }

        // ========================================================================
        // MODE-SPECIFIC ACCESS
        // ========================================================================

        /**
         * @brief Get the underlying VariableGroup (dense mode only)
         * @return Mutable reference to the VariableGroup
         * @throws std::runtime_error if not in dense mode
         */
        VariableGroup& asGroup() {
            if (!isDense()) {
                throw std::runtime_error("VariableContainer::asGroup: not in dense mode");
            }
            return std::get<VariableGroup>(storage_);
        }

        /// @brief Const version of asGroup()
        const VariableGroup& asGroup() const {
            if (!isDense()) {
                throw std::runtime_error("VariableContainer::asGroup: not in dense mode");
            }
            return std::get<VariableGroup>(storage_);
        }

        /**
         * @brief Get the underlying IndexedVariableSet (sparse mode only)
         * @return Mutable reference to the IndexedVariableSet
         * @throws std::runtime_error if not in sparse mode
         */
        IndexedVariableSet& asIndexed() {
            if (!isSparse()) {
                throw std::runtime_error("VariableContainer::asIndexed: not in sparse mode");
            }
            return std::get<IndexedVariableSet>(storage_);
        }

        /// @brief Const version of asIndexed()
        const IndexedVariableSet& asIndexed() const {
            if (!isSparse()) {
                throw std::runtime_error("VariableContainer::asIndexed: not in sparse mode");
            }
            return std::get<IndexedVariableSet>(storage_);
        }

        // ========================================================================
        // UNIFIED ACCESS
        // ========================================================================

        /**
         * @brief Access variable by variadic indices (works for both modes)
         * @tparam I Integral index types
         * @param idx Index values
         * @return Mutable reference to the GRBVar
         * @throws std::runtime_error if container is empty or index invalid
         */
        template<typename... I>
        GRBVar& at(I... idx) {
            switch (mode()) {
                case Mode::Dense:
                    return std::get<VariableGroup>(storage_).at(idx...);
                case Mode::Sparse:
                    return std::get<IndexedVariableSet>(storage_).at(idx...);
                default:
                    throw std::runtime_error("VariableContainer::at: container is empty");
            }
        }

        /// @brief Const version of at()
        template<typename... I>
        const GRBVar& at(I... idx) const {
            return const_cast<VariableContainer*>(this)->at(idx...);
        }

        /// @brief Alias for at() using function call syntax
        template<typename... I>
        GRBVar& operator()(I... idx) { return at(idx...); }

        /// @brief Const alias for at() using function call syntax
        template<typename... I>
        const GRBVar& operator()(I... idx) const { return at(idx...); }

        /**
         * @brief Access variable by vector of indices (works for both modes)
         * @param idxVec Vector of index values
         * @return Mutable reference to the GRBVar
         * @throws std::runtime_error if container is empty or index invalid
         */
        GRBVar& at(const std::vector<int>& idxVec) {
            switch (mode()) {
                case Mode::Dense:
                    return std::get<VariableGroup>(storage_).at(idxVec);
                case Mode::Sparse:
                    return std::get<IndexedVariableSet>(storage_).at(idxVec);
                default:
                    throw std::runtime_error("VariableContainer::at: container is empty");
            }
        }

        /// @brief Const version of at(vector)
        const GRBVar& at(const std::vector<int>& idxVec) const {
            return const_cast<VariableContainer*>(this)->at(idxVec);
        }

        /**
         * @brief Try to access variable by variadic indices
         * @tparam I Integral index types
         * @param idx Index values
         * @return Pointer to GRBVar, or nullptr if not found/empty
         * @noexcept
         */
        template<typename... I>
        GRBVar* try_get(I... idx) noexcept {
            try {
                switch (mode()) {
                    case Mode::Dense:
                        return &std::get<VariableGroup>(storage_).at(idx...);
                    case Mode::Sparse:
                        return std::get<IndexedVariableSet>(storage_).try_get(idx...);
                    default:
                        return nullptr;
                }
            } catch (...) {
                return nullptr;
            }
        }

        /// @brief Const version of try_get()
        template<typename... I>
        const GRBVar* try_get(I... idx) const noexcept {
            return const_cast<VariableContainer*>(this)->try_get(idx...);
        }

        /**
         * @brief Access the scalar value (dense mode, 0-D only)
         * @return Mutable reference to the stored GRBVar
         * @throws std::runtime_error if not dense or not scalar
         */
        GRBVar& scalar() {
            if (!isDense()) {
                throw std::runtime_error("VariableContainer::scalar: not in dense mode");
            }
            return std::get<VariableGroup>(storage_).scalar();
        }

        /// @brief Const version of scalar()
        const GRBVar& scalar() const {
            if (!isDense()) {
                throw std::runtime_error("VariableContainer::scalar: not in dense mode");
            }
            return std::get<VariableGroup>(storage_).scalar();
        }

        // ========================================================================
        // UNIFIED INTROSPECTION
        // ========================================================================

        /**
         * @brief Returns the total number of variables in the container
         * @return Total count of variables
         * @throws std::runtime_error if container is empty
         */
        [[nodiscard]] std::size_t count() const {
            switch (mode()) {
                case Mode::Dense:
                    return std::get<VariableGroup>(storage_).count();
                case Mode::Sparse:
                    return std::get<IndexedVariableSet>(storage_).size();
                default:
                    throw std::runtime_error("VariableContainer::count: container is empty");
            }
        }

        /**
         * @brief Returns true if this is a scalar (dense mode, 0-D)
         * @return true if dense and 0-dimensional, false otherwise
         * @noexcept
         */
        [[nodiscard]] bool isScalar() const noexcept {
            if (!isDense()) return false;
            return std::get<VariableGroup>(storage_).isScalar();
        }

        // ========================================================================
        // UNIFIED ITERATION
        // ========================================================================

        /**
         * @brief Iterate over all variables with their indices
         * @tparam Fn Callable with signature (GRBVar&, const std::vector<int>&)
         * @param fn Function to call for each variable
         * @throws std::runtime_error if container is empty
         */
        template<typename Fn>
        void forEach(Fn&& fn) {
            switch (mode()) {
                case Mode::Dense:
                    std::get<VariableGroup>(storage_).forEach(std::forward<Fn>(fn));
                    break;
                case Mode::Sparse:
                    std::get<IndexedVariableSet>(storage_).forEach(std::forward<Fn>(fn));
                    break;
                default:
                    throw std::runtime_error("VariableContainer::forEach: container is empty");
            }
        }

        /// @brief Const version of forEach()
        template<typename Fn>
        void forEach(Fn&& fn) const {
            switch (mode()) {
                case Mode::Dense:
                    std::get<VariableGroup>(storage_).forEach(std::forward<Fn>(fn));
                    break;
                case Mode::Sparse:
                    std::get<IndexedVariableSet>(storage_).forEach(std::forward<Fn>(fn));
                    break;
                default:
                    throw std::runtime_error("VariableContainer::forEach: container is empty");
            }
        }
    };


    // ============================================================================
    // INTERNAL HELPERS
    // ============================================================================
    /**
     * @namespace variable_detail
     * @brief Internal utilities for addIndexed domain processing
     */
    namespace variable_detail {

        /// @brief Trait to detect tuple-like types (have std::tuple_size)
        template<typename T, typename = void>
        struct is_tuple_like : std::false_type {};

        template<typename T>
        struct is_tuple_like<T,
            std::void_t<decltype(std::tuple_size<T>::value)>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_tuple_like_v = is_tuple_like<T>::value;

        /// @brief Convert domain element to index vector
        template<typename Idx>
        std::vector<int> index_to_vector(const Idx& idx) {
            using Raw = std::remove_cvref_t<Idx>;
            if constexpr (is_tuple_like_v<Raw>) {
                std::vector<int> v;
                v.reserve(static_cast<std::size_t>(std::tuple_size_v<Raw>));
                std::apply(
                    [&](auto&&... args) {
                        (v.push_back(static_cast<int>(args)), ...);
                    },
                    idx
                );
                return v;
            }
            else {
                static_assert(std::is_integral_v<Raw>,
                    "variable_detail::index_to_vector: index must be int or tuple");
                return std::vector<int>{ static_cast<int>(idx) };
            }
        }

    } // namespace variable_detail


    // ============================================================================
    // VARIABLE FACTORY
    // ============================================================================
    /**
     * @class VariableFactory
     * @brief Unified backend for creating rectangular and domain-based variables
     *
     * @details Provides static factory methods for variable creation:
     *          • add(): Creates scalar or rectangular N-D VariableGroup
     *          • addIndexed(): Creates IndexedVariableSet from domain objects
     *
     * @note Variables are named using naming.h utilities when naming_enabled() is true.
     *
     * @example
     *     // Scalar
     *     GRBVar x = VariableFactory::add(model, GRB_BINARY, 0, 1, "x");
     *
     *     // 2D matrix
     *     auto X = VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 10, 20);
     *
     *     // Domain-based
     *     auto Y = VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 10, "Y",
     *         I * J | dsl::filter([](int i, int j) { return i < j; }));
     *
     * @see VariableGroup
     * @see IndexedVariableSet
     */
    class VariableFactory {
    public:
        using Node = VariableGroup::Node;

        // ========================================================================
        // RECTANGULAR VARIABLES
        // ========================================================================

        /**
         * @brief Create scalar or rectangular N-D VariableGroup
         * @tparam Sizes Integral dimension sizes
         * @param model Gurobi model
         * @param vtype Variable type (GRB_BINARY, GRB_CONTINUOUS, GRB_INTEGER)
         * @param lb Lower bound
         * @param ub Upper bound
         * @param baseName Base name for variable naming
         * @param sizes Dimension sizes (empty for scalar)
         * @return GRBVar for scalar, VariableGroup for N-D
         * @throws std::invalid_argument if any size is negative
         *
         * @example
         *     // Scalar
         *     GRBVar x = VariableFactory::add(model, GRB_BINARY, 0, 1, "x");
         *     // 2D matrix
         *     auto X = VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 10, 20);
         */
        template<typename... Sizes>
        static auto add(GRBModel& model,
            int vtype,
            double lb,
            double ub,
            const std::string& baseName,
            Sizes... sizes)
        {
            static_assert((std::is_integral_v<Sizes> && ...),
                "VariableFactory::add: sizes must be integral");

            if constexpr (sizeof...(sizes) == 0) {
                std::string name = ::make_name::concat(baseName);
                return addVarOpt(model, lb, ub, vtype, name);
            }
            else {
                std::vector<int> indices;
                Node root = addNodeImpl(model, vtype, lb, ub,
                    baseName, indices, sizes...);

                return VariableGroup(std::move(root),
                    static_cast<int>(sizeof...(sizes)));
            }
        }

        // ========================================================================
        // DOMAIN-BASED VARIABLES
        // ========================================================================

        /**
         * @brief Create IndexedVariableSet from domain objects
         * @tparam Domain Iterable domain type
         * @param model Gurobi model
         * @param vtype Variable type
         * @param lb Lower bound
         * @param ub Upper bound
         * @param baseName Base name for variable naming
         * @param domain Iterable domain (IndexList, RangeView, Cartesian, Filtered)
         * @return IndexedVariableSet with one variable per domain element
         *
         * @example
         *     auto I = dsl::range(0, 5);
         *     auto J = dsl::range(0, 3);
         *     auto X = VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "X", I * J);
         */
        template<typename Domain>
        static IndexedVariableSet addIndexed(GRBModel& model,
            int vtype,
            double lb,
            double ub,
            const std::string& baseName,
            const Domain& domain)
        {
            IndexedVariableSet result;

            for (auto&& rawIdx : domain) {
                auto idxVec = variable_detail::index_to_vector(rawIdx);

                std::string name =
                    naming_enabled()
                    ? ::make_name::index(baseName, idxVec)
                    : std::string{};

                GRBVar v = addVarOpt(model, lb, ub, vtype, name);
                result.addEntry(std::move(v), std::move(idxVec));
            }

            return result;
        }

    private:
        // ========================================================================
        // PRIVATE HELPERS
        // ========================================================================

        /// @brief Recursive tree construction for rectangular variables
        template<typename SizeType, typename... Sizes>
        static Node addNodeImpl(GRBModel& model,
            int vtype,
            double lb,
            double ub,
            const std::string& baseName,
            std::vector<int>& idx,
            SizeType n,
            Sizes... sizes)
        {
            static_assert(std::is_integral_v<SizeType>,
                "VariableFactory::addNodeImpl: size must be integral");

            if (n < 0) {
                throw std::invalid_argument(
                    std::format("VariableFactory::addNodeImpl: negative size {}", n));
            }

            Node node(static_cast<std::size_t>(n));

            for (SizeType i = 0; i < n; ++i) {
                idx.push_back(static_cast<int>(i));

                if constexpr (sizeof...(Sizes) == 0) {
                    std::string name = ::make_name::index(baseName, idx);
                    node.children[static_cast<std::size_t>(i)] =
                        Node(addVarOpt(model, lb, ub, vtype, name));
                }
                else {
                    node.children[static_cast<std::size_t>(i)] =
                        addNodeImpl(model, vtype, lb, ub,
                            baseName, idx, sizes...);
                }

                idx.pop_back();
            }

            return node;
        }

        /// @brief Create GRBVar with optional naming
        static inline GRBVar addVarOpt(GRBModel& model,
            double lb,
            double ub,
            int vtype,
            const std::string& name)
        {
            if constexpr (naming_enabled()) {
                return model.addVar(lb, ub, 0.0, vtype, name);
            }
            else {
                return model.addVar(lb, ub, 0.0, vtype);
            }
        }
    };


    // ============================================================================
    // VARIABLE TABLE
    // ============================================================================
    /**
     * @class VariableTable
     * @brief Enum-keyed registry for organizing variable collections
     *
     * @tparam EnumT Enum class with COUNT sentinel
     * @tparam MAX Number of enum values, defaults to EnumT::COUNT
     *
     * @details Provides a fixed-size array of VariableContainer objects indexed by
     *          enum values. Each entry can independently hold either dense (VariableGroup)
     *          or sparse (IndexedVariableSet) variables, enabling mixed storage in a
     *          single table.
     *
     * @example
     *     DECLARE_ENUM_WITH_COUNT(Vars, X, Y, Z);
     *
     *     VariableTable<Vars> vt;
     *
     *     // Mix dense and sparse in the same table
     *     auto X = VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 10, 20);
     *     auto Y = VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "Y",
     *         (I * J) | dsl::filter([](int i, int j) { return i < j; }));
     *
     *     vt.set(Vars::X, std::move(X));  // Dense
     *     vt.set(Vars::Y, std::move(Y));  // Sparse
     *
     *     // Unified access works for both
     *     GRBVar& v1 = vt.var(Vars::X, 3, 5);
     *     GRBVar& v2 = vt.var(Vars::Y, 1, 2);
     *
     *     // Mode-specific access when needed
     *     if (vt.get(Vars::X).isDense()) {
     *         auto shape = vt.get(Vars::X).asGroup().shape();
     *     }
     *
     * @see VariableContainer
     * @see VariableGroup
     * @see IndexedVariableSet
     */
    template<
        typename EnumT,
        std::size_t MAX = static_cast<std::size_t>(EnumT::COUNT)>
    class VariableTable {
    private:
        std::array<VariableContainer, MAX> table_;  ///< Fixed-size array of containers

    public:
        // ========================================================================
        // SETTERS
        // ========================================================================

        /**
         * @brief Set a container by enum key
         * @param key Enum key
         * @param container VariableContainer to store (moved)
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, VariableContainer&& container) {
            std::size_t idx = static_cast<std::size_t>(key);
            if (idx >= MAX) {
                throw std::out_of_range(
                    std::format("VariableTable::set: key {} >= {}", idx, MAX));
            }
            table_[idx] = std::move(container);
        }

        /**
         * @brief Set a VariableGroup by enum key (dense mode)
         * @param key Enum key
         * @param group VariableGroup to store (moved)
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, VariableGroup&& group) {
            set(key, VariableContainer(std::move(group)));
        }

        /**
         * @brief Set a VariableGroup by enum key (copy)
         * @param key Enum key
         * @param group VariableGroup to copy
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, const VariableGroup& group) {
            set(key, VariableContainer(group));
        }

        /**
         * @brief Set an IndexedVariableSet by enum key (sparse mode)
         * @param key Enum key
         * @param set IndexedVariableSet to store (moved)
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, IndexedVariableSet&& indexedSet) {
            set(key, VariableContainer(std::move(indexedSet)));
        }

        /**
         * @brief Set an IndexedVariableSet by enum key (copy)
         * @param key Enum key
         * @param set IndexedVariableSet to copy
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, const IndexedVariableSet& indexedSet) {
            set(key, VariableContainer(indexedSet));
        }

        /**
         * @brief Set a scalar GRBVar by enum key
         * @param key Enum key
         * @param v GRBVar to store
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, const GRBVar& v) {
            set(key, VariableContainer(v));
        }

        /**
         * @brief Set a scalar GRBVar by enum key (move)
         * @param key Enum key
         * @param v GRBVar to store (moved)
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, GRBVar&& v) {
            set(key, VariableContainer(std::move(v)));
        }

        // ========================================================================
        // GETTERS
        // ========================================================================

        /**
         * @brief Get a container by enum key
         * @param key Enum key
         * @return Mutable reference to the VariableContainer
         * @throws std::out_of_range if key >= MAX
         */
        VariableContainer& get(EnumT key) {
            std::size_t idx = static_cast<std::size_t>(key);
            if (idx >= MAX) {
                throw std::out_of_range(
                    std::format("VariableTable::get: key {} >= {}", idx, MAX));
            }
            return table_[idx];
        }

        /// @brief Const version of get()
        const VariableContainer& get(EnumT key) const {
            std::size_t idx = static_cast<std::size_t>(key);
            if (idx >= MAX) {
                throw std::out_of_range(
                    std::format("VariableTable::get: key {} >= {}", idx, MAX));
            }
            return table_[idx];
        }

        /// @brief Alias for get() using function call syntax
        VariableContainer& operator()(EnumT key) { return get(key); }

        /// @brief Const alias for get() using function call syntax
        const VariableContainer& operator()(EnumT key) const { return get(key); }

        // ========================================================================
        // DIRECT VARIABLE ACCESS
        // ========================================================================

        /**
         * @brief Direct access to a variable within a container
         * @tparam I Integral index types
         * @param key Enum key for the container
         * @param idx Indices within the container (empty for scalar)
         * @return Mutable reference to the GRBVar
         * @throws std::out_of_range if key >= MAX or indices are invalid
         *
         * @example
         *     GRBVar& v = vt.var(Vars::X, 3, 5);  // Access X(3, 5)
         *     GRBVar& s = vt.var(Vars::Y);         // Access scalar Y
         */
        template<typename... I>
        GRBVar& var(EnumT key, I... idx) {
            static_assert((std::is_integral_v<I> && ...),
                "VariableTable::var: indices must be integral");

            std::size_t k = static_cast<std::size_t>(key);
            if (k >= MAX) {
                throw std::out_of_range(
                    std::format("VariableTable::var: key {} >= {}", k, MAX));
            }

            if constexpr (sizeof...(idx) == 0) {
                return table_[k].scalar();
            }
            else {
                return table_[k].at(idx...);
            }
        }

        /// @brief Const version of var()
        template<typename... I>
        const GRBVar& var(EnumT key, I... idx) const {
            static_assert((std::is_integral_v<I> && ...),
                "VariableTable::var: indices must be integral");

            std::size_t k = static_cast<std::size_t>(key);
            if (k >= MAX) {
                throw std::out_of_range(
                    std::format("VariableTable::var: key {} >= {}", k, MAX));
            }

            if constexpr (sizeof...(idx) == 0) {
                return table_[k].scalar();
            }
            else {
                return table_[k].at(idx...);
            }
        }

        // ========================================================================
        // MODE QUERIES
        // ========================================================================

        /**
         * @brief Check if entry at key is in dense mode
         * @param key Enum key
         * @return true if entry holds a VariableGroup
         * @throws std::out_of_range if key >= MAX
         */
        bool isDense(EnumT key) const {
            return get(key).isDense();
        }

        /**
         * @brief Check if entry at key is in sparse mode
         * @param key Enum key
         * @return true if entry holds an IndexedVariableSet
         * @throws std::out_of_range if key >= MAX
         */
        bool isSparse(EnumT key) const {
            return get(key).isSparse();
        }

        /**
         * @brief Check if entry at key is empty
         * @param key Enum key
         * @return true if entry is uninitialized
         * @throws std::out_of_range if key >= MAX
         */
        bool isEmpty(EnumT key) const {
            return get(key).isEmpty();
        }
    };


    // ============================================================================
    // SOLUTION EXTRACTION
    // ============================================================================
    /**
     * @defgroup SolutionExtraction Solution Extraction Utilities
     * @brief Functions for retrieving optimization results from variables
     *
     * @details After calling model.optimize(), these utilities extract solution
     *          values from individual variables or entire variable collections.
     *          All functions require the model to be in an optimized state with
     *          a valid solution available.
     *
     * @note These functions access GRB_DoubleAttr_X which requires:
     *       - model.optimize() has been called
     *       - Optimization status is GRB_OPTIMAL or has a feasible solution
     *       - model.update() is not required before reading solution values
     *
     * @example
     *     model.optimize();
     *     double x_val = dsl::value(x);           // Single variable
     *     auto X_vals = dsl::values(X);           // VariableGroup -> vector
     *     auto Y_vals = dsl::values(Y);           // IndexedVariableSet -> vector
     *
     * @{
     */

    /**
     * @brief Get the solution value of a single variable
     *
     * @param v The GRBVar to query
     * @return Solution value (GRB_DoubleAttr_X)
     *
     * @throws GRBException if model not optimized or no solution available
     * @complexity O(1)
     *
     * @example
     *     model.optimize();
     *     double val = dsl::value(x);
     */
    inline double value(const GRBVar& v) {
        return v.get(GRB_DoubleAttr_X);
    }

    /**
     * @brief Get all solution values from a VariableGroup
     *
     * @param vg The VariableGroup to query
     * @return Vector of solution values in iteration order (row-major for N-D)
     *
     * @throws GRBException if model not optimized or no solution available
     * @complexity O(n) where n = total number of variables
     *
     * @note Values are returned in the same order as forEach iteration.
     * @note For scalars, returns a single-element vector.
     *
     * @example
     *     auto X = VariableFactory::add(model, GRB_BINARY, 0, 1, "X", 3, 4);
     *     model.optimize();
     *     std::vector<double> vals = dsl::values(X);  // 12 values
     */
    inline std::vector<double> values(const VariableGroup& vg) {
        std::vector<double> result;
        vg.forEach([&](const GRBVar& v, const std::vector<int>&) {
            result.push_back(v.get(GRB_DoubleAttr_X));
        });
        return result;
    }

    /**
     * @brief Get all solution values from an IndexedVariableSet
     *
     * @param vs The IndexedVariableSet to query
     * @return Vector of solution values in storage order
     *
     * @throws GRBException if model not optimized or no solution available
     * @complexity O(n) where n = number of variables in set
     *
     * @note Values are returned in the same order as forEach iteration
     *       (i.e., the order elements were added to the domain).
     *
     * @example
     *     auto Y = VariableFactory::addIndexed(model, GRB_BINARY, 0, 1, "Y", I * J);
     *     model.optimize();
     *     std::vector<double> vals = dsl::values(Y);
     */
    inline std::vector<double> values(const IndexedVariableSet& vs) {
        std::vector<double> result;
        result.reserve(vs.size());
        vs.forEach([&](const GRBVar& v, const std::vector<int>&) {
            result.push_back(v.get(GRB_DoubleAttr_X));
        });
        return result;
    }

    /**
     * @brief Get solution values as index-value pairs from a VariableGroup
     *
     * @param vg The VariableGroup to query
     * @return Vector of pairs: (index vector, solution value)
     *
     * @throws GRBException if model not optimized or no solution available
     * @complexity O(n) where n = total number of variables
     *
     * @note Useful when you need both indices and values together.
     *
     * @example
     *     for (auto& [idx, val] : dsl::valuesWithIndex(X)) {
     *         std::cout << "X[" << idx[0] << "," << idx[1] << "] = " << val << "\n";
     *     }
     */
    inline std::vector<std::pair<std::vector<int>, double>> valuesWithIndex(const VariableGroup& vg) {
        std::vector<std::pair<std::vector<int>, double>> result;
        vg.forEach([&](const GRBVar& v, const std::vector<int>& idx) {
            result.emplace_back(idx, v.get(GRB_DoubleAttr_X));
        });
        return result;
    }

    /**
     * @brief Get solution values as index-value pairs from an IndexedVariableSet
     *
     * @param vs The IndexedVariableSet to query
     * @return Vector of pairs: (index vector, solution value)
     *
     * @throws GRBException if model not optimized or no solution available
     * @complexity O(n) where n = number of variables in set
     *
     * @example
     *     for (auto& [idx, val] : dsl::valuesWithIndex(Y)) {
     *         if (val > 0.5) {
     *             std::cout << "Selected: " << idx[0] << "," << idx[1] << "\n";
     *         }
     *     }
     */
    inline std::vector<std::pair<std::vector<int>, double>> valuesWithIndex(const IndexedVariableSet& vs) {
        std::vector<std::pair<std::vector<int>, double>> result;
        result.reserve(vs.size());
        vs.forEach([&](const GRBVar& v, const std::vector<int>& idx) {
            result.emplace_back(idx, v.get(GRB_DoubleAttr_X));
        });
        return result;
    }

    /**
     * @brief Get all solution values from a VariableContainer
     *
     * @param vc The VariableContainer to query
     * @return Vector of solution values in iteration order
     *
     * @throws GRBException if model not optimized or no solution available
     * @throws std::runtime_error if container is empty
     * @complexity O(n) where n = total number of variables
     *
     * @example
     *     std::vector<double> vals = dsl::values(vc);
     */
    inline std::vector<double> values(const VariableContainer& vc) {
        std::vector<double> result;
        vc.forEach([&](const GRBVar& v, const std::vector<int>&) {
            result.push_back(v.get(GRB_DoubleAttr_X));
        });
        return result;
    }

    /**
     * @brief Get solution values as index-value pairs from a VariableContainer
     *
     * @param vc The VariableContainer to query
     * @return Vector of pairs: (index vector, solution value)
     *
     * @throws GRBException if model not optimized or no solution available
     * @throws std::runtime_error if container is empty
     * @complexity O(n) where n = total number of variables
     */
    inline std::vector<std::pair<std::vector<int>, double>> valuesWithIndex(const VariableContainer& vc) {
        std::vector<std::pair<std::vector<int>, double>> result;
        vc.forEach([&](const GRBVar& v, const std::vector<int>& idx) {
            result.emplace_back(idx, v.get(GRB_DoubleAttr_X));
        });
        return result;
    }

    /**
     * @brief Get solution value at specific indices from a VariableContainer
     *
     * @tparam Indices Integral index types
     * @param vc The VariableContainer to query
     * @param idx Index values
     * @return Solution value at the specified position
     *
     * @throws std::runtime_error if container is empty or index invalid
     * @throws GRBException if model not optimized or no solution available
     */
    template<typename... Indices>
    inline double valueAt(const VariableContainer& vc, Indices... idx) {
        return value(vc.at(idx...));
    }

    /**
     * @brief Get solution value at specific indices from a VariableGroup
     *
     * @tparam Indices Integral index types
     * @param vg The VariableGroup to query
     * @param idx Index values for each dimension
     * @return Solution value at the specified position
     *
     * @throws std::runtime_error if number of indices != dimension()
     * @throws std::out_of_range if any index is out of bounds
     * @throws GRBException if model not optimized or no solution available
     * @complexity O(dims) for index lookup + O(1) for value retrieval
     *
     * @note Mirrors the syntax of vg.at(i, j) but returns the solution value.
     *
     * @example
     *     model.optimize();
     *     double v = dsl::valueAt(X, 1, 2);  // Get X(1,2) solution value
     */
    template<typename... Indices>
    inline double valueAt(const VariableGroup& vg, Indices... idx) {
        return value(vg.at(idx...));
    }

    /**
     * @brief Get solution value at specific indices from an IndexedVariableSet
     *
     * @tparam Indices Integral index types
     * @param vs The IndexedVariableSet to query
     * @param idx Index values
     * @return Solution value at the specified position
     *
     * @throws std::out_of_range if index not found in domain
     * @throws GRBException if model not optimized or no solution available
     * @complexity O(1) average for hash lookup + O(1) for value retrieval
     *
     * @note Mirrors the syntax of vs.at(i, j) but returns the solution value.
     *
     * @example
     *     model.optimize();
     *     double v = dsl::valueAt(Y, 0, 1);  // Get Y(0,1) solution value
     */
    template<typename... Indices>
    inline double valueAt(const IndexedVariableSet& vs, Indices... idx) {
        return value(vs.at(idx...));
    }

    /** @} */ // end of SolutionExtraction group


    // ============================================================================
    // VARIABLE MODIFICATION
    // ============================================================================
    /**
     * @defgroup VariableModification Variable Modification Utilities
     * @brief Functions for modifying variable bounds and providing warm starts
     *
     * @details These utilities simplify common variable modification patterns:
     *          - Fixing variables to specific values (for what-if analysis)
     *          - Restoring original bounds after fixing
     *          - Setting start values for MIP warm starts
     *
     * @note Modifications take effect after model.update() or at next optimize().
     * @note Fixing a binary/integer variable to a fractional value may cause issues.
     *
     * @example
     *     dsl::fix(x, 1.0);                    // Fix single variable
     *     dsl::fixAll(X, solution);            // Fix entire group
     *     dsl::unfix(x, 0.0, 1.0);             // Restore bounds
     *     dsl::setStart(X, warmStartValues);   // Provide MIP start
     *
     * @{
     */

    /**
     * @brief Fix a variable to a specific value
     *
     * @param v Variable to fix
     * @param val Value to fix to (sets LB = UB = val)
     *
     * @throws GRBException on Gurobi API error
     * @complexity O(1)
     *
     * @note Equivalent to setting both LB and UB to the same value.
     * @note Use unfix() to restore original bounds.
     *
     * @example
     *     dsl::fix(x, 1.0);      // Force x = 1
     *     model.optimize();      // Solve with x fixed
     *     dsl::unfix(x, 0, 1);   // Restore x to [0, 1]
     */
    inline void fix(GRBVar& v, double val) {
        v.set(GRB_DoubleAttr_LB, val);
        v.set(GRB_DoubleAttr_UB, val);
    }

    /**
     * @brief Restore variable bounds after fixing
     *
     * @param v Variable to unfix
     * @param lb New lower bound
     * @param ub New upper bound
     *
     * @throws GRBException on Gurobi API error
     * @throws std::invalid_argument if lb > ub
     * @complexity O(1)
     *
     * @example
     *     dsl::fix(x, 1.0);
     *     model.optimize();
     *     dsl::unfix(x, 0.0, 1.0);  // Restore to binary bounds
     */
    inline void unfix(GRBVar& v, double lb, double ub) {
        if (lb > ub) {
            throw std::invalid_argument(
                std::format("unfix: lb ({}) > ub ({})", lb, ub));
        }
        v.set(GRB_DoubleAttr_LB, lb);
        v.set(GRB_DoubleAttr_UB, ub);
    }

    /**
     * @brief Set the MIP start value for a variable
     *
     * @param v Variable to set start value for
     * @param val Start value (GRB_DoubleAttr_Start)
     *
     * @throws GRBException on Gurobi API error
     * @complexity O(1)
     *
     * @note Start values provide a warm start hint to the MIP solver.
     * @note Gurobi will attempt to construct a feasible solution from starts.
     * @note Not all start values need to be set; partial solutions are allowed.
     *
     * @example
     *     dsl::setStart(x, 1.0);   // Hint that x should be 1
     *     model.optimize();        // Solver uses hint if feasible
     */
    inline void setStart(GRBVar& v, double val) {
        v.set(GRB_DoubleAttr_Start, val);
    }

    /**
     * @brief Fix all variables in a VariableGroup to specified values
     *
     * @param vg VariableGroup to fix
     * @param vals Values to fix to (must match iteration order and count)
     *
     * @throws std::invalid_argument if vals.size() doesn't match variable count
     * @throws GRBException on Gurobi API error
     * @complexity O(n) where n = number of variables
     *
     * @note Values must be in the same order as forEach iteration (row-major).
     * @note Use unfixAll() to restore original bounds.
     *
     * @example
     *     std::vector<double> solution = dsl::values(X);  // Save solution
     *     // ... modify model ...
     *     dsl::fixAll(X, solution);   // Fix to previous solution
     *     model.optimize();
     */
    inline void fixAll(VariableGroup& vg, const std::vector<double>& vals) {
        if (vals.size() != vg.count()) {
            throw std::invalid_argument(
                std::format("fixAll: expected {} values, got {}", vg.count(), vals.size()));
        }

        std::size_t i = 0;
        vg.forEach([&](GRBVar& v, const std::vector<int>&) {
            fix(v, vals[i++]);
        });
    }

    /**
     * @brief Fix all variables in an IndexedVariableSet to specified values
     *
     * @param vs IndexedVariableSet to fix
     * @param vals Values to fix to (must match storage order and count)
     *
     * @throws std::invalid_argument if vals.size() doesn't match variable count
     * @throws GRBException on Gurobi API error
     * @complexity O(n) where n = number of variables
     *
     * @example
     *     std::vector<double> solution = dsl::values(Y);
     *     dsl::fixAll(Y, solution);
     */
    inline void fixAll(IndexedVariableSet& vs, const std::vector<double>& vals) {
        if (vals.size() != vs.size()) {
            throw std::invalid_argument(
                std::format("fixAll: expected {} values, got {}", vs.size(), vals.size()));
        }

        std::size_t i = 0;
        vs.forEach([&](GRBVar& v, const std::vector<int>&) {
            fix(v, vals[i++]);
        });
    }

    /**
     * @brief Set MIP start values for all variables in a VariableGroup
     *
     * @param vg VariableGroup to set starts for
     * @param vals Start values (must match iteration order and count)
     *
     * @throws std::invalid_argument if vals.size() doesn't match variable count
     * @throws GRBException on Gurobi API error
     * @complexity O(n) where n = number of variables
     *
     * @note Values must be in the same order as forEach iteration.
     * @note Useful for providing warm starts from previous solutions.
     *
     * @example
     *     std::vector<double> prevSolution = dsl::values(X);
     *     // ... modify model ...
     *     dsl::setStartAll(X, prevSolution);  // Warm start
     *     model.optimize();
     */
    inline void setStartAll(VariableGroup& vg, const std::vector<double>& vals) {
        if (vals.size() != vg.count()) {
            throw std::invalid_argument(
                std::format("setStartAll: expected {} values, got {}", vg.count(), vals.size()));
        }

        std::size_t i = 0;
        vg.forEach([&](GRBVar& v, const std::vector<int>&) {
            setStart(v, vals[i++]);
        });
    }

    /**
     * @brief Set MIP start values for all variables in an IndexedVariableSet
     *
     * @param vs IndexedVariableSet to set starts for
     * @param vals Start values (must match storage order and count)
     *
     * @throws std::invalid_argument if vals.size() doesn't match variable count
     * @throws GRBException on Gurobi API error
     * @complexity O(n) where n = number of variables
     *
     * @example
     *     std::vector<double> prevSolution = dsl::values(Y);
     *     dsl::setStartAll(Y, prevSolution);
     *     model.optimize();
     */
    inline void setStartAll(IndexedVariableSet& vs, const std::vector<double>& vals) {
        if (vals.size() != vs.size()) {
            throw std::invalid_argument(
                std::format("setStartAll: expected {} values, got {}", vs.size(), vals.size()));
        }

        std::size_t i = 0;
        vs.forEach([&](GRBVar& v, const std::vector<int>&) {
            setStart(v, vals[i++]);
        });
    }

    /**
     * @brief Fix all variables in a VariableContainer to specified values
     *
     * @param vc VariableContainer to fix
     * @param vals Values to fix to (must match iteration order and count)
     *
     * @throws std::invalid_argument if vals.size() doesn't match variable count
     * @throws std::runtime_error if container is empty
     * @throws GRBException on Gurobi API error
     * @complexity O(n) where n = number of variables
     */
    inline void fixAll(VariableContainer& vc, const std::vector<double>& vals) {
        if (vals.size() != vc.count()) {
            throw std::invalid_argument(
                std::format("fixAll: expected {} values, got {}", vc.count(), vals.size()));
        }

        std::size_t i = 0;
        vc.forEach([&](GRBVar& v, const std::vector<int>&) {
            fix(v, vals[i++]);
        });
    }

    /**
     * @brief Set MIP start values for all variables in a VariableContainer
     *
     * @param vc VariableContainer to set starts for
     * @param vals Start values (must match iteration order and count)
     *
     * @throws std::invalid_argument if vals.size() doesn't match variable count
     * @throws std::runtime_error if container is empty
     * @throws GRBException on Gurobi API error
     * @complexity O(n) where n = number of variables
     */
    inline void setStartAll(VariableContainer& vc, const std::vector<double>& vals) {
        if (vals.size() != vc.count()) {
            throw std::invalid_argument(
                std::format("setStartAll: expected {} values, got {}", vc.count(), vals.size()));
        }

        std::size_t i = 0;
        vc.forEach([&](GRBVar& v, const std::vector<int>&) {
            setStart(v, vals[i++]);
        });
    }

    /**
     * @brief Clear MIP start value for a variable
     *
     * @param v Variable to clear start for
     *
     * @throws GRBException on Gurobi API error
     * @complexity O(1)
     *
     * @note Sets start to GRB_UNDEFINED to clear any previous hint.
     */
    inline void clearStart(GRBVar& v) {
        v.set(GRB_DoubleAttr_Start, GRB_UNDEFINED);
    }

    /**
     * @brief Get the current lower bound of a variable
     *
     * @param v Variable to query
     * @return Current lower bound (GRB_DoubleAttr_LB)
     *
     * @throws GRBException on Gurobi API error
     * @complexity O(1)
     */
    inline double lb(const GRBVar& v) {
        return v.get(GRB_DoubleAttr_LB);
    }

    /**
     * @brief Get the current upper bound of a variable
     *
     * @param v Variable to query
     * @return Current upper bound (GRB_DoubleAttr_UB)
     *
     * @throws GRBException on Gurobi API error
     * @complexity O(1)
     */
    inline double ub(const GRBVar& v) {
        return v.get(GRB_DoubleAttr_UB);
    }

    /**
     * @brief Set the lower bound of a variable
     *
     * @param v Variable to modify
     * @param bound New lower bound
     *
     * @throws GRBException on Gurobi API error
     * @complexity O(1)
     */
    inline void setLB(GRBVar& v, double bound) {
        v.set(GRB_DoubleAttr_LB, bound);
    }

    /**
     * @brief Set the upper bound of a variable
     *
     * @param v Variable to modify
     * @param bound New upper bound
     *
     * @throws GRBException on Gurobi API error
     * @complexity O(1)
     */
    inline void setUB(GRBVar& v, double bound) {
        v.set(GRB_DoubleAttr_UB, bound);
    }

    /** @} */ // end of VariableModification group

} // namespace dsl
