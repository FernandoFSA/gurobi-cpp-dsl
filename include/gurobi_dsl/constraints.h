#pragma once
/*
===============================================================================
CONSTRAINT MANAGEMENT SYSTEM - Gurobi C++ DSL
===============================================================================

OVERVIEW
--------
Implements the DSL's unified system for creating, storing, and organizing
constraints in Gurobi models. Mirrors the variable system and provides
structured access, iteration, and attribute queries for constraint collections.

KEY COMPONENTS
--------------
- ConstraintGroup        - Dense N-dimensional container of GRBConstr
- IndexedConstraintSet   - Constraints indexed by arbitrary domains (sparse)
- ConstraintContainer    - Unified wrapper for dense/sparse (mirrors VariableContainer)
- ConstraintFactory      - Unified backend for constraint creation
- ConstraintTable        - Enum-keyed registry of constraint collections

DESIGN PHILOSOPHY
-----------------
- Strong index checking with clear error messages
- Zero runtime overhead in release builds (naming disabled)
- No sparsity structure assumed for ConstraintGroup (dense only)
- IndexedConstraintSet handles arbitrary domains, including filtered sets
- API symmetry with VariableGroup and expression helpers

USAGE EXAMPLES
--------------
- Rectangular constraints (add)
    // Generator signature: GRBTempConstr(const std::vector<int>& idx)
    auto cap = ConstraintFactory::add(model, "cap",
        [&](const std::vector<int>& idx) {
            int i = idx[0];
            return x(i) <= capacity[i];  // returns GRBTempConstr
        }, 10);  // creates cap[0]..cap[9]
    GRBConstr& c = cap.at(3);  // access cap[3]

- Domain-based constraints (addIndexed)
    // Generator signature: GRBTempConstr(int i, int j, ...) - one arg per dimension
    dsl::IndexList I{1, 2, 3};
    dsl::IndexList J{4, 5};
    auto flow = ConstraintFactory::addIndexed(model, "flow", I * J,
        [&](int i, int j) {
            return x(i, j) + y(i, j) <= demand[i][j];  // returns GRBTempConstr
        });
    GRBConstr& f = flow.at(2, 5);  // access flow[2,5]

- Filtered domain constraints
    auto upper = ConstraintFactory::addIndexed(model, "upper",
        (I * J) | dsl::filter([](int i, int j) { return i < j; }),
        [&](int i, int j) { return x(i, j) <= bound; });

- Constraint table
    ConstraintTable<CSets> tbl;
    tbl.set(CSets::Cap, std::move(capGroup));
    GRBConstr& c = tbl.constr(CSets::Cap, i);

DEPENDENCIES
------------
- <string>, <vector>, <array>, <tuple> - Container types
- <stdexcept>, <format> - Error handling
- <unordered_map>, <sstream> - Hash-based lookup
- <type_traits>, <concepts> - Compile-time type safety
- "gurobi_c++.h" - Gurobi C++ API
- "naming.h" - Debug-aware naming utilities
- "enum_utils.h" - Enum reflection helpers

PERFORMANCE NOTES
-----------------
- ConstraintGroup: O(1) random access via at()/operator()
- IndexedConstraintSet: O(1) lookup via hash map on index tuples
- Naming: O(1) when naming_disabled() (returns empty string)
- ConstraintFactory: Linear in domain size for constraint creation

THREAD SAFETY
-------------
- All types are value types; concurrent const access is safe
- Mutations require external synchronization
- GRBModel operations follow Gurobi's threading model

EXCEPTION SAFETY
----------------
- Strong exception safety for constraint creation
- Index validation throws std::out_of_range for invalid indices
- Negative size throws std::invalid_argument in ConstraintFactory

===============================================================================
*/

#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <stdexcept>
#include <type_traits>
#include <concepts>
#include <format>
#include <unordered_map>
#include <sstream>
#include <tuple>
#include <variant>

#include "gurobi_c++.h"
#include "naming.h"
#include "enum_utils.h"

namespace dsl {

    class ConstraintFactory; // forward declaration

    // ============================================================================
    // CONSTRAINT GROUP
    // ============================================================================
    /**
     * @class ConstraintGroup
     * @brief Dense N-dimensional container of GRBConstr
     * @details
     *  Represents either:
     *  - Scalar constraint         (dims == 0)
     *  - 1D array of constraints   (dims == 1)
     *  - 2D matrix of constraints  (dims == 2)
     *  - N-dimensional arrays      (dims == N)
     *
     *  Internal representation uses a tree of Node objects where:
     *  - Leaf nodes hold a single GRBConstr
     *  - Container nodes have children.size() sub-nodes
     *
     * @example
     *   ConstraintGroup cap = ConstraintFactory::add(model, "cap", gen, 10);
     *   GRBConstr& c = cap.at(3);  // access cap[3]
     *   cap.forEach([](GRBConstr& c, const auto& idx) { // iterate // });
     */
    class ConstraintGroup {
    public:
        struct Node {
            GRBConstr scalar;
            std::vector<Node> children;

            Node() = default;
            explicit Node(const GRBConstr& c) : scalar(c) {}
            explicit Node(GRBConstr&& c) : scalar(std::move(c)) {}
            explicit Node(size_t n) : children(n) {}
        };

    private:
        Node root;
        int  dims = 0;

    public:
        // ---------------------------------------------------------------------
        // Constructors
        // ---------------------------------------------------------------------
        ConstraintGroup() = default;

        explicit ConstraintGroup(const GRBConstr& c)
            : root(Node(c)), dims(0) {
        }

        explicit ConstraintGroup(GRBConstr&& c)
            : root(Node(std::move(c))), dims(0) {
        }

        ConstraintGroup(Node&& r, int d)
            : root(std::move(r)), dims(d) {
        }

        // ---------------------------------------------------------------------
        // Introspection
        // ---------------------------------------------------------------------

        /// @brief Returns the number of dimensions
        /// @noexcept
        [[nodiscard]] int  dimension() const noexcept { return dims; }

        /// @brief Returns true if this is a scalar (0-D) constraint
        /// @noexcept
        [[nodiscard]] bool isScalar() const noexcept { return dims == 0; }

        /// @brief Returns true if this is a multi-dimensional constraint group
        /// @noexcept
        [[nodiscard]] bool isMultiDim() const noexcept { return dims > 0; }

        /// @brief Returns the shape as a vector of dimension sizes
        /// @complexity O(dims)
        [[nodiscard]] std::vector<size_t> shape() const {
            std::vector<size_t> shp;
            shp.reserve(dims);
            const Node* n = &root;

            for (int d = 0; d < dims; ++d) {
                if (n->children.empty()) break;
                shp.push_back(n->children.size());
                n = &n->children[0];
            }
            return shp;
        }

        /// @brief Returns the size of a specific dimension
        /// @param dim Dimension index (0-based)
        /// @throws std::out_of_range if dim is out of range
        [[nodiscard]] size_t size(int dim) const {
            if (dim < 0 || dim >= dims) {
                throw std::out_of_range(
                    std::format("ConstraintGroup::size: dimension {} out of range [0, {})",
                        dim, dims));
            }
            const Node* n = &root;
            for (int d = 0; d < dim; ++d) {
                n = &n->children[0];
            }
            return n->children.size();
        }

        // ---------------------------------------------------------------------
        // Access
        // ---------------------------------------------------------------------
        template<typename... Indices>
        GRBConstr& at(Indices... idx) {
            static_assert((std::is_integral_v<Indices> && ...),
                "ConstraintGroup::at: indices must be integral");

            constexpr size_t num = sizeof...(idx);
            if (static_cast<int>(num) != dims) {
                throw std::runtime_error(
                    std::format("ConstraintGroup::at: expected {} indices, got {}",
                        dims, num));
            }
            return atRec(root, idx...);
        }

        template<typename... Indices>
        const GRBConstr& at(Indices... idx) const {
            return const_cast<ConstraintGroup*>(this)->at(idx...);
        }

        template<typename... Indices>
        GRBConstr& operator()(Indices... idx) { return at(idx...); }

        template<typename... Indices>
        const GRBConstr& operator()(Indices... idx) const { return at(idx...); }

        // 0-D scalar access
        GRBConstr& scalar() {
            if (dims != 0) {
                throw std::runtime_error(
                    std::format("ConstraintGroup::scalar: group is {}-dimensional", dims));
            }
            return root.scalar;
        }

        const GRBConstr& scalar() const {
            if (dims != 0) {
                throw std::runtime_error(
                    std::format("ConstraintGroup::scalar: group is {}-dimensional", dims));
            }
            return root.scalar;
        }

        GRBConstr& raw() { return scalar(); }
        const GRBConstr& raw() const { return scalar(); }

        // ---------------------------------------------------------------------
        // Convenience attribute access (scalar only)
        // ---------------------------------------------------------------------
        char sense() const { return scalar().get(GRB_CharAttr_Sense); }

        double rhs() const { return scalar().get(GRB_DoubleAttr_RHS); }

        std::string name() const {
            return scalar().get(GRB_StringAttr_ConstrName);
        }

        void setName(const std::string& n) {
            scalar().set(GRB_StringAttr_ConstrName, n);
        }

        double slack() const {
            return scalar().get(GRB_DoubleAttr_Slack);
        }

        double dual() const {
            return scalar().get(GRB_DoubleAttr_Pi);
        }

        // ---------------------------------------------------------------------
        // Iteration
        // ---------------------------------------------------------------------
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
        // leaf
        GRBConstr& atRec(Node& n) {
            return n.scalar;
        }

        // recursive
        template<typename First, typename... Rest>
        GRBConstr& atRec(Node& n, First i, Rest... rest) {
            if (i < 0) {
                throw std::out_of_range(
                    std::format("ConstraintGroup::atRec: negative index {}", i));
            }
            size_t idx = static_cast<size_t>(i);
            if (idx >= n.children.size()) {
                throw std::out_of_range(
                    std::format("ConstraintGroup::atRec: index {} out of range [0, {})",
                        i, n.children.size()));
            }
            return atRec(n.children[idx], rest...);
        }

        template<typename Fn>
        static void forEachRec(Node& n,
            std::vector<int>& idx,
            Fn& fn) {
            if (n.children.empty()) {
                fn(n.scalar, idx);
                return;
            }

            for (size_t i = 0; i < n.children.size(); ++i) {
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

            for (size_t i = 0; i < n.children.size(); ++i) {
                idx.push_back(static_cast<int>(i));
                forEachRecConst(n.children[i], idx, fn);
                idx.pop_back();
            }
        }

        friend class ConstraintFactory;
    };


    // ============================================================================
    // INDEXED CONSTRAINT SET
    // ============================================================================
    /**
     * @class IndexedConstraintSet
     * @brief Constraints indexed by arbitrary index domains
     * @details
     *  Stores constraints created from general index domains such as:
     *  - I * J (Cartesian products)
     *  - I * J | filter(...) (filtered domains)
     *  - Irregular or sparse domains
     *
     *  Not restricted to rectangular shapes. Each stored constraint has an
     *  explicit index vector. Provides O(1) lookup via hash map.
     *
     * @example
     *   auto flow = ConstraintFactory::addIndexed(model, "flow", I * J, gen);
     *   GRBConstr& f = flow.at(2, 5);     // throws if not found
     *   GRBConstr* p = flow.try_get(2, 5); // nullptr if not found
     *   flow.forEach([](GRBConstr& c, const auto& idx) { // iterate // });
     */
    class IndexedConstraintSet {
    public:
        struct Entry {
            GRBConstr constr;
            std::vector<int> index;
        };

    private:
        std::vector<Entry> entries;
        std::unordered_map<std::string, std::size_t> indexMap;

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

        template<typename... I>
        static std::string makeKey(I... idx) {
            static_assert((std::is_integral_v<I> && ...),
                "IndexedConstraintSet::makeKey: indices must be integral");
            std::ostringstream oss;
            bool first = true;
            ((oss << (first ? (first = false, "") : "_")
                << static_cast<long long>(idx)), ...);
            return oss.str();
        }

        void addEntry(GRBConstr&& c, std::vector<int>&& idx) {
            std::size_t pos = entries.size();
            std::string key = makeKeyFromVector(idx);
            indexMap.emplace(std::move(key), pos);
            entries.push_back(Entry{ std::move(c), std::move(idx) });
        }

    public:
        /// @brief Default constructor (empty set)
        IndexedConstraintSet() = default;

        /// @brief Returns the number of constraints in the set
        /// @noexcept
        [[nodiscard]] std::size_t size() const noexcept { return entries.size(); }

        /// @brief Returns true if the set is empty
        /// @noexcept
        [[nodiscard]] bool empty() const noexcept { return entries.empty(); }

        using iterator = std::vector<Entry>::iterator;
        using const_iterator = std::vector<Entry>::const_iterator;

        iterator begin() noexcept { return entries.begin(); }
        iterator end() noexcept { return entries.end(); }

        const_iterator begin() const noexcept { return entries.begin(); }
        const_iterator end() const noexcept { return entries.end(); }

        /// @brief Returns read-only access to all entries
        /// @noexcept
        [[nodiscard]] const std::vector<Entry>& all() const noexcept { return entries; }

        template<typename... I>
        GRBConstr& at(I... idx) {
            std::string key = makeKey(idx...);
            auto it = indexMap.find(key);
            if (it == indexMap.end()) {
                throw std::out_of_range(
                    std::format("IndexedConstraintSet::at: index [{}] not found", key));
            }
            return entries[it->second].constr;
        }

        template<typename... I>
        const GRBConstr& at(I... idx) const {
            return const_cast<IndexedConstraintSet*>(this)->at(idx...);
        }

        template<typename... I>
        GRBConstr& operator()(I... idx) { return at(idx...); }

        template<typename... I>
        const GRBConstr& operator()(I... idx) const { return at(idx...); }

        template<typename... I>
        GRBConstr* try_get(I... idx) noexcept {
            try {
                std::string key = makeKey(idx...);
                auto it = indexMap.find(key);
                if (it == indexMap.end()) {
                    return nullptr;
                }
                return &entries[it->second].constr;
            }
            catch (...) {
                return nullptr;
            }
        }

        template<typename... I>
        const GRBConstr* try_get(I... idx) const noexcept {
            return const_cast<IndexedConstraintSet*>(this)->try_get(idx...);
        }

        template<typename Fn>
        void forEach(Fn&& fn) {
            for (auto& e : entries) {
                fn(e.constr, e.index);
            }
        }

        /// @brief Const version of forEach()
        template<typename Fn>
        void forEach(Fn&& fn) const {
            for (const auto& e : entries) {
                fn(e.constr, e.index);
            }
        }

    private:
        friend class ConstraintFactory;
    };


    // ============================================================================
    // CONSTRAINT CONTAINER (UNIFIED DENSE/SPARSE)
    // ============================================================================
    /**
     * @class ConstraintContainer
     * @brief Unified container that can hold either dense (ConstraintGroup) or sparse (IndexedConstraintSet) constraints
     *
     * @details Provides a single type that can operate in two modes:
     *          - Dense mode: Wraps a ConstraintGroup for rectangular constraint arrays
     *          - Sparse mode: Wraps an IndexedConstraintSet for domain-based constraints
     *
     *          This enables ConstraintTable to hold mixed collections without requiring
     *          separate tables for dense and sparse constraints.
     *
     * @note Mode is determined at construction and cannot be changed afterward.
     * @note Mirrors VariableContainer for API symmetry.
     *
     * @example
     *     ConstraintContainer dense(ConstraintFactory::add(model, "cap", gen, 10));
     *     ConstraintContainer sparse(ConstraintFactory::addIndexed(model, "flow", I * J, gen));
     *
     *     // Unified access works for both
     *     dense.at(3);     // Access dense constraint
     *     sparse.at(1, 2); // Access sparse constraint
     *
     *     // Mode-specific access
     *     if (dense.isDense()) {
     *         auto& group = dense.asGroup();
     *         std::cout << "Dims: " << group.dimension() << "\n";
     *     }
     *
     * @see ConstraintGroup
     * @see IndexedConstraintSet
     * @see ConstraintTable
     */
    class ConstraintContainer {
    public:
        /// @brief Storage mode enumeration
        enum class Mode { Empty, Dense, Sparse };

    private:
        std::variant<std::monostate, ConstraintGroup, IndexedConstraintSet> storage_;

    public:
        // ========================================================================
        // CONSTRUCTORS
        // ========================================================================

        /// @brief Default constructor; creates an empty container
        ConstraintContainer() = default;

        /**
         * @brief Construct from a ConstraintGroup (dense mode)
         * @param group Dense constraint collection to store
         * @post mode() == Mode::Dense
         */
        ConstraintContainer(ConstraintGroup&& group)
            : storage_(std::move(group)) {}

        /**
         * @brief Construct from a ConstraintGroup (copy)
         * @param group Dense constraint collection to copy
         * @post mode() == Mode::Dense
         */
        ConstraintContainer(const ConstraintGroup& group)
            : storage_(group) {}

        /**
         * @brief Construct from an IndexedConstraintSet (sparse mode)
         * @param set Sparse constraint collection to store
         * @post mode() == Mode::Sparse
         */
        ConstraintContainer(IndexedConstraintSet&& set)
            : storage_(std::move(set)) {}

        /**
         * @brief Construct from an IndexedConstraintSet (copy)
         * @param set Sparse constraint collection to copy
         * @post mode() == Mode::Sparse
         */
        ConstraintContainer(const IndexedConstraintSet& set)
            : storage_(set) {}

        /**
         * @brief Construct scalar from GRBConstr (dense mode, 0-D)
         * @param c Scalar constraint to store
         * @post mode() == Mode::Dense, asGroup().isScalar() == true
         */
        ConstraintContainer(const GRBConstr& c)
            : storage_(ConstraintGroup(c)) {}

        /**
         * @brief Construct scalar from GRBConstr (move)
         * @param c Scalar constraint to move
         * @post mode() == Mode::Dense, asGroup().isScalar() == true
         */
        ConstraintContainer(GRBConstr&& c)
            : storage_(ConstraintGroup(std::move(c))) {}

        // ========================================================================
        // MODE INTROSPECTION
        // ========================================================================

        /// @brief Returns the current storage mode
        /// @noexcept
        [[nodiscard]] Mode mode() const noexcept {
            if (std::holds_alternative<std::monostate>(storage_)) return Mode::Empty;
            if (std::holds_alternative<ConstraintGroup>(storage_)) return Mode::Dense;
            return Mode::Sparse;
        }

        /// @brief Returns true if container is empty
        /// @noexcept
        [[nodiscard]] bool isEmpty() const noexcept { return mode() == Mode::Empty; }

        /// @brief Returns true if container holds a ConstraintGroup (dense)
        /// @noexcept
        [[nodiscard]] bool isDense() const noexcept { return mode() == Mode::Dense; }

        /// @brief Returns true if container holds an IndexedConstraintSet (sparse)
        /// @noexcept
        [[nodiscard]] bool isSparse() const noexcept { return mode() == Mode::Sparse; }

        // ========================================================================
        // MODE-SPECIFIC ACCESS
        // ========================================================================

        /**
         * @brief Get the underlying ConstraintGroup (dense mode only)
         * @return Mutable reference to the ConstraintGroup
         * @throws std::runtime_error if not in dense mode
         */
        ConstraintGroup& asGroup() {
            if (!isDense()) {
                throw std::runtime_error("ConstraintContainer::asGroup: not in dense mode");
            }
            return std::get<ConstraintGroup>(storage_);
        }

        /// @brief Const version of asGroup()
        const ConstraintGroup& asGroup() const {
            if (!isDense()) {
                throw std::runtime_error("ConstraintContainer::asGroup: not in dense mode");
            }
            return std::get<ConstraintGroup>(storage_);
        }

        /**
         * @brief Get the underlying IndexedConstraintSet (sparse mode only)
         * @return Mutable reference to the IndexedConstraintSet
         * @throws std::runtime_error if not in sparse mode
         */
        IndexedConstraintSet& asIndexed() {
            if (!isSparse()) {
                throw std::runtime_error("ConstraintContainer::asIndexed: not in sparse mode");
            }
            return std::get<IndexedConstraintSet>(storage_);
        }

        /// @brief Const version of asIndexed()
        const IndexedConstraintSet& asIndexed() const {
            if (!isSparse()) {
                throw std::runtime_error("ConstraintContainer::asIndexed: not in sparse mode");
            }
            return std::get<IndexedConstraintSet>(storage_);
        }

        // ========================================================================
        // UNIFIED ACCESS
        // ========================================================================

        /**
         * @brief Access constraint by variadic indices (works for both modes)
         * @tparam I Integral index types
         * @param idx Index values
         * @return Mutable reference to the GRBConstr
         * @throws std::runtime_error if container is empty or index invalid
         */
        template<typename... I>
        GRBConstr& at(I... idx) {
            switch (mode()) {
                case Mode::Dense:
                    return std::get<ConstraintGroup>(storage_).at(idx...);
                case Mode::Sparse:
                    return std::get<IndexedConstraintSet>(storage_).at(idx...);
                default:
                    throw std::runtime_error("ConstraintContainer::at: container is empty");
            }
        }

        /// @brief Const version of at()
        template<typename... I>
        const GRBConstr& at(I... idx) const {
            return const_cast<ConstraintContainer*>(this)->at(idx...);
        }

        /// @brief Alias for at() using function call syntax
        template<typename... I>
        GRBConstr& operator()(I... idx) { return at(idx...); }

        /// @brief Const alias for at() using function call syntax
        template<typename... I>
        const GRBConstr& operator()(I... idx) const { return at(idx...); }

        /**
         * @brief Try to access constraint by variadic indices
         * @tparam I Integral index types
         * @param idx Index values
         * @return Pointer to GRBConstr, or nullptr if not found/empty
         * @noexcept
         */
        template<typename... I>
        GRBConstr* try_get(I... idx) noexcept {
            try {
                switch (mode()) {
                    case Mode::Dense:
                        return &std::get<ConstraintGroup>(storage_).at(idx...);
                    case Mode::Sparse:
                        return std::get<IndexedConstraintSet>(storage_).try_get(idx...);
                    default:
                        return nullptr;
                }
            } catch (...) {
                return nullptr;
            }
        }

        /// @brief Const version of try_get()
        template<typename... I>
        const GRBConstr* try_get(I... idx) const noexcept {
            return const_cast<ConstraintContainer*>(this)->try_get(idx...);
        }

        /**
         * @brief Access the scalar constraint (dense mode, 0-D only)
         * @return Mutable reference to the stored GRBConstr
         * @throws std::runtime_error if not dense or not scalar
         */
        GRBConstr& scalar() {
            if (!isDense()) {
                throw std::runtime_error("ConstraintContainer::scalar: not in dense mode");
            }
            return std::get<ConstraintGroup>(storage_).scalar();
        }

        /// @brief Const version of scalar()
        const GRBConstr& scalar() const {
            if (!isDense()) {
                throw std::runtime_error("ConstraintContainer::scalar: not in dense mode");
            }
            return std::get<ConstraintGroup>(storage_).scalar();
        }

        // ========================================================================
        // UNIFIED INTROSPECTION
        // ========================================================================

        /**
         * @brief Returns true if this is a scalar (dense mode, 0-D)
         * @return true if dense and 0-dimensional, false otherwise
         * @noexcept
         */
        [[nodiscard]] bool isScalar() const noexcept {
            if (!isDense()) return false;
            return std::get<ConstraintGroup>(storage_).isScalar();
        }

        // ========================================================================
        // UNIFIED ITERATION
        // ========================================================================

        /**
         * @brief Iterate over all constraints with their indices
         * @tparam Fn Callable with signature (GRBConstr&, const std::vector<int>&)
         * @param fn Function to call for each constraint
         * @throws std::runtime_error if container is empty
         */
        template<typename Fn>
        void forEach(Fn&& fn) {
            switch (mode()) {
                case Mode::Dense:
                    std::get<ConstraintGroup>(storage_).forEach(std::forward<Fn>(fn));
                    break;
                case Mode::Sparse:
                    std::get<IndexedConstraintSet>(storage_).forEach(std::forward<Fn>(fn));
                    break;
                default:
                    throw std::runtime_error("ConstraintContainer::forEach: container is empty");
            }
        }

        /// @brief Const version of forEach()
        template<typename Fn>
        void forEach(Fn&& fn) const {
            switch (mode()) {
                case Mode::Dense:
                    std::get<ConstraintGroup>(storage_).forEach(std::forward<Fn>(fn));
                    break;
                case Mode::Sparse:
                    std::get<IndexedConstraintSet>(storage_).forEach(std::forward<Fn>(fn));
                    break;
                default:
                    throw std::runtime_error("ConstraintContainer::forEach: container is empty");
            }
        }
    };


    // ============================================================================
    // INTERNAL HELPERS
    // ============================================================================
    /**
     * @internal
     * Helper utilities for ConstraintFactory::addIndexed.
     * Handles conversion between tuple/scalar indices and std::vector<int>.
     */
    namespace constraint_detail {

        template<typename T, typename = void>
        struct is_tuple_like : std::false_type {};

        template<typename T>
        struct is_tuple_like<T,
            std::void_t<decltype(std::tuple_size<T>::value)>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_tuple_like_v = is_tuple_like<T>::value;

        // converts scalar or tuple index into vector<int>
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
                    "index_to_vector: index must be int or tuple of ints");
                return std::vector<int>{ static_cast<int>(idx) };
            }
        }

        // calls generator with either scalar or expanded tuple
        template<typename Generator, typename Idx>
        GRBTempConstr invoke_on_index(Generator& gen, Idx&& idx) {
            using Raw = std::remove_cvref_t<Idx>;
            if constexpr (is_tuple_like_v<Raw>) {
                return std::apply(
                    [&](auto&&... args) {
                        return gen(std::forward<decltype(args)>(args)...);
                    },
                    std::forward<Idx>(idx)
                );
            }
            else {
                return gen(std::forward<Idx>(idx));
            }
        }

    } // namespace constraint_detail


    // ============================================================================
    // CONSTRAINT FACTORY
    // ============================================================================
    /**
     * @class ConstraintFactory
     * @brief Unified backend for constraint creation
     * @details
     *  Provides static methods for creating constraints:
     *  - add(): Creates rectangular ConstraintGroup (dense N-D arrays)
     *  - addIndexed(): Creates IndexedConstraintSet from arbitrary domains
     *
     *  **Generator Functions**
     *
     *  A generator is a callable that produces a `GRBTempConstr` for each index.
     *  The signature differs between `add()` and `addIndexed()`:
     *
     *  - **add()** generator signature:
     *    ```
     *    GRBTempConstr gen(const std::vector<int>& idx)
     *    ```
     *    The `idx` vector contains the current multi-dimensional index.
     *    For scalar constraints (no sizes), `idx` is empty.
     *
     *  - **addIndexed()** generator signature:
     *    ```
     *    GRBTempConstr gen(int i, int j, ...)  // one argument per dimension
     *    ```
     *    Arguments are unpacked from the domain's tuples.
     *    For 1-D domains, the generator receives a single `int`.
     *
     *  Naming behavior is controlled by naming_enabled():
     *  - Debug mode: human-readable names (cap[3], flow[2,5], etc.)
     *  - Release mode: no symbolic names for performance
     *
     * @example
     *   // Rectangular 2-D constraints using add()
     *   auto cap = ConstraintFactory::add(model, "cap",
     *       [&](const std::vector<int>& idx) {
     *           int i = idx[0], j = idx[1];
     *           return x(i, j) <= capacity[i][j];
     *       }, 10, 5);  // 10x5 array
     *
     *   // Domain-based constraints using addIndexed()
     *   dsl::IndexList I{1, 2, 3};
     *   dsl::IndexList J{4, 5};
     *   auto flow = ConstraintFactory::addIndexed(model, "flow", I * J,
     *       [&](int i, int j) {
     *           return x(i, j) <= demand[i][j];
     *       });
     *
     *   // Filtered domain
     *   auto upper = ConstraintFactory::addIndexed(model, "upper",
     *       (I * J) | dsl::filter([](int i, int j) { return i < j; }),
     *       [&](int i, int j) { return x(i, j) <= 100; });
     */
    class ConstraintFactory {
    public:
        using Node = ConstraintGroup::Node;

        // ---------------------------------------------------------------------
        // Rectangular constraint groups
        // ---------------------------------------------------------------------
        /**
         * @brief Create a dense N-dimensional ConstraintGroup
         * @tparam Generator Callable with signature `GRBTempConstr(const std::vector<int>&)`
         * @tparam Sizes Integral dimension sizes
         * @param model The Gurobi model to add constraints to
         * @param baseName Base name for constraint naming (e.g., "cap" ? cap[0], cap[1], ...)
         * @param gen Generator function that produces a GRBTempConstr for each index
         * @param sizes Dimension sizes (e.g., `10, 5` for a 10×5 array)
         * @return ConstraintGroup containing all created constraints
         *
         * @note If no sizes are provided, creates a scalar constraint.
         *       The generator receives an empty vector for scalar constraints.
         */
        template<typename Generator, typename... Sizes>
        static auto add(
            GRBModel& model,
            const std::string& baseName,
            Generator&& gen,
            Sizes... sizes)
        {
            static_assert((std::is_integral_v<Sizes> && ...),
                "ConstraintFactory::add: sizes must be integral");

            if constexpr (sizeof...(sizes) == 0) {
                // Scalar constraint
                GRBTempConstr tmp = gen(std::vector<int>{});
                std::string   nm = make_name::concat(baseName);
                GRBConstr     c = addConstrOpt(model, tmp, nm);
                return ConstraintGroup(std::move(c));
            }
            else {
                // N-D
                std::vector<int> indices;
                Generator genLocal = std::forward<Generator>(gen);

                Node root = addNodeImpl(
                    model, baseName,
                    genLocal,
                    indices,
                    sizes...);

                return ConstraintGroup(
                    std::move(root),
                    static_cast<int>(sizeof...(sizes)));
            }
        }

        // ---------------------------------------------------------------------
        // Domain-based indexed constraint sets
        // ---------------------------------------------------------------------
        /**
         * @brief Create an IndexedConstraintSet from an arbitrary domain
         * @tparam Domain Iterable domain (e.g., IndexList, RangeView, Cartesian, Filtered)
         * @tparam Generator Callable with signature `GRBTempConstr(int, int, ...)` matching domain arity
         * @param model The Gurobi model to add constraints to
         * @param baseName Base name for constraint naming (e.g., "flow" ? flow[1,4], flow[1,5], ...)
         * @param domain Index domain to iterate over (supports Cartesian products and filters)
         * @param gen Generator function that produces a GRBTempConstr for each index tuple
         * @return IndexedConstraintSet containing all created constraints
         *
         * @note For 1-D domains (e.g., IndexList), generator receives a single int.
         *       For N-D domains (e.g., I * J), generator receives N separate int arguments.
         */
        template<typename Domain, typename Generator>
        static IndexedConstraintSet addIndexed(
            GRBModel& model,
            const std::string& baseName,
            const Domain& domain,
            Generator&& gen)
        {
            IndexedConstraintSet result;
            Generator genLocal = std::forward<Generator>(gen);

            for (auto&& rawIdx : domain) {
                auto idxVec = constraint_detail::index_to_vector(rawIdx);

                GRBTempConstr tmp =
                    constraint_detail::invoke_on_index(genLocal, rawIdx);

                std::string name =
                    make_name::math(baseName, idxVec);

                GRBConstr c = addConstrOpt(model, tmp, name);
                result.addEntry(std::move(c), std::move(idxVec));
            }

            return result;
        }

    private:
        // ---------------------------------------------------------------------
        // Rectangular tree construction
        // ---------------------------------------------------------------------
        template<typename Generator, typename SizeType, typename... Sizes>
        static Node addNodeImpl(
            GRBModel& model,
            const std::string& baseName,
            Generator& gen,
            std::vector<int>& idx,
            SizeType n,
            Sizes... sizes)
        {
            static_assert(std::is_integral_v<SizeType>,
                "ConstraintFactory::addNodeImpl: size must be integral");

            if (n < 0) {
                throw std::invalid_argument(
                    std::format("ConstraintFactory::addNodeImpl: negative size {}", n));
            }

            Node node(static_cast<size_t>(n));

            for (SizeType i = 0; i < n; ++i) {
                idx.push_back(static_cast<int>(i));

                if constexpr (sizeof...(Sizes) == 0) {
                    std::string   nm = make_name::math(baseName, idx);
                    GRBTempConstr tmp = gen(idx);
                    node.children[static_cast<size_t>(i)] =
                        Node(addConstrOpt(model, tmp, nm));
                }
                else {
                    node.children[static_cast<size_t>(i)] =
                        addNodeImpl(model, baseName, gen, idx, sizes...);
                }

                idx.pop_back();
            }

            return node;
        }

        // ---------------------------------------------------------------------
        // Naming-aware addConstr wrapper
        // ---------------------------------------------------------------------
        static inline GRBConstr addConstrOpt(
            GRBModel& model,
            const GRBTempConstr& tmp,
            const std::string& name)
        {
            if constexpr (naming_enabled()) {
                return model.addConstr(tmp, name);
            }
            else {
                return model.addConstr(tmp);
            }
        }
    };


    // ============================================================================
    // CONSTRAINT TABLE
    // ============================================================================
    /**
     * @class ConstraintTable
     * @brief Enum-keyed registry for organizing constraint collections
     *
     * @tparam EnumT Enum class with COUNT sentinel
     * @tparam MAX Number of enum values, defaults to EnumT::COUNT
     *
     * @details Provides a fixed-size array of ConstraintContainer objects indexed by
     *          enum values. Each entry can independently hold either dense (ConstraintGroup)
     *          or sparse (IndexedConstraintSet) constraints, enabling mixed storage in a
     *          single table.
     *
     * @note Mirrors VariableTable for API symmetry.
     *
     * @example
     *     DECLARE_ENUM_WITH_COUNT(Cons, Capacity, Flow, Balance);
     *
     *     ConstraintTable<Cons> ct;
     *
     *     // Mix dense and sparse in the same table
     *     auto cap = ConstraintFactory::add(model, "cap", gen, 10);
     *     auto flow = ConstraintFactory::addIndexed(model, "flow", I * J, gen);
     *
     *     ct.set(Cons::Capacity, std::move(cap));  // Dense
     *     ct.set(Cons::Flow, std::move(flow));     // Sparse
     *
     *     // Unified access works for both
     *     GRBConstr& c1 = ct.constr(Cons::Capacity, 3);
     *     GRBConstr& c2 = ct.constr(Cons::Flow, 1, 2);
     *
     *     // Mode-specific access when needed
     *     if (ct.get(Cons::Capacity).isDense()) {
     *         auto shape = ct.get(Cons::Capacity).asGroup().shape();
     *     }
     *
     * @see ConstraintContainer
     * @see ConstraintGroup
     * @see IndexedConstraintSet
     */
    template<
        typename EnumT,
        std::size_t MAX = static_cast<std::size_t>(EnumT::COUNT)>
    class ConstraintTable {
    private:
        std::array<ConstraintContainer, MAX> table_;  ///< Fixed-size array of containers

    public:
        // ========================================================================
        // SETTERS
        // ========================================================================

        /**
         * @brief Set a container by enum key
         * @param key Enum key
         * @param container ConstraintContainer to store (moved)
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, ConstraintContainer&& container) {
            std::size_t idx = static_cast<std::size_t>(key);
            if (idx >= MAX) {
                throw std::out_of_range(
                    std::format("ConstraintTable::set: key {} >= {}", idx, MAX));
            }
            table_[idx] = std::move(container);
        }

        /**
         * @brief Set a ConstraintGroup by enum key (dense mode)
         * @param key Enum key
         * @param group ConstraintGroup to store (moved)
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, ConstraintGroup&& group) {
            set(key, ConstraintContainer(std::move(group)));
        }

        /**
         * @brief Set a ConstraintGroup by enum key (copy)
         * @param key Enum key
         * @param group ConstraintGroup to copy
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, const ConstraintGroup& group) {
            set(key, ConstraintContainer(group));
        }

        /**
         * @brief Set an IndexedConstraintSet by enum key (sparse mode)
         * @param key Enum key
         * @param indexedSet IndexedConstraintSet to store (moved)
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, IndexedConstraintSet&& indexedSet) {
            set(key, ConstraintContainer(std::move(indexedSet)));
        }

        /**
         * @brief Set an IndexedConstraintSet by enum key (copy)
         * @param key Enum key
         * @param indexedSet IndexedConstraintSet to copy
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, const IndexedConstraintSet& indexedSet) {
            set(key, ConstraintContainer(indexedSet));
        }

        /**
         * @brief Set a scalar GRBConstr by enum key
         * @param key Enum key
         * @param c GRBConstr to store
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, const GRBConstr& c) {
            set(key, ConstraintContainer(c));
        }

        /**
         * @brief Set a scalar GRBConstr by enum key (move)
         * @param key Enum key
         * @param c GRBConstr to store (moved)
         * @throws std::out_of_range if key >= MAX
         */
        void set(EnumT key, GRBConstr&& c) {
            set(key, ConstraintContainer(std::move(c)));
        }

        // ========================================================================
        // GETTERS
        // ========================================================================

        /**
         * @brief Get a container by enum key
         * @param key Enum key
         * @return Mutable reference to the ConstraintContainer
         * @throws std::out_of_range if key >= MAX
         */
        ConstraintContainer& get(EnumT key) {
            std::size_t idx = static_cast<std::size_t>(key);
            if (idx >= MAX) {
                throw std::out_of_range(
                    std::format("ConstraintTable::get: key {} >= {}", idx, MAX));
            }
            return table_[idx];
        }

        /// @brief Const version of get()
        const ConstraintContainer& get(EnumT key) const {
            std::size_t idx = static_cast<std::size_t>(key);
            if (idx >= MAX) {
                throw std::out_of_range(
                    std::format("ConstraintTable::get: key {} >= {}", idx, MAX));
            }
            return table_[idx];
        }

        /// @brief Alias for get() using function call syntax
        ConstraintContainer& operator()(EnumT key) { return get(key); }

        /// @brief Const alias for get() using function call syntax
        const ConstraintContainer& operator()(EnumT key) const { return get(key); }

        // ========================================================================
        // DIRECT CONSTRAINT ACCESS
        // ========================================================================

        /**
         * @brief Direct access to a constraint within a container
         * @tparam I Integral index types
         * @param key Enum key for the container
         * @param idx Indices within the container (empty for scalar)
         * @return Mutable reference to the GRBConstr
         * @throws std::out_of_range if key >= MAX or indices are invalid
         *
         * @example
         *     GRBConstr& c = ct.constr(Cons::Capacity, 3);  // Access Capacity(3)
         *     GRBConstr& s = ct.constr(Cons::Single);        // Access scalar
         */
        template<typename... I>
        GRBConstr& constr(EnumT key, I... idx) {
            static_assert((std::is_integral_v<I> && ...),
                "ConstraintTable::constr: indices must be integral");

            std::size_t k = static_cast<std::size_t>(key);
            if (k >= MAX) {
                throw std::out_of_range(
                    std::format("ConstraintTable::constr: key {} >= {}", k, MAX));
            }

            if constexpr (sizeof...(idx) == 0) {
                return table_[k].scalar();
            }
            else {
                return table_[k].at(idx...);
            }
        }

        /// @brief Const version of constr()
        template<typename... I>
        const GRBConstr& constr(EnumT key, I... idx) const {
            static_assert((std::is_integral_v<I> && ...),
                "ConstraintTable::constr: indices must be integral");

            std::size_t k = static_cast<std::size_t>(key);
            if (k >= MAX) {
                throw std::out_of_range(
                    std::format("ConstraintTable::constr: key {} >= {}", k, MAX));
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
         * @return true if entry holds a ConstraintGroup
         * @throws std::out_of_range if key >= MAX
         */
        bool isDense(EnumT key) const {
            return get(key).isDense();
        }

        /**
         * @brief Check if entry at key is in sparse mode
         * @param key Enum key
         * @return true if entry holds an IndexedConstraintSet
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
    // CONSTRAINT ATTRIBUTE ACCESS
    // ============================================================================
    /**
     * @defgroup ConstraintAttributes Constraint Attribute Utilities
     * @brief Free functions for accessing and modifying constraint attributes
     *
     * @details These utilities provide convenient access to constraint attributes,
     *          mirroring the variable modification utilities in variables.h.
     *          All functions work with individual GRBConstr objects.
     *
     * @note Post-optimization attributes (slack, dual) require the model to be
     *       in an optimized state with a valid solution.
     *
     * @example
     *     double r = dsl::rhs(c);           // Get RHS
     *     dsl::setRHS(c, 100.0);            // Modify RHS
     *     char s = dsl::sense(c);           // Get sense
     *     double sl = dsl::slack(c);        // Post-optimization slack
     *     double pi = dsl::dual(c);         // LP dual value
     *
     * @{
     */

    /**
     * @brief Get the right-hand side of a constraint
     *
     * @param c Constraint to query
     * @return RHS value (GRB_DoubleAttr_RHS)
     *
     * @throws GRBException on Gurobi API error
     * @complexity O(1)
     */
    inline double rhs(const GRBConstr& c) {
        return c.get(GRB_DoubleAttr_RHS);
    }

    /**
     * @brief Set the right-hand side of a constraint
     *
     * @param c Constraint to modify
     * @param val New RHS value
     *
     * @throws GRBException on Gurobi API error
     * @complexity O(1)
     *
     * @note Changes take effect after model.update() or at next optimize().
     *
     * @example
     *     dsl::setRHS(capacity_constr, 150.0);  // Increase capacity
     *     model.optimize();
     */
    inline void setRHS(GRBConstr& c, double val) {
        c.set(GRB_DoubleAttr_RHS, val);
    }

    /**
     * @brief Get the sense of a constraint
     *
     * @param c Constraint to query
     * @return Sense character: '<' (<=), '>' (>=), or '=' (==)
     *
     * @throws GRBException on Gurobi API error
     * @complexity O(1)
     */
    inline char sense(const GRBConstr& c) {
        return c.get(GRB_CharAttr_Sense);
    }

    /**
     * @brief Get the slack value of a constraint (post-optimization)
     *
     * @param c Constraint to query
     * @return Slack value (GRB_DoubleAttr_Slack)
     *
     * @throws GRBException if model not optimized or no solution available
     * @complexity O(1)
     *
     * @note For <= constraints: slack = RHS - LHS (positive if not tight)
     * @note For >= constraints: slack = LHS - RHS (positive if not tight)
     * @note For == constraints: slack should be 0 (or near 0)
     */
    inline double slack(const GRBConstr& c) {
        return c.get(GRB_DoubleAttr_Slack);
    }

    /**
     * @brief Get the dual value (shadow price) of a constraint (LP only)
     *
     * @param c Constraint to query
     * @return Dual value (GRB_DoubleAttr_Pi)
     *
     * @throws GRBException if model not optimized, no solution, or MIP model
     * @complexity O(1)
     *
     * @note Only available for LP models (no integer variables).
     * @note Represents the marginal value of relaxing the constraint.
     */
    inline double dual(const GRBConstr& c) {
        return c.get(GRB_DoubleAttr_Pi);
    }

    /**
     * @brief Get the name of a constraint
     *
     * @param c Constraint to query
     * @return Constraint name (GRB_StringAttr_ConstrName)
     *
     * @throws GRBException on Gurobi API error
     * @complexity O(1)
     */
    inline std::string constrName(const GRBConstr& c) {
        return c.get(GRB_StringAttr_ConstrName);
    }

    /**
     * @brief Set the name of a constraint
     *
     * @param c Constraint to modify
     * @param name New constraint name
     *
     * @throws GRBException on Gurobi API error
     * @complexity O(1)
     */
    inline void setConstrName(GRBConstr& c, const std::string& name) {
        c.set(GRB_StringAttr_ConstrName, name);
    }

    /** @} */ // end of ConstraintAttributes group


    // ============================================================================
    // CONSTRAINT SOLUTION EXTRACTION
    // ============================================================================
    /**
     * @defgroup ConstraintSolution Constraint Solution Utilities
     * @brief Functions for retrieving post-optimization constraint values
     *
     * @details After optimization, these utilities extract slack and dual
     *          values from constraint collections.
     *
     * @{
     */

    /**
     * @brief Get all slack values from a ConstraintGroup
     *
     * @param cg The ConstraintGroup to query
     * @return Vector of slack values in iteration order
     *
     * @throws GRBException if model not optimized or no solution available
     * @complexity O(n) where n = total number of constraints
     */
    inline std::vector<double> slacks(const ConstraintGroup& cg) {
        std::vector<double> result;
        const_cast<ConstraintGroup&>(cg).forEach([&](GRBConstr& c, const std::vector<int>&) {
            result.push_back(c.get(GRB_DoubleAttr_Slack));
        });
        return result;
    }

    /**
     * @brief Get all slack values from an IndexedConstraintSet
     *
     * @param cs The IndexedConstraintSet to query
     * @return Vector of slack values in storage order
     *
     * @throws GRBException if model not optimized or no solution available
     * @complexity O(n) where n = number of constraints
     */
    inline std::vector<double> slacks(const IndexedConstraintSet& cs) {
        std::vector<double> result;
        result.reserve(cs.size());
        for (const auto& e : cs.all()) {
            result.push_back(e.constr.get(GRB_DoubleAttr_Slack));
        }
        return result;
    }

    /**
     * @brief Get all dual values from a ConstraintGroup (LP only)
     *
     * @param cg The ConstraintGroup to query
     * @return Vector of dual values in iteration order
     *
     * @throws GRBException if model not optimized, no solution, or MIP model
     * @complexity O(n) where n = total number of constraints
     */
    inline std::vector<double> duals(const ConstraintGroup& cg) {
        std::vector<double> result;
        const_cast<ConstraintGroup&>(cg).forEach([&](GRBConstr& c, const std::vector<int>&) {
            result.push_back(c.get(GRB_DoubleAttr_Pi));
        });
        return result;
    }

    /**
     * @brief Get all dual values from an IndexedConstraintSet (LP only)
     *
     * @param cs The IndexedConstraintSet to query
     * @return Vector of dual values in storage order
     *
     * @throws GRBException if model not optimized, no solution, or MIP model
     * @complexity O(n) where n = number of constraints
     */
    inline std::vector<double> duals(const IndexedConstraintSet& cs) {
        std::vector<double> result;
        result.reserve(cs.size());
        for (const auto& e : cs.all()) {
            result.push_back(e.constr.get(GRB_DoubleAttr_Pi));
        }
        return result;
    }

    /**
     * @brief Get all slack values from a ConstraintContainer
     *
     * @param cc The ConstraintContainer to query
     * @return Vector of slack values in iteration order
     *
     * @throws GRBException if model not optimized or no solution available
     * @throws std::runtime_error if container is empty
     * @complexity O(n) where n = total number of constraints
     */
    inline std::vector<double> slacks(const ConstraintContainer& cc) {
        std::vector<double> result;
        cc.forEach([&](const GRBConstr& c, const std::vector<int>&) {
            result.push_back(c.get(GRB_DoubleAttr_Slack));
        });
        return result;
    }

    /**
     * @brief Get all dual values from a ConstraintContainer (LP only)
     *
     * @param cc The ConstraintContainer to query
     * @return Vector of dual values in iteration order
     *
     * @throws GRBException if model not optimized, no solution, or MIP model
     * @throws std::runtime_error if container is empty
     * @complexity O(n) where n = total number of constraints
     */
    inline std::vector<double> duals(const ConstraintContainer& cc) {
        std::vector<double> result;
        cc.forEach([&](const GRBConstr& c, const std::vector<int>&) {
            result.push_back(c.get(GRB_DoubleAttr_Pi));
        });
        return result;
    }

    /** @} */ // end of ConstraintSolution group

} // namespace dsl

