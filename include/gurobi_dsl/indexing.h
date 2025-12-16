/*
===============================================================================
INDEXING SYSTEM — Lazy discrete index domains for the C++ Gurobi DSL
===============================================================================

OVERVIEW
--------
Implements the indexing layer of the DSL. Provides a small, consistent set of
"index domain" types representing integer index sets and their combinations.
Bridges mathematical notation and efficient C++ iteration without exposing model
details.

KEY COMPONENTS
--------------
• dsl::IndexList — Materialized, ordered list of integer indices (`std::vector<int>`)
• dsl::RangeView / dsl::range_view() — Lazy half-open range [begin, end) with positive step
• dsl::range(begin, end) — Materialized helper for [begin, end)
• Set operations on IndexList — `+` union, `&` intersection, `-` difference, `^` symmetric difference
• dsl::Cartesian<Sets...> — Lazy N-dimensional Cartesian product for set-like types
• dsl::Filtered<Product, Pred> — Lazy filtered view over any product
• dsl::filter(...) + operator| — Pipe adaptor that combines predicates with logical AND
• Printing utilities (operator<<) — Human-readable formatting

DESIGN PHILOSOPHY
-----------------
• Predictable and stable ordering: insertion order preserved; lexicographic product order; filters preserve order
• Low overhead and laziness: minimal storage, no per-iteration allocations
• Simple, uniform interface: each set-like exposes `size()`, `operator[]`, `begin()/end()`
• Extensibility: any set-like type with the above interface participates naturally

USAGE EXAMPLES
--------------
• IndexList
    dsl::IndexList I{1, 3, 7};
    for (int i : I) { // visits 1,3,7 in order // }

• Range construction
    std::vector<int> v = {4, 5, 6};
    dsl::IndexList J(v);

• Materialized integer range
    auto K = dsl::range(2, 5); // {2,3,4}

• Lazy stepped range
    auto R = dsl::range_view(0, 10, 2); // 0,2,4,6,8

• Set operations
    auto U = I + K;                // union: keep I's order, add new from K
    auto D = I - dsl::IndexList{3}; // {1, 7}

• Cartesian products
    for (auto [i, k] : I * K) { // tuples in lexicographic order // }

• Filtering
    auto even = dsl::range_view(0, 10) | dsl::filter([](int x){ return x % 2 == 0; });

DEPENDENCIES
------------
• <tuple>, <array>, <utility>, <vector>, <initializer_list>, <algorithm>
• <ranges>, <type_traits>, <iterator>, <ostream>

PERFORMANCE NOTES
-----------------
• IndexList: O(1) random access; filtering is linear in size
• RangeView: O(1) storage; indexing and iteration are arithmetic-only
• Cartesian: nested-loop indexing; no dynamic allocation per iteration
• Filtered: wraps underlying range; skips non-matching elements lazily

THREAD SAFETY
-------------
• All views are value types; concurrent const iteration is safe
• Mutations of underlying containers require external synchronization

EXCEPTION SAFETY
----------------
• RangeView construction is noexcept aside from arithmetic; invalid parameters yield empty ranges
• IndexList operations use standard library guarantees; no custom throwing behavior

===============================================================================
*/

#pragma once

#include <tuple>
#include <array>
#include <utility>
#include <vector>
#include <initializer_list>
#include <algorithm>
#include <ranges>
#include <type_traits>
#include <iterator>
#include <ostream>

namespace dsl {

    // ============================================================================
    // INDEX SET
    // ============================================================================
    /**
     * @class IndexList
     * @brief Finite, ordered collection of integer indices
     * @details
     *  - Backed by `std::vector<int>`
     *  - Preserves insertion order and allows duplicates
     *  - Does not sort or deduplicate automatically
     *
     * This behavior is intentional for reproducibility and mapping indices to
     * external data in modeling DSLs.
     *
     * @example
     *   dsl::IndexList I{1, 3, 7};
     *   for (int i : I) { // ... // }
     */
    class IndexList {
    private:
        std::vector<int> data_;

    public:
        // ------------------------------------------------------------------------
        // Constructors
        // ------------------------------------------------------------------------

        /// @brief Default constructor (empty index set)
        /// @complexity Constant time
        /// @noexcept
        IndexList() = default;

        /// @brief Construct from initializer list
        /// @example IndexList I{1,2,3};
        IndexList(std::initializer_list<int> init)
            : data_(init)
        {
        }

        /**
         * @brief Construct from any `std::ranges::input_range` of integral type
         * @example
         *   std::vector<int> v = {4,5,6};
         *   dsl::IndexList I(v);
         */
        template<std::ranges::input_range Range>
            requires std::is_integral_v<std::ranges::range_value_t<Range>>
        IndexList(const Range& r)
            : data_(std::ranges::begin(r), std::ranges::end(r))
        {
        }

        /// @brief Construct with n default-initialized (zero) elements
        /// @example IndexList I(5); // {0,0,0,0,0}
        explicit IndexList(std::size_t n)
            : data_(n)
        {
        }

        /// @brief Internal constructor from vector (used by helpers like `range()`)
        explicit IndexList(std::vector<int>&& v)
            : data_(std::move(v))
        {
        }

        // Optional convenience (not used by tests, but handy for DSL code)
        void push_back(int v) { data_.push_back(v); }
        void reserve(std::size_t n) { data_.reserve(n); }

        // ------------------------------------------------------------------------
        // Iteration
        // ------------------------------------------------------------------------

        /// @brief Begin iterator
        /// @noexcept
        auto begin() noexcept { return data_.begin(); }

        /// @brief Begin iterator (const)
        /// @noexcept
        auto begin() const noexcept { return data_.begin(); }

        /// @brief End iterator
        /// @noexcept
        auto end() noexcept { return data_.end(); }

        /// @brief End iterator (const)
        /// @noexcept
        auto end() const noexcept { return data_.end(); }

        // ------------------------------------------------------------------------
        // Basic properties
        // ------------------------------------------------------------------------

        /// @brief Number of elements
        /// @return Count of stored indices
        /// @noexcept
        int size() const noexcept { return static_cast<int>(data_.size()); }

        /// @brief Returns true if the set is empty
        /// @noexcept
        bool empty() const noexcept { return data_.empty(); }

        /// @brief Check whether the set contains `value`
        /// @complexity O(size()) linear search
        /// @noexcept
        bool contains(int value) const noexcept {
            return std::find(data_.begin(), data_.end(), value) != data_.end();
        }

        /// @brief Read-only access to the internal vector
        /// @details Intended for low-level extensions and debugging; most users
        ///          should iterate with range-for instead.
        /// @noexcept
        const std::vector<int>& raw() const noexcept {
            return data_;
        }

        // ------------------------------------------------------------------------
        // Random access by position (for Cartesian products and views)
        // ------------------------------------------------------------------------
        /**
         * @brief Return the element at logical position `i`
         * @details Used by `Cartesian<Sets...>` to index each dimension
         */
        int operator[](std::size_t i) const {
            return data_[i];
        }

        // ------------------------------------------------------------------------
        // Filtering: multi-predicate version
        // ------------------------------------------------------------------------
        /**
         * @brief Create a lazy filtered view of this set
         * @details Each predicate must be callable as `bool pred(int)`.
         *          All predicates are combined with logical AND.
         * @example
         *   dsl::IndexList I{1,2,3,4,5,6};
         *   auto even_gt3 = I.filter(
         *       [](int x){ return x % 2 == 0; },
         *       [](int x){ return x > 3; }
         *   ); // yields 4, 6
         */
        template<typename... Preds>
        auto filter(const Preds&... preds) const;
    };

    // ============================================================================
    // SET OPERATIONS FOR IndexList
    // ============================================================================
    //
    // Semantics:
    //  - Union (+):        keep A's order, append elements from B that are not in A
    //  - Intersection (&): preserve A's order, keep elements that are also in B
    //  - Difference (-):   preserve A's order, remove elements that are in B
    //  - Symmetric (^):    (A - B) + (B - A)
    //
    // Notes:
    //  - Duplicates *within* a single IndexList are preserved.
    //  - Deduplication is only done across operands for union.
    // ============================================================================

    /// Internal helper: membership test consistent with IndexList behavior.
    /// @brief Internal helper: membership test consistent with `IndexList`
    /// @noexcept
    inline bool contains_linear(const IndexList& S, int x)
    {
        return S.contains(x);
    }

    /// Union: A + B (preserves order of A).
    /// @brief Union operator: `A + B` (preserves order of A)
    inline IndexList operator+(const IndexList& A, const IndexList& B)
    {
        std::vector<int> out;
        out.reserve(static_cast<std::size_t>(A.size()) +
            static_cast<std::size_t>(B.size()));

        // Copy A fully (preserve order).
        for (int x : A.raw())
            out.push_back(x);

        // Append elements from B that are not already in A or already added.
        for (int x : B.raw()) {
            if (!contains_linear(A, x)) {
                // Also check if x is already in out (to handle duplicates in B)
                bool already_added = std::find(out.begin(), out.end(), x) != out.end();
                if (!already_added)
                    out.push_back(x);
            }
        }

        return IndexList(std::move(out));
    }

    /// Intersection: A & B (preserves order of A).
    /// @brief Intersection operator: `A & B` (preserves order of A)
    inline IndexList operator&(const IndexList& A, const IndexList& B)
    {
        std::vector<int> out;
        out.reserve(static_cast<std::size_t>(std::min(A.size(), B.size())));

        for (int x : A.raw())
            if (contains_linear(B, x))
                out.push_back(x);

        return IndexList(std::move(out));
    }

    /// Difference: A - B (preserves order of A).
    /// @brief Difference operator: `A - B` (preserves order of A)
    inline IndexList operator-(const IndexList& A, const IndexList& B)
    {
        std::vector<int> out;
        out.reserve(static_cast<std::size_t>(A.size()));

        for (int x : A.raw())
            if (!contains_linear(B, x))
                out.push_back(x);

        return IndexList(std::move(out));
    }

    /// Symmetric difference: A ^ B = (A - B) + (B - A).
    /// @brief Symmetric difference: `A ^ B = (A - B) + (B - A)`
    inline IndexList operator^(const IndexList& A, const IndexList& B)
    {
        return (A - B) + (B - A);
    }

    // ============================================================================
    // RANGEVIEW: LAZY HALF-OPEN INTEGER RANGE [begin, end) WITH POSITIVE STEP
    // ============================================================================
    /**
     * Lazy range of integers [begin, end) with positive step.
     *
     *  - No allocation, uses only three integers internally.
     *  - Integrates with Cartesian and Filtered via:
     *       size()
     *       operator[](size_t)
     *       begin()/end()
     *
     * Example:
     *   auto R = dsl::range_view(0, 10, 3); // 0,3,6,9
     *   for (int x : R) { ... }
     *
     *   // Cartesian product with another set-like:
     *   for (auto [i,j] : dsl::range_view(0,3) * dsl::range_view(10,12)) {
     *       // (0,10), (0,11), (1,10), (1,11), (2,10), (2,11)
     *   }
     */
    /**
     * @class RangeView
     * @brief Lazy half-open integer range [begin, end) with positive step
     * @details No allocation, integrates with Cartesian and Filtered via
     *          `size()`, `operator[]`, and `begin()/end()`.
     * @example
     *   auto R = dsl::range_view(0, 10, 3); // 0,3,6,9
     *   for (int x : R) { // ... // }
     */
    class RangeView {
    public:
        // ------------------------------------------------------------------------
        // Iterator
        // ------------------------------------------------------------------------
        class iterator {
            int current_;
            int step_;

        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = int;
            using difference_type = std::ptrdiff_t;
            using pointer = const int*;
            using reference = int;

            iterator() = default;

            iterator(int current, int step)
                : current_(current)
                , step_(step)
            {
            }

            /// @brief Dereference current value
            int operator*() const { return current_; }

            /// @brief Prefix increment
            iterator& operator++() {
                current_ += step_;
                return *this;
            }

            /// @brief Postfix increment
            iterator operator++(int) {
                iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            /// @brief Equality comparison
            bool operator==(const iterator& other) const noexcept {
                return current_ == other.current_;
            }

            bool operator!=(const iterator& other) const noexcept {
                return !(*this == other);
            }
        };

    private:
        int         start_ = 0;
        int         step_ = 1;
        std::size_t size_ = 0;

        // === UB-safe size computation ==========================================
        // Use long long for the intermediate arithmetic so that (end - begin)
        // and (span + step - 1) cannot trigger signed overflow.
        static std::size_t compute_size(int begin, int end, int step) {
            if (step <= 0) return 0;
            if (end <= begin) return 0;

            const long long span = static_cast<long long>(end)
                - static_cast<long long>(begin);
            const long long s = static_cast<long long>(step);

            const long long num = span + s - 1;   // ceil(span / s)
            if (num <= 0) return 0;               // extra safety

            return static_cast<std::size_t>(num / s);
        }

    public:
        RangeView() = default;

        /**
         * Construct a range [begin, end) with given step (>0).
         *
         * If step <= 0 or end <= begin, the range is empty.
         */
        /// @brief Construct a range [begin, end) with given positive step
        /// @note If step <= 0 or end <= begin, the range is empty
        RangeView(int begin, int end, int step = 1)
            : start_(begin)
            , step_(step)
            , size_(compute_size(begin, end, step))
        {
        }

        // ------------------------------------------------------------------------
        // Basic properties
        // ------------------------------------------------------------------------
        /// @brief Number of elements
        /// @noexcept
        std::size_t size()  const noexcept { return size_; }
        /// @brief Returns true if the range is empty
        /// @noexcept
        bool        empty() const noexcept { return size_ == 0; }

        // ------------------------------------------------------------------------
        // Random access via logical index (for Cartesian)
        // ------------------------------------------------------------------------
        /// @brief Random access by logical index
        int operator[](std::size_t i) const {
            // i is size_t; promote to long long, compute, then cast back.
            const long long idx = static_cast<long long>(i);
            const long long base = static_cast<long long>(start_);
            const long long step = static_cast<long long>(step_);
            return static_cast<int>(base + idx * step);
        }

        // ------------------------------------------------------------------------
        // Iteration
        // ------------------------------------------------------------------------
        /// @brief Begin iterator
        iterator begin() const {
            if (empty())
                return iterator(start_ + static_cast<int>(size_) * step_, step_);
            return iterator(start_, step_);
        }

        /// @brief End iterator
        iterator end() const {
            return iterator(start_ + static_cast<int>(size_) * step_, step_);
        }

        // ------------------------------------------------------------------------
        // Filtering support (reuse Filtered<T,Pred>)
        // ------------------------------------------------------------------------
        /**
         * @brief Create a lazy filtered view of this range
         * @example
         *   auto even = dsl::range_view(0, 10).filter(
         *       [](int x){ return x % 2 == 0; }
         *   ); // yields 0,2,4,6,8
         */
        template<typename... Preds>
        auto filter(const Preds&... preds) const;
    };

    /// Convenience helper: build a RangeView [begin, end) with optional step.
    /// @brief Convenience helper: build a `RangeView` [begin, end) with optional step
    inline RangeView range_view(int begin, int end, int step = 1) {
        return RangeView(begin, end, step);
    }

    // ============================================================================
    // RANGE GENERATOR (MATERIALIZED)
    // ============================================================================
    /**
     * Construct an IndexList representing the half-open range [begin, end).
     *
     * Example:
     *   auto I = dsl::range(2, 5);  // {2,3,4}
     *
     * For large ranges where laziness matters, prefer RangeView:
     *   auto R = dsl::range_view(2, 5);  // lazy 2,3,4
     */
    /// @brief Construct an `IndexList` representing the half-open range [begin, end)
    inline IndexList range(int begin, int end) {
        std::vector<int> v;
        v.reserve((end > begin) ? static_cast<std::size_t>(end - begin) : 0u);

        for (int i = begin; i < end; ++i)
            v.push_back(i);

        return IndexList(std::move(v));
    }

    // ============================================================================
    // DETAIL NAMESPACE: HELPER UTILITIES FOR CARTESIAN AND FILTERING
    // ============================================================================
    /**
     * @internal
     * Helper utilities used by Cartesian<Sets...> and Filtered<Product,Pred>.
     *
     * A "set-like" type used with Cartesian/Filtered must provide:
     *   - size()             -> number of elements
     *   - int operator[](i)  -> logical i-th element
     *   - begin()/end()      -> for use with filtering and range-for
     */
    namespace detail {

        // ------------------------------------------------------------------------
        // Build tuple<const Sets*...> from references
        // ------------------------------------------------------------------------
        /// @brief Build `tuple<const Sets*...>` from references
        template<typename... Sets>
        auto make_pointer_tuple(const Sets&... sets) {
            return std::tuple<const Sets*...>{ &sets... };
        }

        // ------------------------------------------------------------------------
        // Read a tuple<int...> from tuple<const Sets*...> and index array
        // ------------------------------------------------------------------------
        /// @brief Read `tuple<int...>` from `tuple<const Sets*...>` and index array
        template<std::size_t... Is, typename Tuple, typename IndexArray>
        auto deref_tuple_impl(const Tuple& ptrs,
            const IndexArray& idx,
            std::index_sequence<Is...>)
        {
            // Each set type must provide:
            //   int operator[](std::size_t) const;
            return std::make_tuple((*std::get<Is>(ptrs))[idx[Is]]...);
        }

        // ------------------------------------------------------------------------
        // Compute sizes[] for each dimension
        // ------------------------------------------------------------------------
        /// @brief Compute sizes[] for each dimension
        template<typename Tuple, std::size_t... Is>
        auto sizes_from_tuple_impl(const Tuple& ptrs, std::index_sequence<Is...>) {
            return std::array<int, sizeof...(Is)>{
                static_cast<int>(std::get<Is>(ptrs)->size())...
            };
        }

        template<typename... Sets>
        auto sizes_from_tuple(const std::tuple<const Sets*...>& ptrs) {
            return sizes_from_tuple_impl(ptrs,
                std::make_index_sequence<sizeof...(Sets)>{});
        }

        // ------------------------------------------------------------------------
        // Helper: apply tuple of ints to predicate of arity N
        // ------------------------------------------------------------------------
        template<typename Pred, typename Tuple, std::size_t... Is>
        bool apply_pred_impl(const Pred& pred,
            const Tuple& t,
            std::index_sequence<Is...>)
        {
            return std::apply(pred, t);
        }

        // ------------------------------------------------------------------------
        // Trait: detect tuple-like types (have std::tuple_size)
        // ------------------------------------------------------------------------
        template<typename T, typename = void>
        struct is_tuple_like : std::false_type {};

        template<typename T>
        struct is_tuple_like<T, std::void_t<decltype(std::tuple_size<T>::value)>>
            : std::true_type {
        };

        template<typename T>
        inline constexpr bool is_tuple_like_v = is_tuple_like<T>::value;

        // ------------------------------------------------------------------------
        // Apply predicate either to a tuple<int...> (expanded) or scalar (int)
        // ------------------------------------------------------------------------
        /// @brief Apply predicate to a tuple<int...> or scalar int
        template<typename Pred, typename T>
        bool apply_pred(const Pred& pred, const T& value)
        {
            using U = std::remove_cvref_t<T>;

            if constexpr (is_tuple_like_v<U>) {
                constexpr std::size_t N = std::tuple_size_v<U>;
                return apply_pred_impl(pred, value,
                    std::make_index_sequence<N>{});
            }
            else {
                // Scalar case (e.g., int from IndexList / RangeView)
                return pred(value);
            }
        }

        // ------------------------------------------------------------------------
        // Combine multiple predicates into a single AND predicate
        // ------------------------------------------------------------------------
        /// @brief Combine multiple predicates into a single AND predicate
        template<typename... Preds>
        struct PredAll {
            std::tuple<Preds...> preds;

            template<typename... Args>
            bool operator()(const Args&... args) const {
                return std::apply(
                    [&](auto const&... ps) {
                        return (ps(args...) && ...);
                    },
                    preds
                );
            }
        };

    } // namespace detail

    // ============================================================================
    // Cartesian<Sets...>: LAZY N-DIMENSIONAL CARTESIAN PRODUCT
    // ============================================================================
    /**
     * @class Cartesian
     * @brief Lazy N-dimensional Cartesian product of set-like objects
     * @details A set-like type must provide: `size()`, `operator[]`, and
     *          `begin()/end()` for iteration in filters. Iteration order is
     *          lexicographic over dimensions.
     * @example
     *   dsl::IndexList I{1,2};
     *   auto J = dsl::range_view(10, 13); // 10,11,12
     *   for (auto [i, j] : I * J) {
     *       // (1,10), (1,11), (1,12), (2,10), (2,11), (2,12)
     *   }
     */
    template<typename... Sets>
    class Cartesian {
    private:
        static constexpr std::size_t N = sizeof...(Sets);
        std::tuple<const Sets*...>   sets_;

    public:
        // ------------------------------------------------------------------------
        // Iterator for N-dimensional product
        // ------------------------------------------------------------------------
        class iterator {
            std::tuple<const Sets*...> ptrs_;
            std::array<int, N>         sizes_;
            std::array<int, N>         idx_;

        public:
            iterator() = default;

            iterator(const std::tuple<const Sets*...>& ptrs, bool is_end)
                : ptrs_(ptrs)
            {
                sizes_ = detail::sizes_from_tuple(ptrs_);

                bool any_empty = false;
                for (std::size_t i = 0; i < N; ++i) {
                    if (sizes_[i] == 0) {
                        any_empty = true;
                        break;
                    }
                }

                if (is_end || any_empty) {
                    idx_.fill(0);
                    if constexpr (N > 0)
                        idx_[0] = sizes_[0];
                    return;
                }

                idx_.fill(0); // begin state
            }

            /// @brief Dereference -> tuple<int,...> with N components
            auto operator*() const {
                return detail::deref_tuple_impl(
                    ptrs_, idx_, std::make_index_sequence<N>{});
            }

            /// @brief Prefix ++ (lexicographic "odometer" increment)
            iterator& operator++() {
                for (int d = static_cast<int>(N) - 1; d >= 0; --d) {
                    ++idx_[static_cast<std::size_t>(d)];
                    if (idx_[static_cast<std::size_t>(d)] < sizes_[static_cast<std::size_t>(d)])
                        return *this;
                    idx_[static_cast<std::size_t>(d)] = 0;
                }

                // overflow -> end
                if constexpr (N > 0)
                    idx_[0] = sizes_[0];
                return *this;
            }

            bool operator==(const iterator& other) const noexcept {
                return idx_ == other.idx_;
            }
            bool operator!=(const iterator& other) const noexcept {
                return !(*this == other);
            }
        };

        // ------------------------------------------------------------------------
        // Constructors + begin/end
        // ------------------------------------------------------------------------
        Cartesian(const Sets&... sets)
            : sets_(detail::make_pointer_tuple(sets...))
        {
        }

        /// @brief Begin iterator
        iterator begin() const { return iterator(sets_, false); }
        /// @brief End iterator
        iterator end()   const { return iterator(sets_, true); }

        // Optional helpers: size/empty (not used by tests, but handy)
        /// @brief Total size of the product (materialized count)
        std::size_t size() const {
            auto sizes = detail::sizes_from_tuple(sets_);
            std::size_t total = 1;
            for (int s : sizes) {
                if (s == 0) return 0;
                total *= static_cast<std::size_t>(s);
            }
            return total;
        }
        /// @brief Returns true if any dimension is empty
        bool empty() const { return size() == 0; }

        // ------------------------------------------------------------------------
        // Internal accessor for operator*
        // ------------------------------------------------------------------------
        /// @brief Accessor for underlying set pointers
        /// @noexcept
        const std::tuple<const Sets*...>& raw_sets() const noexcept {
            return sets_;
        }

        // ------------------------------------------------------------------------
        // Filtering: multi-predicate version
        // ------------------------------------------------------------------------
        /**
         * @brief Create a lazy filtered view of this Cartesian product
         * @details Each predicate must be callable as `bool pred(int, int, ...)`.
         * @example
         *   auto P = (I * J).filter([](int i, int j){ return i < j; });
         *   for (auto [i,j] : P) { // only pairs with i < j // }
         */
        template<typename... Preds>
        auto filter(const Preds&... preds) const;
    };

    // ============================================================================
    // Filtered<Product, Pred>: GENERIC LAZY FILTERING VIEW
    // ============================================================================
    /**
     * Lazy filtered view over any Product (IndexList, RangeView, Cartesian, ...).
     *
     * This is created indirectly by calls to:
     *   - IndexList::filter(...)
     *   - RangeView::filter(...)
     *   - Cartesian::filter(...)
     *   - product | dsl::filter(...)
     *
     * It does not own any indices itself; it simply wraps another range-like
     * object and skips elements that do not satisfy the predicate.
     */
    /**
     * @class Filtered
     * @brief Generic lazy filtering view over any Product
     * @details Wraps another range-like object and skips elements that do not
     *          satisfy the predicate. Does not own indices; preserves order.
     */
    template<typename Product, typename Pred>
    class Filtered {
    private:
        Product product_;
        Pred    pred_;

        using UnderIter = decltype(std::declval<const Product&>().begin());
        using UnderValue = std::remove_cvref_t<decltype(*std::declval<UnderIter>())>;

    public:
        // ------------------------------------------------------------------------
        // Iterator
        // ------------------------------------------------------------------------
        class iterator {
            UnderIter   it_;
            UnderIter   end_;
            const Pred* pred_;

        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = UnderValue;
            using difference_type = std::ptrdiff_t;
            using pointer = const UnderValue*;
            using reference = UnderValue;

            iterator() = default;

            iterator(UnderIter it, UnderIter end, const Pred* pred)
                : it_(it)
                , end_(end)
                , pred_(pred)
            {
                // Advance to first valid element
                while (it_ != end_ && !detail::apply_pred(*pred_, *it_))
                    ++it_;
            }

            auto operator*() const { return *it_; }

            iterator& operator++()
            {
                do { ++it_; } while (it_ != end_ && !detail::apply_pred(*pred_, *it_));
                return *this;
            }

            iterator operator++(int)
            {
                iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            bool operator==(const iterator& other) const noexcept {
                return it_ == other.it_;
            }
            bool operator!=(const iterator& other) const noexcept {
                return it_ != other.it_;
            }
        };

        // ------------------------------------------------------------------------
        // Constructors + begin/end
        // ------------------------------------------------------------------------
        /// @brief Construct filtered view from product and predicate
        Filtered(const Product& p, const Pred& pred)
            : product_(p)
            , pred_(pred)
        {
        }

        /// @brief Begin iterator (advanced to first valid element)
        auto begin() const { return iterator(product_.begin(), product_.end(), &pred_); }
        /// @brief End iterator
        auto end()   const { return iterator(product_.end(), product_.end(), &pred_); }
    };

    // ============================================================================
    // filter() IMPLEMENTATIONS (MEMBERS)
    // ============================================================================

    // For IndexList
    template<typename... Preds>
    auto IndexList::filter(const Preds&... preds) const
    {
        using Combined = detail::PredAll<std::decay_t<Preds>...>;
        Combined combined{ std::make_tuple(preds...) };
        return Filtered<IndexList, Combined>(*this, combined);
    }

    // For Cartesian<Sets...>
    template<typename... Sets>
    template<typename... Preds>
    auto Cartesian<Sets...>::filter(const Preds&... preds) const
    {
        using Combined = detail::PredAll<std::decay_t<Preds>...>;
        Combined combined{ std::make_tuple(preds...) };
        return Filtered<Cartesian<Sets...>, Combined>(*this, combined);
    }

    // For RangeView
    template<typename... Preds>
    auto RangeView::filter(const Preds&... preds) const
    {
        using Combined = detail::PredAll<std::decay_t<Preds>...>;
        Combined combined{ std::make_tuple(preds...) };
        return Filtered<RangeView, Combined>(*this, combined);
    }

    // ============================================================================
    // PIPE ADAPTOR: dsl::filter(...) + operator|
    // ============================================================================
    /**
     * Pipe adaptor for filtering.
     *
     * Examples:
     *   auto F = I | dsl::filter(p1, p2);
     *   auto G = (I * J) | dsl::filter([](int i,int j){ return i < j; });
     *
     * Semantics:
     *   - dsl::filter(preds...) combines all predicates with logical AND.
     *   - Works with any Product that supports begin()/end() and whose
     *     iterator dereference is either an int or a tuple<int,...>.
     *
     * Capturing external variables:
     *   int K = 7;
     *   auto P = (I * I)
     *           | dsl::filter([&](int i,int j){ return i + j <= K; });
     */
    /**
     * @struct filter_adaptor
     * @brief Pipe adaptor for filtering predicates (logical AND)
     */
    template<typename PredAllT>
    struct filter_adaptor {
        PredAllT pred_all;
    };

    /// Build a variadic filter adaptor: dsl::filter(p1, p2, ...).
    /// @brief Build a variadic filter adaptor: `dsl::filter(p1, p2, ...)`
    template<typename... Preds>
    auto filter(Preds&&... preds)
    {
        using Combined = detail::PredAll<std::decay_t<Preds>...>;
        Combined combined{ std::make_tuple(std::forward<Preds>(preds)...) };
        return filter_adaptor<Combined>{ combined };
    }

    /// Pipe operator to create a Filtered<Product, PredAllT> view.
    /// @brief Pipe operator to create a `Filtered<Product, PredAllT>` view
    template<typename Product, typename PredAllT>
    auto operator|(const Product& product, const filter_adaptor<PredAllT>& adaptor)
    {
        return Filtered<Product, PredAllT>(product, adaptor.pred_all);
    }

    // ============================================================================
    // operator* OVERLOADS FOR CARTESIAN PRODUCTS
    // ============================================================================
    //
    // These create Cartesian<Sets...> objects from combinations of IndexList,
    // RangeView, and existing Cartesian instances.
    // ============================================================================

    /// Base case: Cartesian product of two IndexList objects.
    /// @brief Cartesian product: `IndexList * IndexList`
    inline auto operator*(const IndexList& A, const IndexList& B) {
        return Cartesian<IndexList, IndexList>(A, B);
    }

    /// Extend an existing Cartesian with an IndexList on the right.
    /// @brief Extend existing Cartesian with `IndexList` on the right
    template<typename... Sets>
    inline auto operator*(const Cartesian<Sets...>& P, const IndexList& S) {
        return std::apply(
            [&](const Sets*... ptrs) {
                return Cartesian<Sets..., IndexList>(*ptrs..., S);
            },
            P.raw_sets()
        );
    }

    /// Cartesian product: RangeView * RangeView
    /// @brief Cartesian product: `RangeView * RangeView`
    inline auto operator*(const RangeView& A, const RangeView& B)
    {
        return Cartesian<RangeView, RangeView>(A, B);
    }

    /// Cartesian product: RangeView * IndexList
    /// @brief Cartesian product: `RangeView * IndexList`
    inline auto operator*(const RangeView& A, const IndexList& B)
    {
        return Cartesian<RangeView, IndexList>(A, B);
    }

    /// Cartesian product: IndexList * RangeView
    /// @brief Cartesian product: `IndexList * RangeView`
    inline auto operator*(const IndexList& A, const RangeView& B)
    {
        return Cartesian<IndexList, RangeView>(A, B);
    }

    /// Extend an existing Cartesian with a RangeView on the right.
    /// @brief Extend existing Cartesian with `RangeView` on the right
    template<typename... Sets>
    inline auto operator*(const Cartesian<Sets...>& P, const RangeView& R) {
        return std::apply(
            [&](const Sets*... ptrs) {
                return Cartesian<Sets..., RangeView>(*ptrs..., R);
            },
            P.raw_sets()
        );
    }

    // ============================================================================
    // PRINTING UTILITIES (operator<<)
    // ============================================================================
    //
    // Lightweight, human-readable string representations for the main
    // indexing structures. Intended for:
    //
    //   - Diagnostics and logging
    //   - Test failure messages
    //   - Quick inspection in examples and REPLs
    //
    // The format is intentionally simple and stable so that:
    //   - You can grep logs for a given index or tuple.
    //   - Unit tests can compare against fixed strings if needed.
    // ============================================================================

    /**
     * Stream insertion for IndexList.
     *
     * Format (example):
     *   IndexList{1,3,7}  ->  "{1, 3, 7}"
     *
     * Complexity:
     *   - O(size()) over the number of stored indices.
     */
    inline std::ostream& operator<<(std::ostream& os, const IndexList& I)
    {
        os << "{";
        for (std::size_t i = 0; i < I.raw().size(); ++i) {
            os << I.raw()[i];
            if (i + 1 < I.raw().size()) os << ", ";
        }
        os << "}";
        return os;
    }

    /**
     * Stream insertion for RangeView.
     *
     * Format (example):
     *   range_view(0, 10, 2)  ->  "range_view(0, 10, step=2) -> [0, 2, 4, ...]"
     *
     * Notes:
     *   - This is a *view*, so we print the parameters (begin, end, step)
     *     instead of materializing all elements.
     */
    inline std::ostream& operator<<(std::ostream& os, const RangeView& R)
    {
        os << "range_view(";

        if (R.empty()) {
            os << "empty";
            return os << ")";
        }

        int first = R[0];
        int step = (R.size() >= 2 ? R[1] - R[0] : 1);
        int last = R[R.size() - 1];

        os << first << ", " << (last + step) << ", step=" << step << ") -> [";

        const std::size_t limit = 10;
        std::size_t       printed = 0;
        bool              first_elem = true;
        for (auto v : R) {
            if (printed++ >= limit) { os << "..."; break; }
            if (!first_elem) os << ", ";
            os << v;
            first_elem = false;
        }

        os << "]";
        return os;
    }

    /**
     * Stream insertion for Cartesian<Sets...>.
     *
     * Format (example):
     *   I = {1,3}, K = {2,4}
     *   I * K  ->  "( {1, 3} x {2, 4} ) = { (1,2), (1,4), (3,2), (3,4) }"
     *
     * Complexity:
     *   - O(|product|) applications of operator[] on the underlying sets.
     *
     * Intended usage:
     *   - For small products in logs and unit tests.
     *   - Not meant for huge Cartesian products in tight loops.
     */
    template<typename... Sets>
    inline std::ostream& operator<<(std::ostream& os, const Cartesian<Sets...>& C)
    {
        os << "(";

        // Print underlying set representations
        std::apply([&](auto const*... ptrs) {
            std::size_t idx = 0;
            ((os << *ptrs << (++idx < sizeof...(Sets) ? " x " : "")), ...);
            }, C.raw_sets());

        os << ")";

        // Print enumerated tuples
        os << " = { ";

        std::size_t            count = 0;
        constexpr std::size_t  limit = 10;
        bool                   first_tuple = true;

        for (auto tup : C) {
            if (count++ >= limit) { os << "... "; break; }
            if (!first_tuple) os << ", ";
            std::apply([&](auto const&... xs) {
                os << "(";
                std::size_t i = 0;
                ((os << xs << (++i < sizeof...(Sets) ? ", " : "")), ...);
                os << ")";
                }, tup);
            first_tuple = false;
        }

        os << " }";

        return os;
    }

} // namespace dsl