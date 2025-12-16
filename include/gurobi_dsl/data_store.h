#pragma once
/*
===============================================================================
DATA STORE — Lightweight typed parameter/value holder for the DSL
===============================================================================

OVERVIEW
--------
Provides a minimal, flexible container for storing typed values associated with
string keys. Used across the DSL to attach model parameters, metadata, algorithmic
settings, and intermediate computed quantities without requiring variant-based
or template-heavy configuration hierarchies.

KEY COMPONENTS
--------------
• Value: Type-erased wrapper around std::any with safe access methods
• DataStore: Alias for std::unordered_map<std::string, Value>
• Access patterns: try_get(), get(), get_or(), getOrCompute(), getStrictOrCompute()

DESIGN PHILOSOPHY
-----------------
• Zero-cost lookup beyond std::any casting overhead
• Elegant API supporting common idioms (optional access, default values, lazy computation)
• Type-safe with clear failure modes (exceptions vs. nullopt)
• No dynamic polymorphism; fully value-based storage
• Natural usage patterns via operator[] and implicit conversions

USAGE EXAMPLES
--------------
    DataStore params;
    params["time_limit"] = 10.0;
    params["verbose"] = true;

    // Safe access with defaults
    double limit = params["time_limit"].get_or(60.0);

    // Lazy computation with memoization
    int capacity = params["capacity"].getOrCompute<int>([] { return 100; });

    // Strict type enforcement
    int threads = params["threads"].getStrictOrCompute<int>([] { return 4; });

DEPENDENCIES
------------
• <any> - Type-erased storage
• <unordered_map> - Key-value storage
• <optional> - Optional return types
• <typeinfo> - Type introspection

PERFORMANCE NOTES
-----------------
• Value storage: O(1) access, std::any overhead similar to std::variant
• DataStore lookup: O(1) average, O(n) worst-case (hash collisions)
• Memory: Overhead of std::any + std::unordered_map node (~40-60 bytes per entry)
• Copy: Deep copy of stored values; move operations preferred when possible

THREAD SAFETY
-------------
• Value: Thread-safe for concurrent const access; modifications require external synchronization
• DataStore: Not thread-safe; concurrent modifications require external locking
• Note: getOrCompute() and getStrictOrCompute() are not atomic

EXCEPTION SAFETY
----------------
• Value constructors: Strong guarantee (all or nothing)
• get<T>(): Throws std::bad_any_cast on type mismatch
• get_or<T>(): No-throw guarantee (returns default on mismatch)
• getOrCompute<T>(): Strong guarantee (compute may throw)
• getStrictOrCompute<T>(): Strong guarantee, throws std::bad_any_cast on type mismatch

===============================================================================
*/

#include <any>
#include <string>
#include <unordered_map>
#include <typeinfo>
#include <optional>

/**
 * @class Value
 * @brief Type-erased container for any copyable/movable type with safe access methods
 *
 * @details Wraps std::any to provide a clean, intuitive API for storing values of
 *          arbitrary types while maintaining type safety. Supports multiple access
 *          patterns including safe retrieval with defaults, lazy computation, and
 *          strict type enforcement.
 *
 * @note Stored types must be copyable or movable. References and arrays decay to
 *       pointers when stored.
 *
 * @example
 *     Value v = 42;
 *     int x = v.get<int>();           // 42
 *     double y = v.get_or<double>(0); // 0.0 (type mismatch)
 *     v.reset();                      // Clears stored value
 *
 * @see DataStore
 * @see std::any
 */
class Value
{
    std::any storage;

public:
    // ========================================================================
    // CONSTRUCTORS AND ASSIGNMENT
    // ========================================================================

    /**
     * @brief Default constructor; creates an empty Value
     *
     * @complexity Constant time
     * @post has_value() == false
     */
    Value() = default;

    /**
     * @brief Constructs a Value from any storable type
     *
     * @tparam T Type of value to store; must be copyable or movable
     * @param v Value to store; forwarded to std::any
     *
     * @complexity Depends on T's copy/move constructor
     * @post has_value() == true, is<T>() == true
     *
     * @note Uses perfect forwarding; rvalues are moved, lvalues are copied
     */
    template <typename T>
    Value(T&& v)
        : storage(std::forward<T>(v))
    {
    }

    /**
     * @brief Assigns a new value of any type
     *
     * @tparam T Type of value to assign
     * @param v Value to assign
     *
     * @return Reference to *this
     *
     * @complexity Depends on T's copy/move assignment
     * @post Previous value (if any) is destroyed
     * @post has_value() == true, is<T>() == true
     *
     * @note Can change the stored type (unlike getStrictOrCompute)
     */
    template <typename T>
    Value& operator=(T&& v)
    {
        storage = std::forward<T>(v);
        return *this;
    }

    // ========================================================================
    // INTROSPECTION
    // ========================================================================

    /**
     * @brief Checks if a value is currently stored
     *
     * @return true if a value is stored, false otherwise
     *
     * @complexity Constant time
     * @noexcept
     */
    bool has_value() const noexcept
    {
        return storage.has_value();
    }

    /**
     * @brief Returns type information of the stored value
     *
     * @return std::type_info of the stored type, or typeid(void) if empty
     *
     * @complexity Constant time
     * @noexcept
     *
     * @note Returns typeid(void) when has_value() == false
     */
    const std::type_info& type() const noexcept
    {
        return storage.type();
    }

    /**
     * @brief Checks if the stored value is exactly of type T
     *
     * @tparam T Type to check against
     * @return true if stored value is of type T, false otherwise
     *
     * @complexity Constant time
     * @noexcept
     *
     * @note Returns false when has_value() == false
     * @note Performs exact type match; no inheritance or conversion considered
     */
    template <typename T>
    bool is() const noexcept
    {
        return storage.type() == typeid(T);
    }

    // ========================================================================
    // ACCESS METHODS
    // ========================================================================

    /**
     * @brief Attempts to retrieve a reference to the stored value
     *
     * @tparam T Expected type of the stored value
     * @return std::optional containing a const reference wrapper if type matches,
     *         std::nullopt otherwise
     *
     * @complexity Constant time
     * @noexcept
     *
     * @note Returns a reference wrapper; use .get() to access the actual reference
     * @example
     *     if (auto opt = v.try_get<int>()) {
     *         int value = opt->get();  // Access the int
     *     }
     */
    template <typename T>
    std::optional<std::reference_wrapper<const T>> try_get() const noexcept
    {
        if (!is<T>())
            return std::nullopt;

        return std::cref(std::any_cast<const T&>(storage));
    }

    /**
     * @brief Retrieves a mutable reference to the stored value
     *
     * @tparam T Expected type of the stored value
     * @return Mutable reference to the stored value
     *
     * @throws std::bad_any_cast if stored type is not T
     * @complexity Constant time
     *
     * @pre has_value() == true
     * @post No type change; stored value remains of type T
     *
     * @warning Modifying the returned reference modifies the stored value
     */
    template <typename T>
    T& get()
    {
        return std::any_cast<T&>(storage);
    }

    /**
     * @brief Retrieves a const reference to the stored value
     *
     * @tparam T Expected type of the stored value
     * @return Const reference to the stored value
     *
     * @throws std::bad_any_cast if stored type is not T
     * @complexity Constant time
     *
     * @pre has_value() == true
     */
    template <typename T>
    const T& get() const
    {
        return std::any_cast<const T&>(storage);
    }

    /**
     * @brief Implicit conversion to the stored type
     *
     * @tparam T Target type for conversion
     * @return Copy of the stored value converted to type T
     *
     * @throws std::bad_any_cast if stored type is not T
     * @complexity Depends on T's copy constructor
     *
     * @note Uses get<T>() internally; same exception guarantees
     * @note Creates a copy; use get() for reference access
     *
     * @example
     *     Value v = 42;
     *     int x = v;  // Equivalent to: int x = v.get<int>();
     */
    template <typename T>
    operator T() const
    {
        return get<T>();
    }

    /**
     * @brief Retrieves the stored value or a default if type mismatches
     *
     * @tparam T Expected type of the stored value
     * @param default_value Value to return if type mismatch or empty
     * @return Stored value if type matches T, otherwise default_value
     *
     * @complexity Constant time
     * @noexcept
     *
     * @note Always returns by value (copy)
     * @note Returns default_value when has_value() == false
     *
     * @example
     *     Value v = 3.14;
     *     double d = v.get_or<double>(0.0);  // 3.14
     *     int i = v.get_or<int>(0);          // 0 (type mismatch)
     */
    template <typename T>
    T get_or(const T& default_value) const
    {
        if (is<T>())
            return get<T>();
        return default_value;
    }

    /**
     * @brief Retrieves stored value or computes and stores it
     *
     * @tparam T Expected return type
     * @tparam F Callable returning T
     * @param func Function to compute value if not present or type mismatches
     * @return Stored value if type matches T, otherwise computed value
     *
     * @throws May propagate exceptions from func
     * @complexity O(1) if type matches, otherwise O(cost of func)
     *
     * @note If type mismatches, replaces stored value with computed value
     * @note Useful for lazy initialization with flexible type checking
     *
     * @example
     *     Value v;
     *     // First call computes and stores
     *     int x = v.getOrCompute<int>([] { return expensive_computation(); });
     *     // Second call returns cached value
     *     int y = v.getOrCompute<int>([] { return different_value(); });
     */
    template<typename T, typename F>
    T getOrCompute(F&& func)
    {
        if (is<T>())
            return get<T>();

        T computed = func();
        storage = computed;
        return computed;
    }

    /**
     * @brief Retrieves stored value or computes if empty (strict type enforcement)
     *
     * @tparam T Expected return type
     * @tparam F Callable returning T
     * @param func Function to compute value if empty
     * @return Stored value if type matches T, otherwise computed value (if empty)
     *
     * @throws std::bad_any_cast if has_value() && !is<T>()
     * @throws May propagate exceptions from func
     * @complexity O(1) if type matches, otherwise O(cost of func) if empty
     *
     * @note Only computes if Value is empty; enforces type stability otherwise
     * @note Useful for configuration parameters that shouldn't change type
     *
     * @example
     *     Value v;
     *     // Computes and stores
     *     int x = v.getStrictOrCompute<int>([] { return 4; });
     *     // Returns cached value
     *     int y = v.getStrictOrCompute<int>([] { return 8; });
     *     // Throws std::bad_any_cast (type mismatch)
     *     double z = v.getStrictOrCompute<double>([] { return 3.14; });
     */
    template<typename T, typename F>
    T getStrictOrCompute(F&& func)
    {
        if (!has_value()) {
            T computed = func();
            storage = computed;
            return computed;
        }

        return get<T>(); // throws on mismatch
    }

    // ========================================================================
    // MODIFIERS
    // ========================================================================

    /**
     * @brief Clears the stored value
     *
     * @complexity Constant time
     * @noexcept
     *
     * @post has_value() == false
     * @post type() == typeid(void)
     *
     * @note Destroys the stored value if present
     */
    void reset() noexcept
    {
        storage.reset();
    }
};

/**
 * @typedef DataStore
 * @brief String-keyed map of Value objects for parameter storage
 *
 * @details Provides a convenient, type-safe way to store configuration parameters,
 *          metadata, and intermediate results. Uses std::unordered_map for O(1)
 *          average lookup time with string keys.
 *
 * @note Not thread-safe; external synchronization required for concurrent access
 * @note Keys are case-sensitive
 *
 * @example
 *     DataStore config;
 *     config["time_limit"] = 10.0;
 *     config["threads"] = 4;
 *     config["verbose"] = true;
 *
 *     double limit = config["time_limit"].get<double>();
 *     int threads = config["threads"].get_or<int>(1);
 *
 *     // Safe iteration
 *     for (const auto& [key, value] : config) {
 *         if (value.is<double>()) {
 *             std::cout << key << ": " << value.get<double>() << "\n";
 *         }
 *     }
 *
 * @see Value
 * @see std::unordered_map
 */
using DataStore = std::unordered_map<std::string, Value>;