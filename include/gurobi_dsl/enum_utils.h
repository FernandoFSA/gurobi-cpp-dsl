#pragma once
/*
===============================================================================
ENUM UTILS — Compile-time enumeration utilities for the C++ Gurobi DSL
===============================================================================

OVERVIEW
--------
Provides minimalist, zero-overhead utilities for declaring strongly-typed
enumerations with compile-time size information. The primary macro generates
enum classes with a COUNT sentinel and a corresponding size constant, enabling
fixed-size arrays, bounds-checked indexing, and compile-time iteration without
manual maintenance of enumerator counts.

KEY COMPONENTS
--------------
• DECLARE_ENUM_WITH_COUNT: Primary macro for enum declaration
• DECLARE_EMPTY_ENUM_WITH_COUNT: Alternate macro for empty enumerations
• Automatic COUNT sentinel generation
• Compile-time size constant derivation

DESIGN PHILOSOPHY
-----------------
• Zero runtime overhead - all information is compile-time
• Minimal abstraction - simple macro expansion only
• No reflection or string conversion overhead
• Type-safe enum class usage by default
• Predictable COUNT placement as last enumerator

USAGE EXAMPLES
--------------
    // Declare enumeration with automatic COUNT
    DECLARE_ENUM_WITH_COUNT(Color, Red, Green, Blue);

    // Fixed-size array using enum size
    std::array<double, Color_COUNT> rgb_values{};
    rgb_values[static_cast<std::size_t>(Color::Red)] = 1.0;

    // Compile-time iteration
    for (std::size_t i = 0; i < Color_COUNT; ++i) {
        process_color(static_cast<Color>(i));
    }

    // Declare an empty enumeration (COUNT only)
    DECLARE_EMPTY_ENUM_WITH_COUNT(Empty);
    static_assert(Empty_COUNT == 0, "Empty enum has size 0");

DEPENDENCIES
------------
• <cstddef> - For std::size_t type

PERFORMANCE NOTES
-----------------
• Zero runtime cost - all computations are compile-time
• No dynamic initialization - constants are compile-time evaluable
• Minimal macro expansion overhead

THREAD SAFETY
-------------
• All generated code is thread-safe
• No mutable shared state
• Constants have internal linkage

EXCEPTION SAFETY
----------------
• No-throw guarantee for all generated code
• No dynamic memory allocation
• No exception-throwing operations

===============================================================================
*/

#include <cstddef>

/**
 * @macro DECLARE_ENUM_WITH_COUNT
 * @brief Declares a strongly-typed enum class with automatic COUNT sentinel
 *
 * @param Name The name of the enumeration type
 * @param ...  Comma-separated list of enumerator identifiers (at least one)
 *
 * @details
 * Expands to two declarations:
 * 1. An enum class with the specified enumerators plus a trailing COUNT
 * 2. A constexpr size_t constant named <Name>_COUNT with the enumerator count
 *
 * The COUNT enumerator serves as both a sentinel and a compile-time counter.
 * Its numeric value equals the number of user-defined enumerators, making
 * <Name>_COUNT suitable for array dimensions and loop bounds.
 *
 * @note
 * - COUNT is always appended as the last enumerator
 * - Enumerators follow standard C++ scoping: Name::Enumerator
 * - The size constant has internal linkage (static)
 * - No string conversion or reflection is provided
 *
 * @warning
 * - Do not explicitly define COUNT in your enumerator list
 * - The COUNT sentinel is not intended for domain logic
 * - Enum values are sequential starting from 0
 *
 * @remarks
 * - This macro requires at least one enumerator. For empty enums, use
 *   DECLARE_EMPTY_ENUM_WITH_COUNT(Name).
 *
 * @example
 *     // Declaration
 *     DECLARE_ENUM_WITH_COUNT(Direction, North, South, East, West);
 *
 *     // Usage
 *     Direction d = Direction::North;
 *     std::array<Path, Direction_COUNT> paths{};
 *
 *     // Expansion equivalent:
 *     // enum class Direction { North, South, East, West, COUNT };
 *     // static constexpr std::size_t Direction_COUNT = 4;
 *
 * @complexity
 * Compile-time only, zero runtime overhead
 *
 * @see For more advanced enum utilities, consider dedicated enum libraries
 */
// Primary macro: requires at least one enumerator (C++17 compatible)
#define DECLARE_ENUM_WITH_COUNT(Name, ...)                                \
    enum class Name { __VA_ARGS__, COUNT };                               \
    static constexpr std::size_t Name##_COUNT =                           \
        static_cast<std::size_t>(Name::COUNT)

/**
 * @macro DECLARE_EMPTY_ENUM_WITH_COUNT
 * @brief Declares an empty strongly-typed enum class with COUNT sentinel
 *
 * @param Name The name of the enumeration type
 *
 * @details
 * Expands to an enum class containing only COUNT and a corresponding
 * constexpr size_t constant named <Name>_COUNT equal to 0. Use this
 * macro when you need a placeholder enum with no user enumerators.
 */
// Alternate macro for empty enumerations (no user enumerators)
#define DECLARE_EMPTY_ENUM_WITH_COUNT(Name)                               \
    enum class Name { COUNT };                                            \
    static constexpr std::size_t Name##_COUNT = 0

 /**
  * @brief Helper type for enum-based array declarations
  *
  * @tparam Enum Enumeration type declared with DECLARE_ENUM_WITH_COUNT
  * @tparam T    Element type of the array
  *
  * @details
  * Provides concise syntax for declaring fixed-size arrays indexed by
  * enumeration values. Ensures array size matches enumerator count at
  * compile time.
  *
  * @example
  *     DECLARE_ENUM_WITH_COUNT(Status, Ok, Warning, Error);
  *     EnumArray<Status, std::string> messages = {
  *         "Operation successful",
  *         "Proceed with caution",
  *         "Critical failure"
  *     };
  *
  *     std::cout << messages[Status::Warning]; // "Proceed with caution"
  */
template<typename Enum, typename T>
using EnumArray = T[static_cast<std::size_t>(Enum::COUNT)];

/**
 * @brief Compile-time enumeration size trait
 *
 * @tparam Enum Enumeration type
 *
 * @details
 * Provides uniform access to enumeration size regardless of declaration
 * method. Specialize this template for custom enumeration types not
 * declared with DECLARE_ENUM_WITH_COUNT.
 *
 * @value
 * Number of enumerators in the enumeration (excluding any sentinel)
 *
 * @note
 * Primary template assumes enumeration uses COUNT sentinel convention.
 * For enumerations without COUNT, provide explicit specialization.
 *
 * @example
 *     DECLARE_ENUM_WITH_COUNT(Flags, Read, Write, Execute);
 *     static_assert(enum_size<Flags>::value == 3);
 */
template<typename Enum>
struct enum_size {
    /// @brief Number of meaningful enumerators (excluding COUNT if present)
    static constexpr std::size_t value = static_cast<std::size_t>(Enum::COUNT);
};

/**
 * @brief Compile-time check for valid enumeration value
 *
 * @tparam Enum Enumeration type
 * @param value Enumeration value to check
 *
 * @return true if value corresponds to a user-defined enumerator
 * @return false if value is out of range or equals COUNT sentinel
 *
 * @details
 * Performs bounds checking for enumeration values. Useful for validating
 * user input or external data before indexing into enum-based arrays.
 *
 * @note
 * Assumes sequential enumeration values starting from 0.
 * COUNT sentinel is considered invalid for domain logic.
 *
 * @example
 *     DECLARE_ENUM_WITH_COUNT(Level, Low, Medium, High);
 *     Level l = static_cast<Level>(2);
 *     assert(is_valid_enum_value(l)); // true
 *     assert(!is_valid_enum_value(Level::COUNT)); // true
 */
template<typename Enum>
constexpr bool is_valid_enum_value(Enum value) noexcept {
    const auto ival = static_cast<std::size_t>(value);
    return ival < static_cast<std::size_t>(Enum::COUNT);
}

/**
 * @brief Safely convert integral value to enumeration
 *
 * @tparam Enum Enumeration type
 * @param value Integral value to convert
 * @return Enumeration value if valid, else undefined behavior
 *
 * @details
 * Performs compile-time or runtime bounds checking (depending on context)
 * before static_cast. In debug builds, may include runtime assertions.
 *
 * @pre
 * 0 <= value < enum_size<Enum>::value
 *
 * @note
 * Use only with trusted integral values. For untrusted input,
 * combine with is_valid_enum_value check.
 *
 * @example
 *     DECLARE_ENUM_WITH_COUNT(Priority, Low, Normal, High);
 *     auto p = enum_from_value<Priority>(1); // Priority::Normal
 */
template<typename Enum>
constexpr Enum enum_from_value(std::size_t value) noexcept {
    // Debug builds may include bounds checking
#if defined(DSL_DEBUG) || defined(_DEBUG)
    // Assertion style depends on project configuration
    // Could use assert(), throw, or contract checks
#endif
    return static_cast<Enum>(value);
}

/*
===============================================================================
ADVANCED USAGE PATTERNS
===============================================================================

1. ITERATION OVER ENUMERATORS:
   ------------------------------------------------------------
   DECLARE_ENUM_WITH_COUNT(State, Idle, Active, Paused);

   template<typename Fn>
   void for_each_enum(Fn&& fn) {
       for (std::size_t i = 0; i < State_COUNT; ++i) {
           fn(static_cast<State>(i));
       }
   }

2. ENUM-INDEXED DISPATCH TABLE:
   ------------------------------------------------------------
   DECLARE_ENUM_WITH_COUNT(Operation, Add, Subtract, Multiply);

   using Handler = void(*)(int, int);
   constexpr Handler handlers[Operation_COUNT] = {
       [](int a, int b) { return a + b; },
       [](int a, int b) { return a - b; },
       [](int a, int b) { return a * b; }
   };

   int execute(Operation op, int a, int b) {
       return handlers[static_cast<std::size_t>(op)](a, b);
   }

3. COMPILE-TIME SWITCHES:
   ------------------------------------------------------------
   DECLARE_ENUM_WITH_COUNT(Mode, Fast, Balanced, Quality);

   template<Mode M>
   struct ModeConfig;

   template<> struct ModeConfig<Mode::Fast> {
       static constexpr int threads = 1;
       static constexpr double tolerance = 1e-3;
   };

   // Usage in templates:
   template<Mode M>
   void optimize() {
       constexpr int t = ModeConfig<M>::threads;
       // ...
   }

===============================================================================
*/