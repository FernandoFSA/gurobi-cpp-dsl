# Contributing to Gurobi C++ DSL

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## How to Contribute

### Reporting Issues

1. **Check existing issues** to avoid duplicates
2. **Use the issue template** when available
3. **Provide minimal reproducible examples** for bugs
4. **Include environment details**: OS, compiler, Gurobi version

### Submitting Changes

1. **Fork the repository** and create a feature branch
2. **Follow the coding style** (see below)
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

## Development Setup

### Prerequisites

- C++20 compatible compiler
- Gurobi 10.0+
- CMake 3.20+ (optional)

### Building

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/gurobi-cpp-dsl.git
cd gurobi-cpp-dsl

# Build with CMake
mkdir build && cd build
cmake -DDSL_BUILD_TESTS=ON ..
cmake --build .

# Run tests
ctest --output-on-failure
```

### Visual Studio

Open `MINI_DSL_GBR_CPP.vcxproj` and build the solution.

## Coding Style

### General Guidelines

- **C++20 features**: Use modern C++ (concepts, ranges, etc.) where appropriate
- **Header-only**: Keep the library header-only for easy integration
- **No external dependencies**: Except Gurobi and standard library

### Formatting

- **Indentation**: 4 spaces (no tabs)
- **Braces**: Allman style for functions, K&R for control structures
- **Line length**: ~100 characters soft limit
- **Naming**:
  - `camelCase` for functions and variables
  - `PascalCase` for types and classes
  - `UPPER_CASE` for macros and constants
  - Trailing underscore for member variables: `member_`

### Documentation

- **Doxygen-style comments** for public APIs
- **File headers** with overview, components, and usage
- **Inline examples** where helpful

### Example

```cpp
/**
 * @brief Brief description of the function
 *
 * @param param1 Description of parameter
 * @return Description of return value
 *
 * @throws std::exception When something goes wrong
 *
 * @example
 *     auto result = myFunction(42);
 */
template<typename T>
T myFunction(int param1) {
    if (param1 < 0) {
        throw std::invalid_argument("param1 must be non-negative");
    }
    
    T result{};
    // Implementation...
    return result;
}
```

## Testing

### Running Tests

```bash
# All tests
ctest --output-on-failure

# Specific test file
./dsl_tests "[expressions]"

# Verbose output
./dsl_tests -v
```

### Writing Tests

- Use **Catch2** (amalgamated version in `dsl/tests/`)
- Follow existing test organization (sections A-J, etc.)
- Use descriptive test names: `"A1: SumLambda::Sum1DIndexList"`
- Include `@covers` tags for traceability

```cpp
TEST_CASE("X1: Category::TestName", "[tag1][tag2]")
{
    // Arrange
    GRBModel model = makeModel();
    
    // Act
    auto result = dsl::someFunction(...);
    
    // Assert
    REQUIRE(result == expected);
}
```

## Pull Request Process

1. **Update CHANGELOG.md** with your changes under "Unreleased"
2. **Ensure tests pass** locally
3. **Request review** from maintainers
4. **Address feedback** promptly
5. **Squash commits** if requested

## Questions?

Open an issue with the "question" label or start a discussion.

---

Thank you for contributing! ??
