# Gurobi C++ DSL

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gurobi](https://img.shields.io/badge/Gurobi-10.0%2B-green.svg)](https://www.gurobi.com/)

A modern C++20 domain-specific language (DSL) for building mathematical optimization models with [Gurobi](https://www.gurobi.com/). Write clean, expressive code that mirrors mathematical notation.

## Features

- **Mathematical Notation** - Write `sum(I, [&](int i) { return c[i] * x(i); })` instead of manual loops
- **Type-Safe Indexing** - Cartesian products, filters, and multi-dimensional domains
- **ModelBuilder Pattern** - Structured model construction with lifecycle hooks
- **Dense and Sparse Variables** - `VariableGroup` for dense arrays, `IndexedVariableSet` for sparse
- **Comprehensive Diagnostics** - IIS computation, solution analysis, model summaries
- **QP Support** - `quadSum()` for quadratic programming objectives

## Quick Start

```cpp
#include <gurobi_dsl/dsl.h>

// Define enum keys for type-safe variable/constraint access
DECLARE_ENUM_WITH_COUNT(Vars, X);
DECLARE_ENUM_WITH_COUNT(Cons, Capacity);

class ProductionModel : public dsl::ModelBuilder<Vars, Cons> {
    std::vector<double> profit_ = {10, 15, 12};
    std::vector<double> hours_ = {2, 3, 2.5};
    double capacity_ = 100;

protected:
    void addVariables() override {
        auto X = dsl::VariableFactory::add(
            model(), GRB_CONTINUOUS, 0, GRB_INFINITY, "x", 3
        );
        variables().set(Vars::X, std::move(X));
    }

    void addConstraints() override {
        auto& X = variables().get(Vars::X);
        auto P = dsl::range(0, 3);
        
        dsl::ConstraintFactory::add(model(), "capacity",
            [&](auto) { 
                return dsl::sum(P, [&](int p) { 
                    return hours_[p] * X.at(p); 
                }) <= capacity_; 
            }
        );
    }

    void addObjective() override {
        auto& X = variables().get(Vars::X);
        auto P = dsl::range(0, 3);
        
        maximize(dsl::sum(P, [&](int p) { 
            return profit_[p] * X.at(p); 
        }));
    }
};

int main() {
    ProductionModel model;
    model.optimize();
    
    if (model.hasSolution()) {
        std::cout << "Optimal profit: $" << model.objVal() << "\n";
    }
    return 0;
}
```

## Installation

### Prerequisites

- C++20 compatible compiler (MSVC 2019+, GCC 10+, Clang 12+)
- [Gurobi Optimizer](https://www.gurobi.com/) 10.0 or later
- `GUROBI_HOME` environment variable set

### Header-Only

Copy the `include/gurobi_dsl/` directory to your project:

```cpp
#include <gurobi_dsl/dsl.h>
```

### CMake

```cmake
add_subdirectory(gurobi-cpp-dsl)
target_link_libraries(your_target PRIVATE gurobi::dsl)
```

## Core Concepts

### Index Domains

```cpp
auto I = dsl::range(0, 10);           // Materialized {0,1,...,9}
auto R = dsl::range_view(0, 100);     // Lazy (no allocation)
auto P = I * J;                       // Cartesian product
auto F = P | dsl::filter([](int i, int j) { return i < j; });
```

### Variables

```cpp
// Dense arrays (rectangular)
auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "x", m, n);

// Sparse (filtered domains)
auto Y = dsl::VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 1, "y", 
    (I * J) | dsl::filter([](int i, int j) { return qualified[i][j]; }));
```

### Expressions

```cpp
// Linear: sum over i of c[i]*x[i]
dsl::sum(I, [&](int i) { return c[i] * X(i); });

// Quadratic: sum over i,j of sigma[i][j]*x[i]*x[j] (for QP)
dsl::quadSum(I * J, [&](int i, int j) { 
    return sigma[i][j] * X(i) * X(j); 
});
```

### Constraints

```cpp
// Single constraint
auto con = dsl::ConstraintFactory::add(model, "budget",
    [&](auto) { return dsl::sum(I, [&](int i) { return X(i); }) <= budget; });

// Dense (indexed over rectangular domain)
auto cons = dsl::ConstraintFactory::addIndexed(model, "demand", Customers,
    [&](int c) { 
        return dsl::sum(Facilities, [&](int f) { return X(f, c); }) >= demand[c]; 
    });

// Sparse (indexed over filtered domain)
auto sparse = dsl::ConstraintFactory::addIndexed(model, "arc", filteredArcs,
    [&](int i, int j) { return X(i, j) <= capacity[i][j]; });
```

### Solution Access

```cpp
double val = dsl::value(x);                     // Single variable
auto vals = dsl::values(X);                     // All as vector
double shadow = dsl::dual(constraint);          // Shadow price
double slack = dsl::slack(constraint);          // Constraint slack
```

## Examples

See the [`examples/`](examples/) directory for complete examples:

| # | Example | Type | Key Features |
|---|---------|------|--------------|
| 01 | Simple LP | LP | `sum()`, shadow prices |
| 02 | Knapsack | MIP | `ModelBuilder`, what-if analysis |
| 03 | Transportation | LP | 2D variables, dual values |
| 04 | Assignment | MIP | Filtered domains, IIS |
| 05 | Facility Location | MIP | Multiple variable groups |
| 06 | Diet Problem | LP | Range constraints |
| 07 | Scheduling | MIP | Big-M, warm start |
| 08 | Portfolio | QP | `quadSum()`, efficient frontier |
| 09 | VRP | MIP | Triple-indexed, MTZ subtours |
| 10 | Cutting Stock | IP | Pattern generation |

## Project Structure

```
gurobi-cpp-dsl/
+-- include/
|   +-- gurobi_dsl/        # Header-only library
|       +-- dsl.h          # Main include
|       +-- model_builder.h
|       +-- variables.h
|       +-- ...
+-- examples/              # 10 progressive examples
+-- tests/                 # Catch2 test suite
+-- CMakeLists.txt
+-- LICENSE
+-- README.md
```

## Running Tests

```bash
# Build with CMake
mkdir build && cd build
cmake -DDSL_BUILD_TESTS=ON ..
cmake --build .

# Run tests
ctest --output-on-failure
```

## Requirements

| Component | Minimum Version |
|-----------|-----------------|
| C++ Standard | C++20 |
| Gurobi | 10.0 |
| MSVC | 2019 (v142) |
| GCC | 10.0 |
| Clang | 12.0 |
| CMake | 3.20 (optional) |

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Gurobi Optimization](https://www.gurobi.com/) for the underlying solver
- The C++ community for modern language features that make this DSL possible

---

**Note:** This is an independent project and is not affiliated with or endorsed by Gurobi Optimization, LLC.

