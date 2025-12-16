# DSL Examples

This directory contains 10 example programs demonstrating the Gurobi C++ DSL's capabilities for building optimization models with clean, mathematical notation. Examples are organized by increasing complexity.

## Examples Overview

| # | Example | Difficulty | Problem Type | Key DSL Features |
|---|---------|------------|--------------|------------------|
| 01 | [Simple LP](#01-simple-lp---production-planning) | Beginner | LP | `sum()`, `range()`, shadow prices |
| 02 | [Knapsack MIP](#02-knapsack-mip---item-selection) | Beginner | MIP | `ModelBuilder`, `fix()`/`unfix()`, what-if |
| 03 | [Transportation](#03-transportation---supply-chain) | Beginner | LP | 2D variables, dual values |
| 04 | [Assignment](#04-assignment---worker-task) | Intermediate | MIP | Filtered domains, sparse variables, IIS |
| 05 | [Facility Location](#05-facility-location---capacitated) | Intermediate | MIP | Multiple variable groups, linking constraints |
| 06 | [Diet Problem](#06-diet-problem---nutrition) | Intermediate | LP | Range constraints, sensitivity analysis |
| 07 | [Scheduling](#07-scheduling---job-shop) | Intermediate | MIP | Big-M constraints, warm start, Gantt chart |
| 08 | [Portfolio](#08-portfolio---mean-variance) | Advanced | QP | Quadratic objective, efficient frontier |
| 09 | [VRP](#09-vrp---vehicle-routing) | Advanced | MIP | Triple-indexed variables, MTZ subtour elimination |
| 10 | [Cutting Stock](#10-cutting-stock---pattern-based) | Advanced | IP | Pattern generation, objective comparison |

---

## 01. Simple LP - Production Planning

**File:** `01_simple_lp.cpp`  
**Difficulty:** Beginner  
**Problem Type:** Linear Programming (LP)

### Problem
A factory produces 3 products using 2 machines. Maximize profit subject to machine capacity constraints.

### Mathematical Model
```
max  sum_p profit[p] * x[p]
s.t. sum_p hours[m,p] * x[p] <= capacity[m]   for all m
     x[p] >= 0
```

### DSL Features
- `dsl::range(0, n)` - Create index domains
- `dsl::sum(domain, lambda)` - Mathematical summation
- `VariableFactory::add()` - Create variable arrays
- `ConstraintFactory::addIndexed()` - Domain-based constraints
- `maximize()` - Objective helper
- `slack()`, `dual()` - LP sensitivity (shadow prices)
- `statusString()`, `modelSummary()` - Diagnostics

---

## 02. Knapsack MIP - Item Selection

**File:** `02_knapsack_mip.cpp`  
**Difficulty:** Beginner  
**Problem Type:** Mixed-Integer Programming (MIP)

### Problem
Select items for a backpack to maximize value while respecting capacity. Some item pairs are incompatible.

### Mathematical Model
```
max  sum_i value[i] * x[i]
s.t. sum_i weight[i] * x[i] <= capacity
     x[i] + x[j] <= 1              for all (i,j) in Conflicts
     x[i] in {0,1}
```

### DSL Features
- `ModelBuilder<VarEnum, ConEnum>` - Template method pattern
- `DECLARE_ENUM_WITH_COUNT` - Type-safe enum keys
- `applyPreset(Preset::Fast)` - Parameter presets
- `mipGapLimit()` - MIP gap settings
- `DataStore` - Metadata storage
- `fix()`, `unfix()` - What-if analysis
- `hasSolution()`, `objVal()`, `mipGap()` - Solution diagnostics

---

## 03. Transportation - Supply Chain

**File:** `03_transportation.cpp`  
**Difficulty:** Beginner  
**Problem Type:** Linear Programming (LP)

### Problem
Ship goods from warehouses to stores minimizing transportation cost while meeting demands.

### Mathematical Model
```
min  sum_{w,s} cost[w,s] * x[w,s]
s.t. sum_s x[w,s] <= supply[w]        for all w  (Supply)
     sum_w x[w,s] >= demand[s]        for all s  (Demand)
     x[w,s] >= 0
```

### DSL Features
- Two-dimensional variables `X(w, s)`
- `W * S` - Cartesian product domains
- Multiple constraint families (Supply, Demand)
- `minimize()` - Objective helper
- Dual values interpretation for marginal costs

---

## 04. Assignment - Worker-Task

**File:** `04_assignment.cpp`  
**Difficulty:** Intermediate  
**Problem Type:** Mixed-Integer Programming (MIP)

### Problem
Assign workers to tasks minimizing cost. Workers have qualifications for specific tasks.

### Mathematical Model
```
Let Q = {(w,t) : qualified[w,t] = 1}

min  sum_{(w,t) in Q} cost[w,t] * x[w,t]
s.t. sum_{w:(w,t) in Q} x[w,t] = 1      for all t  (Task coverage)
     sum_{t:(w,t) in Q} x[w,t] <= 1     for all w  (Worker limit)
     x[w,t] in {0,1}
```

### DSL Features
- `(W * T) | dsl::filter(predicate)` - Filtered Cartesian domains
- `IndexedVariableSet` - Sparse variable structures
- `IndexedConstraintSet` - Domain-based constraints
- `try_get()` - Safe sparse access
- `computeIIS()` - Infeasibility analysis

---

## 05. Facility Location - Capacitated

**File:** `05_facility_location.cpp`  
**Difficulty:** Intermediate  
**Problem Type:** Mixed-Integer Programming (MIP)

### Problem
Decide which facilities to open and how to assign customers to minimize fixed + transportation costs.

### Mathematical Model
```
min  sum_f fixedCost[f]*y[f] + sum_{f,c} transCost[f,c]*demand[c]*x[f,c]
s.t. sum_f x[f,c] = 1                              for all c  (Demand)
     sum_c demand[c]*x[f,c] <= capacity[f]*y[f]    for all f  (Capacity)
     x[f,c] <= y[f]                                for all f,c (Linking)
     y[f] in {0,1}, x[f,c] in [0,1]
```

### DSL Features
- Multiple variable groups (binary `y[f]`, continuous `x[f,c]`)
- Linking constraints `x <= y`
- Capacity with binary variables
- `timeLimit()` - Solve time management

---

## 06. Diet Problem - Nutrition

**File:** `06_diet_problem.cpp`  
**Difficulty:** Intermediate  
**Problem Type:** Linear Programming (LP)

### Problem
Design a minimum-cost diet meeting nutritional requirements with min/max bounds on nutrients.

### Mathematical Model
```
min  sum_f cost[f] * x[f]
s.t. sum_f nutrients[f,n] * x[f] >= minIntake[n]    for all n
     sum_f nutrients[f,n] * x[f] <= maxIntake[n]    for all n
     x[f] in [0, maxServings[f]]
```

### DSL Features
- Range constraints (min <= expr <= max)
- Multiple constraint types (min/max bounds)
- `setUB()` - Bound modification
- Dual values interpretation for nutrition constraints

---

## 07. Scheduling - Job Shop

**File:** `07_scheduling.cpp`  
**Difficulty:** Intermediate  
**Problem Type:** Mixed-Integer Programming (MIP)

### Problem
Schedule jobs on a single machine to minimize total weighted tardiness.

### Mathematical Model
```
min  sum_j penalty[j] * tardy[j]
s.t. tardy[j] >= start[j] + processing[j] - due[j]
     start[j] >= start[i] + processing[i] - M*(1-before[i,j])   for all i<j
     start[i] >= start[j] + processing[j] - M*before[i,j]       for all i<j
     before[i,j] in {0,1}
```

### DSL Features
- Big-M constraints for disjunctive scheduling
- Auxiliary variables (tardiness linearization)
- `setStart()` - Warm-start hints (EDD heuristic)
- Filtered pair domain `i < j`
- Visual Gantt-style output

---

## 08. Portfolio - Mean-Variance

**File:** `08_portfolio.cpp`  
**Difficulty:** Advanced  
**Problem Type:** Quadratic Programming (QP)

### Problem
Allocate capital to minimize portfolio variance while meeting a minimum return target.

### Mathematical Model
```
min  sum_{a,b} sigma[a,b] * x[a] * x[b]    (Variance)
s.t. sum_a x[a] = 1                         (Budget)
     sum_a mu[a] * x[a] >= minReturn        (Return target)
     x[a] in [0, maxAllocation]
```

### DSL Features
- `dsl::quadSum()` - Quadratic expression construction
- Covariance matrix handling
- Efficient frontier computation
- Risk-return trade-off analysis
- Sharpe ratio optimization

---

## 09. VRP - Vehicle Routing

**File:** `09_vrp_basic.cpp`  
**Difficulty:** Advanced  
**Problem Type:** Mixed-Integer Programming (MIP)

### Problem
Route vehicles from depot to customers minimizing total distance with capacity constraints.

### Mathematical Model
```
min  sum_{i,j,k} dist[i,j] * x[i,j,k]
s.t. sum_{j,k} x[i,j,k] = 1                              for all i in C  (Visit)
     sum_i x[i,j,k] = sum_i x[j,i,k]                     for all j,k  (Flow)
     sum_j x[0,j,k] <= 1                                 for all k    (Depot)
     u[j,k] >= u[i,k] + demand[j] - M*(1-x[i,j,k])       MTZ
     x[i,j,k] in {0,1}
```

### DSL Features
- Triple-indexed variables `x[i,j,k]`
- MTZ subtour elimination constraints
- Complex filtered domains
- Route extraction from solution
- Distance matrix computation

---

## 10. Cutting Stock - Pattern-Based

**File:** `10_cutting_stock.cpp`  
**Difficulty:** Advanced  
**Problem Type:** Integer Programming (IP)

### Problem
Cut stock rolls to fulfill orders minimizing rolls used or total waste.

### Mathematical Model
```
min  sum_p x[p]   or   min sum_p waste[p] * x[p]
s.t. sum_p pattern[p,i] * x[p] >= demand[i]    for all i
     x[p] in Z+
```

### DSL Features
- Pattern generation (pre-solve enumeration)
- `GRB_INTEGER` variables
- Objective switching (rolls vs waste)
- Solution interpretation (physical cuts)
- Model modification and re-solve

---

## Building Examples

### Prerequisites

- C++20 compatible compiler (MSVC 2019+, GCC 10+, Clang 12+)
- Gurobi Optimizer 10.0+ installed
- `GUROBI_HOME` environment variable set

### Visual Studio (Windows)

The examples are included in the project but **excluded from build** by default.

To build an example:
1. Right-click on the example file in Solution Explorer
2. Select "Properties"
3. Set "Excluded from Build" to "No"
4. Temporarily exclude other source files with `main()`
5. Build

### CMake (Cross-platform)

```cmake
cmake_minimum_required(VERSION 3.20)
project(dsl_examples)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Gurobi
find_path(GUROBI_INCLUDE_DIR gurobi_c++.h
    HINTS $ENV{GUROBI_HOME}/include)
find_library(GUROBI_LIBRARY 
    NAMES gurobi100 gurobi110 gurobi120
    HINTS $ENV{GUROBI_HOME}/lib)
find_library(GUROBI_CXX_LIBRARY 
    NAMES gurobi_c++ gurobi_c++md2019
    HINTS $ENV{GUROBI_HOME}/lib)

# Include DSL headers
include_directories(../include)
include_directories(${GUROBI_INCLUDE_DIR})

# Build each example
foreach(EXAMPLE 
    01_simple_lp 02_knapsack_mip 03_transportation 04_assignment
    05_facility_location 06_diet_problem 07_scheduling
    08_portfolio 09_vrp_basic 10_cutting_stock)
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})
endforeach()
```

---

## DSL Quick Reference

### Index Domains
```cpp
auto I = dsl::range(0, 10);           // Materialized: {0,1,...,9}
auto R = dsl::range_view(0, 100);     // Lazy range
auto P = I * J;                       // Cartesian product
auto F = P | dsl::filter(predicate);  // Filtered domain
```

### Variables
```cpp
// Dense array
auto X = dsl::VariableFactory::add(model, GRB_BINARY, 0, 1, "X", m, n);

// Sparse (domain-based)
auto Y = dsl::VariableFactory::addIndexed(model, GRB_CONTINUOUS, 0, 1, "Y", filteredDomain);
```

### Expressions
```cpp
dsl::sum(I, [&](int i) { return c[i] * X(i); });        // sum_i c[i]*x[i]
dsl::sum(I * J, [&](int i, int j) { return X(i,j); });  // sum_{i,j} x[i,j]
dsl::sum(X);                                             // Sum all variables

// Quadratic expressions (for QP objectives)
dsl::quadSum(I, [&](int i) { return X(i) * X(i); });    // sum_i x[i]^2
dsl::quadSum(I * J, [&](int i, int j) {                 // sum_{i,j} sigma[i,j]*x[i]*x[j]
    return sigma[i][j] * X(i) * X(j);
});
```

### Constraints
```cpp
// Domain-based creation
auto cons = dsl::ConstraintFactory::addIndexed(model, "name", domain,
    [&](int i, int j) { return expr <= rhs; });
```

### Solution Access
```cpp
double val = dsl::value(x);                    // Single variable
auto vals = dsl::values(X);                    // All values as vector
for (auto& [idx, v] : dsl::valuesWithIndex(X)) // With indices
```

### ModelBuilder Pattern
```cpp
class MyModel : public dsl::ModelBuilder<VarEnum, ConEnum> {
protected:
    void addParameters() override { quiet(); applyPreset(Preset::Fast); }
    void addVariables() override { /* create variables */ }
    void addConstraints() override { /* create constraints */ }
    void addObjective() override { maximize(expr); }
    void afterOptimize() override { /* post-processing */ }
};
```

---

## Learning Path

1. **Start with LP basics:** Examples 01, 03, 06
2. **Learn MIP concepts:** Examples 02, 04
3. **Explore advanced MIP:** Examples 05, 07
4. **Master advanced topics:** Examples 08 (QP), 09 (VRP), 10 (patterns)

Each example builds on concepts from earlier examples. The comments in each file explain both the optimization model and the DSL features used.
