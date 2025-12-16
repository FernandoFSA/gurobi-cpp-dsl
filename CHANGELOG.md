# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Fixed
- Nothing yet

---

## [1.0.0] - 2024-XX-XX

### Added

#### Core DSL
- `ModelBuilder<VarEnum, ConEnum>` template for structured model construction
- `VariableFactory` for creating dense (`VariableGroup`) and sparse (`IndexedVariableSet`) variables
- `ConstraintFactory` for domain-based constraint generation
- `DataStore` for flexible metadata storage with type-safe access

#### Indexing System
- `IndexList` for materialized index sets
- `RangeView` for lazy range iteration
- Cartesian product operator (`I * J`)
- Filter operator (`domain | dsl::filter(predicate)`)
- Support for 1D, 2D, and 3D index tuples

#### Expressions
- `sum(domain, lambda)` for linear expression construction
- `sum(VariableGroup)` and `sum(IndexedVariableSet)` overloads
- `quadSum(domain, lambda)` for quadratic expressions (QP support)

#### Solution Access
- `value()`, `values()`, `valuesWithIndex()` for variable values
- `dual()`, `slack()` for constraint analysis
- `reducedCost()` for variable analysis

#### Diagnostics
- `computeIIS()` for infeasibility analysis
- `statusString()` for human-readable status
- `modelSummary()` for model statistics
- `Preset::Fast`, `Preset::Precise`, `Preset::Balanced` parameter presets

#### Callbacks
- `CallbackAdapter` base class with virtual hooks
- Lazy constraint support (`addLazy()`)
- User cut support (`addCut()`)
- Heuristic solution injection (`setSolution()`)
- Progress logging and termination control

#### Examples
- 01: Simple LP (production planning)
- 02: Knapsack MIP (item selection)
- 03: Transportation (supply chain)
- 04: Assignment (worker-task with IIS)
- 05: Facility Location (capacitated)
- 06: Diet Problem (nutrition optimization)
- 07: Scheduling (single machine with tardiness)
- 08: Portfolio (Markowitz mean-variance QP)
- 09: VRP (vehicle routing with MTZ)
- 10: Cutting Stock (pattern-based)

#### Testing
- Comprehensive Catch2 test suite
- Tests for all major components
- Edge case and error handling coverage

#### Documentation
- Detailed header file documentation
- Examples README with learning path
- Project README with quick start

### Technical Requirements
- C++20 standard
- Gurobi 10.0+
- Header-only library design

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | TBD | Initial public release |

[Unreleased]: https://github.com/FernandoFSA/MINI_DSL_GBR_CPP/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/FernandoFSA/MINI_DSL_GBR_CPP/releases/tag/v1.0.0
