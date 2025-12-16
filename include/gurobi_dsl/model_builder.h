#pragma once
/*
===============================================================================
MODEL BUILDER — High-level orchestration for constructing optimization models
===============================================================================

Overview
--------
ModelBuilder is the central abstraction that coordinates:

    * Environment creation (GRBEnv)
    * Model creation (GRBModel)
    * Variable creation (via VariableTable)
    * Constraint creation (via ConstraintTable)
    * Parameter assignment
    * Objective construction
    * Optimization lifecycle management

It implements the "template method" pattern:

    initialize()
    optimize() {
        addVariables();
        addConstraints();
        addParameters();
        addObjective();
        beforeOptimize();
        model.optimize();
        afterOptimize();
    }

The goal is to provide a clean and reusable architecture for model-building
without mixing environment concerns, parameter setup, and model logic.
Derived builders override the relevant virtual hooks.

Key Features
------------
1. Lazy initialization:
       - Constructor performs no heavy solver actions.
       - initialize() is called automatically when optimize() is executed.
       - No environment or model objects are created until needed.

2. External model support:
       - A ModelBuilder may be constructed using a pre-existing GRBModel.
       - In this case no environment is created and lifecycle control belongs
         to the user.

3. Fully generic:
       - ModelBuilder is templated on two enums:
           VarEnum: variable registry keys
           ConEnum: constraint registry keys

         allowing strongly typed sets of variable and constraint groups.

4. Strong separation of responsibilities:
       - Variables and constraints are stored in VariableTable and ConstraintTable
         (see variables.h and constraints.h).
       - DataStore provides a key-value storage for auxiliary information.

5. Safe, deterministic behavior:
       - initialize() only runs once.
       - optimize() always results in a ready model regardless of whether the
         user manually invoked initialize().

6. Solution diagnostics:
       - Convenient accessors for post-optimization queries: status(), objVal(),
         mipGap(), runtime(), isOptimal(), hasSolution(), etc.
       - Eliminates need to remember Gurobi attribute macros.

7. Objective helpers:
       - minimize(expr) and maximize(expr) for setting the objective.
       - Eliminates need to remember GRB_MINIMIZE/GRB_MAXIMIZE macros.

8. Parameter presets and convenience setters:
       - Named methods: timeLimit(), mipGapLimit(), threads(), quiet(), etc.
       - Presets: applyPreset(Preset::Fast), Preset::Accurate, etc.
       - All tracked in store() for diagnostics (store()["param:TimeLimit"], etc.)

Typical Usage
-------------
    DECLARE_ENUM_WITH_COUNT(Vars, X, Y);
    DECLARE_ENUM_WITH_COUNT(Cons, Cap, Flow);

    class MyBuilder : public dsl::ModelBuilder<Vars, Cons> {
        void addVariables() override {
            auto& m = model();
            variables().set(Vars::X,
                VariableFactory::add(m, GRB_CONTINUOUS, 0, 10, "X", 5));
        }
        void addConstraints() override {
            // use constraints().set(...)
        }
        void addObjective() override {
            // model().setObjective(...)
        }
    };

    MyBuilder b;
    b.optimize();
    GRBModel& m = b.model();

Design Notes
------------
* No solver operations are executed in the constructor.
* configureEnvironment() is called exactly once when we own the environment.
* setParam(p, v) provides a convenient interface for GRBModel::set().
* Internal logic ensures that optimize() can always be safely called.
* The derived builder may treat ModelBuilder as an abstract workflow engine.

===============================================================================
*/

#include <memory>
#include "gurobi_c++.h"

#include "variables.h"
#include "constraints.h"
#include "data_store.h"

namespace dsl {

    /*
    ===============================================================================
    MODEL BUILDER TEMPLATE
    ===============================================================================
    */
    template <typename VarEnum, typename ConEnum>
    class ModelBuilder {
    public:
        using VarTable = VariableTable<VarEnum>;
        using ConTable = ConstraintTable<ConEnum>;

    private:
        // If owned, these hold the solver environment and model.
        std::unique_ptr<GRBEnv>   env_;
        std::unique_ptr<GRBModel> owned_model_;

        // If provided by user, we do not own the model or its environment.
        GRBModel* external_model_ = nullptr;

        bool initialized_ = false;

    protected:
        // Variable registry
        VarTable vars_;

        // Constraint registry
        ConTable cons_;

        // Arbitrary key-value store for metadata, parameters, etc.
        DataStore store_;

    public:
        // -------------------------------------------------------------------------
        // Constructors
        // -------------------------------------------------------------------------

        /// @brief Default constructor. No environment or model created yet.
        ModelBuilder() = default;

        /**
         * @brief Construct with external model.
         *
         * No environment or model will be created internally.
         * User is responsible for lifetime of externalModel and its environment.
         */
        explicit ModelBuilder(GRBModel& externalModel)
            : external_model_(&externalModel),
            initialized_(true)   // already initialized (nothing to do)
        {
        }

        virtual ~ModelBuilder() = default;

        // -------------------------------------------------------------------------
        // Initialization
        // -------------------------------------------------------------------------
        /**
         * @brief Initialize environment and model if not already initialized.
         *
         * Behavior:
         *   * If an external model was provided, skip all initialization.
         *   * Otherwise:
         *         - Create GRBEnv (deferred start)
         *         - Allow derived classes to configure environment parameters
         *         - Call env->start()
         *         - Create GRBModel owned internally
         */
        void initialize()
        {
            if (initialized_)
                return;

            if (!external_model_) {
                env_ = std::make_unique<GRBEnv>(true);  // defer license check and load
                configureEnvironment(*env_);            // derived hook
                env_->start();
                owned_model_ = std::make_unique<GRBModel>(*env_);
            }

            initialized_ = true;
        }

        // -------------------------------------------------------------------------
        // Accessors
        // -------------------------------------------------------------------------
        /**
         * @brief Get mutable model reference, auto-initializing if needed.
         */
        GRBModel& model()
        {
            if (!initialized_)
                initialize();
            return external_model_ ? *external_model_ : *owned_model_;
        }

        /**
         * @brief Get const model reference.
         */
        const GRBModel& model() const
        {
            return external_model_ ? *external_model_ : *owned_model_;
        }

        VarTable& variables() noexcept { return vars_; }
        const VarTable& variables() const noexcept { return vars_; }

        ConTable& constraints() noexcept { return cons_; }
        const ConTable& constraints() const noexcept { return cons_; }

        DataStore& store() noexcept { return store_; }
        const DataStore& store() const noexcept { return store_; }

        // -------------------------------------------------------------------------
        // Parameter Configuration
        // -------------------------------------------------------------------------
        // These methods provide convenient parameter setters that:
        // 1. Eliminate need to remember Gurobi parameter macros
        // 2. Track parameters in store() for diagnostics
        // 3. Support presets for common configurations

        /**
         * @brief Set a Gurobi model parameter (e.g. GRB_DoubleParam_MIPGap).
         *
         * @note For common parameters, prefer the named methods (timeLimit(), 
         *       threads(), etc.) which also track values in store().
         *
         * @example
         *     setParam(GRB_DoubleParam_TimeLimit, 10.0);
         */
        template <typename Param, typename Val>
        void setParam(Param p, Val&& value)
        {
            model().set(p, std::forward<Val>(value));
        }

        /**
         * @brief Set a Gurobi parameter with explicit tracking in store()
         * @param p Gurobi parameter constant
         * @param value Parameter value
         * @param name Key name for store() tracking (e.g., "TimeLimit")
         *
         * @example
         *     setParam(GRB_DoubleParam_Heuristics, 0.5, "Heuristics");
         *     // Later: store()["param:Heuristics"].get<double>()
         */
        template <typename Param, typename Val>
        void setParam(Param p, Val&& value, const std::string& name)
        {
            model().set(p, value);
            store_[std::string("param:") + name] = value;
        }

        /**
         * @brief Set optimization time limit in seconds
         * @param seconds Maximum runtime (0 = no limit)
         * @note Tracked in store()["param:TimeLimit"]
         */
        void timeLimit(double seconds) {
            setParam(GRB_DoubleParam_TimeLimit, seconds);
            store_["param:TimeLimit"] = seconds;
        }

        /**
         * @brief Set relative MIP optimality gap tolerance
         * @param gap Gap as fraction (e.g., 0.01 = 1%)
         * @note Tracked in store()["param:MIPGap"]
         */
        void mipGapLimit(double gap) {
            setParam(GRB_DoubleParam_MIPGap, gap);
            store_["param:MIPGap"] = gap;
        }

        /**
         * @brief Set number of threads for parallel optimization
         * @param n Thread count (0 = automatic)
         * @note Tracked in store()["param:Threads"]
         */
        void threads(int n) {
            setParam(GRB_IntParam_Threads, n);
            store_["param:Threads"] = n;
        }

        /**
         * @brief Suppress solver output
         * @note Tracked in store()["param:OutputFlag"]
         */
        void quiet() {
            setParam(GRB_IntParam_OutputFlag, 0);
            store_["param:OutputFlag"] = 0;
        }

        /**
         * @brief Enable solver output
         * @note Tracked in store()["param:OutputFlag"]
         */
        void verbose() {
            setParam(GRB_IntParam_OutputFlag, 1);
            store_["param:OutputFlag"] = 1;
        }

        /**
         * @brief Set presolve level
         * @param level -1=auto, 0=off, 1=conservative, 2=aggressive
         * @note Tracked in store()["param:Presolve"]
         */
        void presolve(int level) {
            setParam(GRB_IntParam_Presolve, level);
            store_["param:Presolve"] = level;
        }

        /**
         * @brief Set MIP focus strategy
         * @param focus 0=balanced, 1=feasibility, 2=optimality, 3=bound
         * @note Tracked in store()["param:MIPFocus"]
         */
        void mipFocus(int focus) {
            setParam(GRB_IntParam_MIPFocus, focus);
            store_["param:MIPFocus"] = focus;
        }

        // -------------------------------------------------------------------------
        // Parameter Presets
        // -------------------------------------------------------------------------

        /// @brief Predefined parameter configurations for common scenarios
        enum class Preset {
            Fast,       ///< Quick solve: 60s limit, 5% gap, max threads
            Accurate,   ///< Precise solve: 1hr limit, 0.01% gap
            Feasibility,///< Find any solution quickly: MIPFocus=1
            Quiet,      ///< Suppress all output
            Debug       ///< Verbose output, no presolve (for debugging)
        };

        /**
         * @brief Apply a predefined parameter configuration
         * @param p Preset to apply
         *
         * @details Presets configure multiple parameters at once:
         *   - Fast: TimeLimit=60, MIPGap=5%, Threads=0 (auto)
         *   - Accurate: TimeLimit=3600, MIPGap=0.01%
         *   - Feasibility: MIPFocus=1 (prioritize finding feasible solution)
         *   - Quiet: OutputFlag=0
         *   - Debug: OutputFlag=1, Presolve=0
         *
         * @note Preset name is tracked in store()["param:Preset"]
         *
         * @example
         *     void addParameters() override {
         *         applyPreset(Preset::Fast);
         *     }
         */
        void applyPreset(Preset p) {
            switch (p) {
                case Preset::Fast:
                    timeLimit(60.0);
                    mipGapLimit(0.05);
                    threads(0);
                    store_["param:Preset"] = std::string("Fast");
                    break;

                case Preset::Accurate:
                    timeLimit(3600.0);
                    mipGapLimit(0.0001);
                    store_["param:Preset"] = std::string("Accurate");
                    break;

                case Preset::Feasibility:
                    mipFocus(1);
                    store_["param:Preset"] = std::string("Feasibility");
                    break;

                case Preset::Quiet:
                    quiet();
                    store_["param:Preset"] = std::string("Quiet");
                    break;

                case Preset::Debug:
                    verbose();
                    presolve(0);
                    store_["param:Preset"] = std::string("Debug");
                    break;
            }
        }

        // -------------------------------------------------------------------------
        // Objective Helpers
        // -------------------------------------------------------------------------
        // These methods provide a clean interface for setting the objective,
        // eliminating the need to remember GRB_MINIMIZE/GRB_MAXIMIZE macros.

        /**
         * @brief Set the objective to minimize the given expression
         * @param expr Linear expression to minimize
         *
         * @example
         *     minimize(sum(I, [&](int i) { return cost[i] * X(i); }));
         */
        void minimize(const GRBLinExpr& expr) {
            model().setObjective(expr, GRB_MINIMIZE);
        }

        /**
         * @brief Set the objective to maximize the given expression
         * @param expr Linear expression to maximize
         *
         * @example
         *     maximize(sum(I, [&](int i) { return profit[i] * X(i); }));
         */
        void maximize(const GRBLinExpr& expr) {
            model().setObjective(expr, GRB_MAXIMIZE);
        }

        // -------------------------------------------------------------------------
        // Solution Diagnostics
        // -------------------------------------------------------------------------
        // These methods provide convenient access to common post-optimization
        // attributes, eliminating the need to remember Gurobi attribute macros.
        // All methods require the model to have been optimized.

        /**
         * @brief Returns the optimization status code
         * @return Gurobi status code (GRB_OPTIMAL, GRB_INFEASIBLE, etc.)
         * @note Call after optimize()
         */
        int status() const {
            return model().get(GRB_IntAttr_Status);
        }

        /**
         * @brief Returns true if optimal solution was found
         * @return true if status == GRB_OPTIMAL
         */
        bool isOptimal() const {
            return status() == GRB_OPTIMAL;
        }

        /**
         * @brief Returns true if a feasible solution exists
         * @return true if status indicates a solution is available
         * @note Returns true for OPTIMAL, SUBOPTIMAL, SOLUTION_LIMIT, TIME_LIMIT (with solution)
         */
        bool hasSolution() const {
            int s = status();
            return s == GRB_OPTIMAL ||
                   s == GRB_SUBOPTIMAL ||
                   s == GRB_SOLUTION_LIMIT ||
                   (s == GRB_TIME_LIMIT && solutionCount() > 0) ||
                   (s == GRB_NODE_LIMIT && solutionCount() > 0);
        }

        /**
         * @brief Returns true if model is infeasible
         * @return true if status == GRB_INFEASIBLE
         */
        bool isInfeasible() const {
            return status() == GRB_INFEASIBLE;
        }

        /**
         * @brief Returns true if model is unbounded
         * @return true if status == GRB_UNBOUNDED
         */
        bool isUnbounded() const {
            return status() == GRB_UNBOUNDED;
        }

        /**
         * @brief Returns the objective value of the best solution
         * @return Objective value (GRB_DoubleAttr_ObjVal)
         * @throws GRBException if no solution available
         */
        double objVal() const {
            return model().get(GRB_DoubleAttr_ObjVal);
        }

        /**
         * @brief Returns the best known bound on optimal objective
         * @return Best objective bound (GRB_DoubleAttr_ObjBound)
         * @note For MIP: provides dual bound; for LP: equals objVal when optimal
         */
        double objBound() const {
            return model().get(GRB_DoubleAttr_ObjBound);
        }

        /**
         * @brief Returns the relative MIP optimality gap
         * @return MIP gap as fraction (e.g., 0.01 = 1%)
         * @note Only meaningful for MIP models with a solution
         */
        double mipGap() const {
            return model().get(GRB_DoubleAttr_MIPGap);
        }

        /**
         * @brief Returns the optimization runtime in seconds
         * @return Wall-clock time spent in optimize() (GRB_DoubleAttr_Runtime)
         */
        double runtime() const {
            return model().get(GRB_DoubleAttr_Runtime);
        }

        /**
         * @brief Returns the number of solutions found
         * @return Solution count (GRB_IntAttr_SolCount)
         * @note For MIP: number of feasible solutions in solution pool
         */
        int solutionCount() const {
            return model().get(GRB_IntAttr_SolCount);
        }

        /**
         * @brief Returns the number of branch-and-bound nodes explored
         * @return Node count (GRB_DoubleAttr_NodeCount)
         * @note Returns 0 for LP models
         */
        double nodeCount() const {
            return model().get(GRB_DoubleAttr_NodeCount);
        }

        /**
         * @brief Returns the number of simplex iterations
         * @return Iteration count (GRB_DoubleAttr_IterCount)
         */
        double iterCount() const {
            return model().get(GRB_DoubleAttr_IterCount);
        }

        // -------------------------------------------------------------------------
        // Template-method hooks for derived classes
        // -------------------------------------------------------------------------

        /// @brief Configure environment before model creation (license, threads, etc.)
        virtual void configureEnvironment(GRBEnv& env) {}

        /// @brief Add model-level numeric parameters (TimeLimit, MIPGap, Threads).
        virtual void addParameters() {}

        /// @brief Define decision variables into vars_ and underlying GRBModel.
        virtual void addVariables() {}

        /// @brief Define constraints into cons_ and underlying GRBModel.
        virtual void addConstraints() {}

        /// @brief Define objective function (min or max).
        virtual void addObjective() {}

        /// @brief Optional pre-optimization hook (warm starts, fixing variables).
        virtual void beforeOptimize() {}

        /// @brief Optional post-optimization hook (solution extraction).
        virtual void afterOptimize() {}

        // -------------------------------------------------------------------------
        // Main orchestration
        // -------------------------------------------------------------------------

        /**
         * @brief Construct and optimize the model using the template workflow.
         *
         * Steps:
         *     1. initialize()               (if not already)
         *     2. addVariables()
         *     3. addConstraints()
         *     4. addParameters()
         *     5. addObjective()
         *     6. beforeOptimize()
         *     7. model.optimize()
         *     8. afterOptimize()
         *
         * Returns:
         *     Mutable reference to the underlying GRBModel.
         */
        GRBModel& optimize()
        {
            initialize();

            addVariables();
            addConstraints();
            addParameters();
            addObjective();

            beforeOptimize();
            model().optimize();
            afterOptimize();

            return model();
        }
    };

} // namespace dsl