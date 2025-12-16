/*
================================================================================
EXAMPLE 04: ASSIGNMENT PROBLEM - Worker-Task Assignment with Qualifications
================================================================================
DIFFICULTY: Intermediate
PROBLEM TYPE: Mixed-Integer Programming (MIP)

PROBLEM DESCRIPTION
-------------------
A company needs to assign workers to tasks. Not all workers are qualified for
all tasks. Each qualified worker-task pair has an associated cost. The goal is
to minimize total assignment cost while:
- Each task must be completed by exactly one qualified worker
- Each worker can be assigned to at most one task

MATHEMATICAL MODEL
------------------
Sets:
    W = {0, 1, ..., m-1}                Workers
    T = {0, 1, ..., n-1}                Tasks
    Q = {(w,t) : qualified[w,t] = 1}    Qualified worker-task pairs

Parameters:
    cost[w,t]       Cost of assigning worker w to task t (only for (w,t) in Q)

Variables:
    x[w,t] in {0,1}     1 if worker w is assigned to task t (only for (w,t) in Q)

Objective:
    min  sum_{(w,t) in Q} cost[w,t] * x[w,t]

Constraints:
    TaskCover[t]:    sum_{w: (w,t) in Q} x[w,t] = 1     for all t in T
    WorkerLimit[w]:  sum_{t: (w,t) in Q} x[w,t] <= 1    for all w in W

DSL FEATURES DEMONSTRATED
-------------------------
- Cartesian product           W * T
- Filtered domains            (W * T) | dsl::filter(predicate)
- IndexedVariableSet          Sparse variable structures
- IndexedConstraintSet        Domain-based constraints
- operator<<                  Domain printing utilities
- try_get()                   Safe sparse access
- computeIIS()                Infeasibility analysis
- Conditional constraint gen  Based on domain elements

================================================================================
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <gurobi_dsl/dsl.h>

// ============================================================================
// TYPE-SAFE ENUM KEYS
// ============================================================================
DECLARE_ENUM_WITH_COUNT(Vars, X);
DECLARE_ENUM_WITH_COUNT(Cons, TaskCover, WorkerLimit);

// ============================================================================
// ASSIGNMENT BUILDER
// ============================================================================
class AssignmentBuilder : public dsl::ModelBuilder<Vars, Cons> {
private:
    std::vector<std::string> workerNames_;
    std::vector<std::string> taskNames_;
    std::vector<std::vector<int>> qualified_;     // qualified[w][t]
    std::vector<std::vector<double>> cost_;       // cost[w][t]
    
    int nWorkers_;
    int nTasks_;

public:
    AssignmentBuilder(
        const std::vector<std::string>& workerNames,
        const std::vector<std::string>& taskNames,
        const std::vector<std::vector<int>>& qualified,
        const std::vector<std::vector<double>>& cost
    ) : workerNames_(workerNames), taskNames_(taskNames),
        qualified_(qualified), cost_(cost),
        nWorkers_(static_cast<int>(workerNames.size())),
        nTasks_(static_cast<int>(taskNames.size()))
    {
        store()["n_workers"] = nWorkers_;
        store()["n_tasks"] = nTasks_;
        
        // Count qualified pairs
        int nQualified = 0;
        for (int w = 0; w < nWorkers_; ++w) {
            for (int t = 0; t < nTasks_; ++t) {
                if (qualified_[w][t]) nQualified++;
            }
        }
        store()["n_qualified_pairs"] = nQualified;
    }

protected:
    void addParameters() override {
        quiet();
    }

    void addVariables() override {
        auto W = dsl::range(0, nWorkers_);
        auto T = dsl::range(0, nTasks_);
        
        // Create filtered domain: only qualified pairs
        auto qualifiedDomain = (W * T) | dsl::filter([this](int w, int t) {
            return qualified_[w][t] == 1;
        });

        // Sparse binary variables only for qualified pairs
        auto X = dsl::VariableFactory::addIndexed(
            model(), GRB_BINARY, 0.0, 1.0, "x", qualifiedDomain
        );
        
        variables().set(Vars::X, std::move(X));
    }

    void addConstraints() override {
        auto& X = variables().get(Vars::X);
        auto W = dsl::range(0, nWorkers_);
        auto T = dsl::range(0, nTasks_);

        // TaskCover[t]: Each task must be assigned to exactly one worker
        auto taskConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "task_cover", T,
            [&](int t) {
                GRBLinExpr lhs = 0;
                for (int w : W) {
                    auto* var = X.asIndexed().try_get(w, t);
                    if (var) lhs += *var;
                }
                return lhs == 1;
            }
        );
        constraints().set(Cons::TaskCover, std::move(taskConstrs));

        // WorkerLimit[w]: Each worker can do at most one task
        auto workerConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "worker_limit", W,
            [&](int w) {
                GRBLinExpr lhs = 0;
                for (int t : T) {
                    auto* var = X.asIndexed().try_get(w, t);
                    if (var) lhs += *var;
                }
                return lhs <= 1;
            }
        );
        constraints().set(Cons::WorkerLimit, std::move(workerConstrs));
    }

    void addObjective() override {
        auto& X = variables().get(Vars::X);
        
        // Minimize total assignment cost
        GRBLinExpr obj = 0;
        for (const auto& entry : X.asIndexed()) {
            int w = entry.index[0];
            int t = entry.index[1];
            obj += cost_[w][t] * entry.var;
        }
        minimize(obj);
    }

    void afterOptimize() override {
        if (hasSolution()) {
            store()["total_cost"] = objVal();
            store()["runtime"] = runtime();
            
            // Count assignments
            int assigned = 0;
            auto& X = variables().get(Vars::X);
            for (const auto& entry : X.asIndexed()) {
                if (dsl::value(entry.var) > 0.5) assigned++;
            }
            store()["n_assignments"] = assigned;
        }
    }

public:
    // Accessors
    const std::vector<std::string>& workerNames() const { return workerNames_; }
    const std::vector<std::string>& taskNames() const { return taskNames_; }
    int nWorkers() const { return nWorkers_; }
    int nTasks() const { return nTasks_; }
    bool isQualified(int w, int t) const { return qualified_[w][t] == 1; }
    double assignmentCost(int w, int t) const { return cost_[w][t]; }
    
    // Get the optimal assignments
    std::vector<std::pair<int, int>> getAssignments() const {
        std::vector<std::pair<int, int>> assignments;
        const auto& X = variables().get(Vars::X);
        
        for (const auto& entry : X.asIndexed()) {
            if (dsl::value(entry.var) > 0.5) {
                assignments.emplace_back(entry.index[0], entry.index[1]);
            }
        }
        return assignments;
    }
};

// ============================================================================
// MAIN PROGRAM
// ============================================================================
int main() {
    std::cout << "================================================================\n";
    std::cout << "EXAMPLE 04: Assignment Problem with Qualifications\n";
    std::cout << "================================================================\n\n";

    try {
        // ====================================================================
        // PROBLEM DATA
        // ====================================================================
        std::vector<std::string> workerNames = {
            "Alice", "Bob", "Carol", "David", "Eve"
        };
        
        std::vector<std::string> taskNames = {
            "Design", "Coding", "Testing", "Documentation"
        };

        // Qualification matrix (1 = qualified, 0 = not qualified)
        //                        Design  Coding  Testing  Documentation
        std::vector<std::vector<int>> qualified = {
            {1, 1, 0, 1},  // Alice
            {0, 1, 1, 0},  // Bob
            {1, 0, 1, 1},  // Carol
            {0, 1, 1, 1},  // David
            {1, 1, 0, 0}   // Eve
        };
        
        // Cost matrix (cost for each qualified assignment)
        //                        Design  Coding  Testing  Documentation
        std::vector<std::vector<double>> cost = {
            {50, 60,  0, 40},  // Alice
            { 0, 45, 55,  0},  // Bob
            {65,  0, 50, 35},  // Carol
            { 0, 70, 45, 50},  // David
            {55, 50,  0,  0}   // Eve
        };

        const int nW = static_cast<int>(workerNames.size());
        const int nT = static_cast<int>(taskNames.size());

        // ====================================================================
        // PRINT PROBLEM DESCRIPTION
        // ====================================================================
        std::cout << "PROBLEM DATA\n";
        std::cout << "------------\n";
        std::cout << "Workers: " << nW << "\n";
        std::cout << "Tasks: " << nT << "\n\n";

        std::cout << "Qualification & Cost Matrix:\n";
        std::cout << "  (Q = Qualified, '-' = Not Qualified, numbers = cost)\n\n";
        
        std::cout << std::setw(12) << "";
        for (int t = 0; t < nT; ++t) {
            std::cout << std::setw(14) << taskNames[t];
        }
        std::cout << "\n" << std::string(12 + nT * 14, '-') << "\n";
        
        for (int w = 0; w < nW; ++w) {
            std::cout << std::setw(12) << workerNames[w];
            for (int t = 0; t < nT; ++t) {
                if (qualified[w][t]) {
                    std::cout << std::setw(14) << ("$" + std::to_string(static_cast<int>(cost[w][t])));
                } else {
                    std::cout << std::setw(14) << "-";
                }
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        // Show qualified pairs domain
        auto W = dsl::range(0, nW);
        auto T = dsl::range(0, nT);
        auto qualDomain = (W * T) | dsl::filter([&](int w, int t) {
            return qualified[w][t] == 1;
        });
        
        std::cout << "Qualified worker-task pairs:\n  ";
        int count = 0;
        for (auto [w, t] : qualDomain) {
            std::cout << "(" << workerNames[w] << "," << taskNames[t] << ") ";
            count++;
            if (count % 4 == 0) std::cout << "\n  ";
        }
        std::cout << "\n\n";

        // ====================================================================
        // BUILD AND SOLVE MODEL
        // ====================================================================
        std::cout << "SOLVING...\n";
        std::cout << "----------\n";
        
        AssignmentBuilder builder(workerNames, taskNames, qualified, cost);
        builder.optimize();

        std::cout << "Status: " << dsl::statusString(builder.status()) << "\n";
        std::cout << dsl::modelSummary(builder.model()) << "\n";

        // ====================================================================
        // DISPLAY RESULTS
        // ====================================================================
        if (builder.hasSolution()) {
            std::cout << std::fixed << std::setprecision(2);
            
            std::cout << "\nOPTIMAL SOLUTION\n";
            std::cout << "----------------\n";
            std::cout << "Total Assignment Cost: $" << builder.objVal() << "\n";
            std::cout << "Assignments Made: " << builder.store()["n_assignments"].get<int>() 
                      << " / " << nT << " tasks\n\n";

            std::cout << "Optimal Assignments:\n";
            double totalCost = 0;
            for (auto [w, t] : builder.getAssignments()) {
                double c = cost[w][t];
                totalCost += c;
                std::cout << "  " << std::setw(10) << std::left << workerNames[w]
                          << std::right << " -> " << std::setw(14) << std::left << taskNames[t]
                          << std::right << " (Cost: $" << c << ")\n";
            }
            std::cout << "\n";

            // Show unassigned workers
            std::cout << "Worker Status:\n";
            auto assignments = builder.getAssignments();
            for (int w = 0; w < nW; ++w) {
                bool assigned = false;
                for (auto [aw, at] : assignments) {
                    if (aw == w) {
                        assigned = true;
                        break;
                    }
                }
                std::cout << "  " << std::setw(10) << workerNames[w] << ": "
                          << (assigned ? "Assigned" : "Available") << "\n";
            }

        } else if (builder.status() == GRB_INFEASIBLE) {
            // ================================================================
            // INFEASIBILITY ANALYSIS
            // ================================================================
            std::cout << "\nModel is INFEASIBLE! Computing IIS...\n";
            
            auto iis = dsl::computeIIS(builder.model());
            
            std::cout << "\nIrreducible Inconsistent Subsystem (IIS):\n";
            std::cout << "  Constraints in IIS: " << iis.constraints.size() << "\n";
            for (const auto& [name, constr] : iis.constraints) {
                std::cout << "    - " << name << "\n";
            }
            std::cout << "  Variable lower bounds in IIS: " << iis.lowerBounds.size() << "\n";
            for (const auto& [varName, var] : iis.lowerBounds) {
                std::cout << "    - " << varName << " (lower bound)\n";
            }
            std::cout << "  Variable upper bounds in IIS: " << iis.upperBounds.size() << "\n";
            for (const auto& [varName, var] : iis.upperBounds) {
                std::cout << "    - " << varName << " (upper bound)\n";
            }
        }

    } catch (GRBException& e) {
        std::cerr << "Gurobi Error " << e.getErrorCode() << ": " << e.getMessage() << "\n";
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n================================================================\n";
    return 0;
}
