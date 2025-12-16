/*
================================================================================
EXAMPLE 02: BINARY KNAPSACK - Item Selection with Conflicts
================================================================================
DIFFICULTY: Beginner
PROBLEM TYPE: Mixed-Integer Programming (MIP)

PROBLEM DESCRIPTION
-------------------
A hiker needs to select items for a backpack. Each item has a value and weight.
The goal is to maximize total value while respecting the weight capacity.
Additionally, some items are incompatible and cannot be both selected.

MATHEMATICAL MODEL
------------------
Sets:
    I = {0, 1, ..., n-1}         Items
    C = {(i,j) : incompatible}   Conflict pairs

Parameters:
    value[i]        Value of item i
    weight[i]       Weight of item i
    capacity        Maximum weight capacity

Variables:
    x[i] in {0,1}   1 if item i is selected, 0 otherwise

Objective:
    max  sum_{i in I} value[i] * x[i]

Constraints:
    Capacity:   sum_{i in I} weight[i] * x[i] <= capacity
    Conflict:   x[i] + x[j] <= 1    for all (i,j) in C

DSL FEATURES DEMONSTRATED
-------------------------
- ModelBuilder<VarEnum, ConEnum>  Template method pattern
- DECLARE_ENUM_WITH_COUNT         Type-safe enum keys
- applyPreset(Preset::Fast)       Parameter presets
- mipGapLimit()                   MIP gap setting
- DataStore                       Metadata storage
- fix(), unfix()                  What-if analysis
- hasSolution(), objVal()         Solution diagnostics
- solutionCount(), mipGap()       MIP-specific diagnostics

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
DECLARE_ENUM_WITH_COUNT(Cons, Capacity, Conflict);

// ============================================================================
// KNAPSACK BUILDER
// ============================================================================
class KnapsackBuilder : public dsl::ModelBuilder<Vars, Cons> {
private:
    std::vector<std::string> itemNames_;
    std::vector<double> values_;
    std::vector<double> weights_;
    double capacity_;
    std::vector<std::pair<int, int>> conflicts_;

    int n() const { return static_cast<int>(values_.size()); }

public:
    KnapsackBuilder(
        const std::vector<std::string>& itemNames,
        const std::vector<double>& values,
        const std::vector<double>& weights,
        double capacity,
        const std::vector<std::pair<int, int>>& conflicts = {}
    ) : itemNames_(itemNames), values_(values), weights_(weights),
        capacity_(capacity), conflicts_(conflicts)
    {
        store()["n_items"] = n();
        store()["n_conflicts"] = static_cast<int>(conflicts.size());
        store()["capacity"] = capacity;
    }

protected:
    void addParameters() override {
        applyPreset(Preset::Fast);  // TimeLimit=60s, MIPGap=5%
        quiet();
        mipGapLimit(0.001);         // Override: 0.1% gap
    }

    void addVariables() override {
        // Binary selection variables
        auto X = dsl::VariableFactory::add(
            model(), GRB_BINARY, 0.0, 1.0, "x", n()
        );
        variables().set(Vars::X, std::move(X));
    }

    void addConstraints() override {
        auto& X = variables().get(Vars::X);
        auto I = dsl::range(0, n());

        // Capacity constraint
        auto capConstr = dsl::ConstraintFactory::add(
            model(), "capacity",
            [&](const std::vector<int>&) {
                return dsl::sum(I, [&](int i) {
                    return weights_[i] * X.at(i);
                }) <= capacity_;
            }
        );
        constraints().set(Cons::Capacity, std::move(capConstr));

        // Conflict constraints
        if (!conflicts_.empty()) {
            auto K = dsl::range(0, static_cast<int>(conflicts_.size()));
            
            auto conflictConstrs = dsl::ConstraintFactory::addIndexed(
                model(), "conflict", K,
                [&](int k) {
                    int i = conflicts_[k].first;
                    int j = conflicts_[k].second;
                    return X.at(i) + X.at(j) <= 1;
                }
            );
            constraints().set(Cons::Conflict, std::move(conflictConstrs));
        }
    }

    void addObjective() override {
        auto& X = variables().get(Vars::X);
        auto I = dsl::range(0, n());

        maximize(dsl::sum(I, [&](int i) {
            return values_[i] * X.at(i);
        }));
    }

    void afterOptimize() override {
        if (hasSolution()) {
            store()["objective"] = objVal();
            store()["runtime"] = runtime();
            store()["gap"] = mipGap();
            store()["solutions_found"] = solutionCount();
            
            // Compute solution statistics
            int selected = 0;
            double totalWeight = 0;
            double totalValue = 0;
            auto& X = variables().get(Vars::X);
            
            for (int i = 0; i < n(); ++i) {
                if (dsl::value(X.at(i)) > 0.5) {
                    selected++;
                    totalWeight += weights_[i];
                    totalValue += values_[i];
                }
            }
            store()["selected_count"] = selected;
            store()["total_weight"] = totalWeight;
            store()["total_value"] = totalValue;
        }
    }

public:
    // ========================================================================
    // Solution Accessors
    // ========================================================================
    
    std::vector<int> getSelectedItems() const {
        std::vector<int> selected;
        const auto& X = variables().get(Vars::X);
        
        for (int i = 0; i < n(); ++i) {
            if (dsl::value(X.at(i)) > 0.5) {
                selected.push_back(i);
            }
        }
        return selected;
    }
    
    const std::vector<std::string>& itemNames() const { return itemNames_; }
    const std::vector<double>& values() const { return values_; }
    const std::vector<double>& weights() const { return weights_; }
    
    // ========================================================================
    // What-If Analysis
    // ========================================================================
    
    double solveWithItemFixed(int item, bool include) {
        auto& X = variables().get(Vars::X);
        
        // Fix the variable
        dsl::fix(X.at(item), include ? 1.0 : 0.0);
        
        // Re-optimize
        model().optimize();
        double obj = hasSolution() ? objVal() : -1;
        
        // Restore variable bounds
        dsl::unfix(X.at(item), 0.0, 1.0);
        
        return obj;
    }
};

// ============================================================================
// MAIN PROGRAM
// ============================================================================
int main() {
    std::cout << "================================================================\n";
    std::cout << "EXAMPLE 02: Binary Knapsack with Conflicts\n";
    std::cout << "================================================================\n\n";

    try {
        // ====================================================================
        // PROBLEM DATA
        // ====================================================================
        std::vector<std::string> itemNames = {
            "Tent", "Sleeping Bag", "Cooking Set", "Water Filter",
            "First Aid Kit", "Flashlight", "Camera", "Book"
        };
        
        std::vector<double> values  = {60, 100, 120, 80, 90, 75, 110, 95};
        std::vector<double> weights = {10,  20,  30, 15, 25, 12, 28, 18};
        double capacity = 80;
        
        // Item conflicts (cannot select both)
        std::vector<std::pair<int, int>> conflicts = {
            {0, 1},  // Tent and Sleeping Bag (redundant shelter)
            {2, 3},  // Cooking Set and Water Filter (weight limit)
            {4, 6}   // First Aid Kit and Camera (space constraint)
        };

        const int nItems = static_cast<int>(values.size());

        // ====================================================================
        // PRINT PROBLEM DESCRIPTION
        // ====================================================================
        std::cout << "PROBLEM DATA\n";
        std::cout << "------------\n";
        std::cout << "Backpack Capacity: " << capacity << " kg\n\n";

        std::cout << "Available Items:\n";
        std::cout << std::setw(4) << "ID" << std::setw(16) << "Item" 
                  << std::setw(10) << "Value($)" << std::setw(12) << "Weight(kg)"
                  << std::setw(12) << "Value/kg\n";
        std::cout << std::string(54, '-') << "\n";
        
        for (int i = 0; i < nItems; ++i) {
            std::cout << std::setw(4) << i 
                      << std::setw(16) << itemNames[i]
                      << std::setw(10) << values[i]
                      << std::setw(12) << weights[i]
                      << std::setw(12) << std::fixed << std::setprecision(2) 
                      << values[i]/weights[i] << "\n";
        }

        std::cout << "\nConflicting Item Pairs (cannot select both):\n";
        for (const auto& [i, j] : conflicts) {
            std::cout << "  - " << itemNames[i] << " and " << itemNames[j] << "\n";
        }
        std::cout << "\n";

        // ====================================================================
        // BUILD AND SOLVE MODEL
        // ====================================================================
        std::cout << "SOLVING...\n";
        std::cout << "----------\n";
        
        KnapsackBuilder builder(itemNames, values, weights, capacity, conflicts);
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
            std::cout << "Total Value:  $" << builder.store()["total_value"].get<double>() << "\n";
            std::cout << "Total Weight: " << builder.store()["total_weight"].get<double>() 
                      << " / " << capacity << " kg\n";
            std::cout << "Items Selected: " << builder.store()["selected_count"].get<int>() 
                      << " / " << nItems << "\n";
            std::cout << "MIP Gap: " << builder.store()["gap"].get<double>() * 100 << "%\n";
            std::cout << "Solutions Found: " << builder.store()["solutions_found"].get<int>() << "\n\n";

            std::cout << "Selected Items:\n";
            for (int i : builder.getSelectedItems()) {
                std::cout << "  [x] " << std::setw(16) << std::left << itemNames[i]
                          << std::right << " Value: $" << std::setw(6) << values[i]
                          << "  Weight: " << std::setw(5) << weights[i] << " kg\n";
            }

            std::cout << "\nNot Selected:\n";
            auto selected = builder.getSelectedItems();
            for (int i = 0; i < nItems; ++i) {
                if (std::find(selected.begin(), selected.end(), i) == selected.end()) {
                    std::cout << "  [ ] " << std::setw(16) << std::left << itemNames[i]
                              << std::right << " Value: $" << std::setw(6) << values[i]
                              << "  Weight: " << std::setw(5) << weights[i] << " kg\n";
                }
            }

            // ================================================================
            // WHAT-IF ANALYSIS
            // ================================================================
            std::cout << "\nWHAT-IF ANALYSIS\n";
            std::cout << "----------------\n";
            std::cout << "Original optimal value: $" << builder.objVal() << "\n\n";

            // Test forcing each unselected item
            std::cout << "Impact of forcing unselected items:\n";
            for (int i = 0; i < nItems; ++i) {
                if (std::find(selected.begin(), selected.end(), i) == selected.end()) {
                    double objForced = builder.solveWithItemFixed(i, true);
                    double change = objForced - builder.objVal();
                    std::cout << "  Force " << std::setw(16) << std::left << itemNames[i]
                              << std::right << ": $" << objForced;
                    if (objForced >= 0) {
                        std::cout << " (change: " << (change >= 0 ? "+" : "") << change << ")\n";
                    } else {
                        std::cout << " (infeasible)\n";
                    }
                }
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
