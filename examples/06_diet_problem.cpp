/*
================================================================================
EXAMPLE 06: DIET PROBLEM - Nutritional Optimization
================================================================================
DIFFICULTY: Intermediate
PROBLEM TYPE: Linear Programming (LP)

PROBLEM DESCRIPTION
-------------------
A nutritionist wants to design a minimum-cost diet that meets daily nutritional
requirements. Each food has a cost per serving and provides certain amounts of
various nutrients. The diet must satisfy minimum and maximum intake levels for
each nutrient.

This is a classic LP problem demonstrating bounded constraints and multiple
resources (nutrients) being balanced.

MATHEMATICAL MODEL
------------------
Sets:
    F = {0, 1, ..., m-1}    Foods
    N = {0, 1, ..., n-1}    Nutrients

Parameters:
    cost[f]             Cost per serving of food f
    nutrients[f,n]      Amount of nutrient n per serving of food f
    minIntake[n]        Minimum required intake of nutrient n
    maxIntake[n]        Maximum allowed intake of nutrient n
    maxServings[f]      Maximum servings of food f (variety constraint)

Variables:
    x[f] >= 0           Servings of food f to include in diet

Objective:
    min  sum_f cost[f] * x[f]

Constraints:
    NutrientMin[n]:  sum_f nutrients[f,n] * x[f] >= minIntake[n]   for all n
    NutrientMax[n]:  sum_f nutrients[f,n] * x[f] <= maxIntake[n]   for all n
    MaxServings[f]:  x[f] <= maxServings[f]                         for all f

DSL FEATURES DEMONSTRATED
-------------------------
- Range constraints             min <= expr <= max
- Multiple constraint types     Min/Max nutrient bounds
- Dual values interpretation    Shadow prices for nutrition
- Solution analysis             Binding vs slack constraints
- setLB(), setUB()              Bound modification utilities

================================================================================
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <gurobi_dsl/dsl.h>

// ============================================================================
// TYPE-SAFE ENUM KEYS
// ============================================================================
DECLARE_ENUM_WITH_COUNT(Vars, X);
DECLARE_ENUM_WITH_COUNT(Cons, NutrientMin, NutrientMax, MaxServings);

// ============================================================================
// DIET BUILDER
// ============================================================================
class DietBuilder : public dsl::ModelBuilder<Vars, Cons> {
private:
    std::vector<std::string> foodNames_;
    std::vector<std::string> nutrientNames_;
    std::vector<double> cost_;                          // cost per serving
    std::vector<std::vector<double>> nutrients_;        // nutrients[food][nutrient]
    std::vector<double> minIntake_;
    std::vector<double> maxIntake_;
    std::vector<double> maxServings_;
    
    int nFoods_;
    int nNutrients_;

public:
    DietBuilder(
        const std::vector<std::string>& foodNames,
        const std::vector<std::string>& nutrientNames,
        const std::vector<double>& cost,
        const std::vector<std::vector<double>>& nutrients,
        const std::vector<double>& minIntake,
        const std::vector<double>& maxIntake,
        const std::vector<double>& maxServings
    ) : foodNames_(foodNames), nutrientNames_(nutrientNames),
        cost_(cost), nutrients_(nutrients),
        minIntake_(minIntake), maxIntake_(maxIntake), maxServings_(maxServings),
        nFoods_(static_cast<int>(foodNames.size())),
        nNutrients_(static_cast<int>(nutrientNames.size()))
    {
        store()["n_foods"] = nFoods_;
        store()["n_nutrients"] = nNutrients_;
    }

protected:
    void addParameters() override {
        quiet();
    }

    void addVariables() override {
        // Servings of each food (continuous, with upper bound from maxServings)
        auto X = dsl::VariableFactory::add(
            model(), GRB_CONTINUOUS, 0.0, GRB_INFINITY, "x", nFoods_
        );
        
        // Set upper bounds based on maxServings
        for (int f = 0; f < nFoods_; ++f) {
            dsl::setUB(X.at(f), maxServings_[f]);
        }
        
        variables().set(Vars::X, std::move(X));
    }

    void addConstraints() override {
        auto& X = variables().get(Vars::X);
        auto F = dsl::range(0, nFoods_);
        auto N = dsl::range(0, nNutrients_);

        // NutrientMin[n]: Minimum nutrient intake
        auto minConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "nutrient_min", N,
            [&](int n) {
                return dsl::sum(F, [&](int f) { 
                    return nutrients_[f][n] * X.at(f); 
                }) >= minIntake_[n];
            }
        );
        constraints().set(Cons::NutrientMin, std::move(minConstrs));

        // NutrientMax[n]: Maximum nutrient intake
        auto maxConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "nutrient_max", N,
            [&](int n) {
                return dsl::sum(F, [&](int f) { 
                    return nutrients_[f][n] * X.at(f); 
                }) <= maxIntake_[n];
            }
        );
        constraints().set(Cons::NutrientMax, std::move(maxConstrs));
    }

    void addObjective() override {
        auto& X = variables().get(Vars::X);
        auto F = dsl::range(0, nFoods_);

        // Minimize total cost
        minimize(dsl::sum(F, [&](int f) {
            return cost_[f] * X.at(f);
        }));
    }

    void afterOptimize() override {
        if (hasSolution()) {
            store()["total_cost"] = objVal();
            store()["runtime"] = runtime();
            
            // Count foods used
            auto& X = variables().get(Vars::X);
            int foodsUsed = 0;
            for (int f = 0; f < nFoods_; ++f) {
                if (dsl::value(X.at(f)) > 0.01) foodsUsed++;
            }
            store()["foods_used"] = foodsUsed;
        }
    }

public:
    // Accessors
    const std::vector<std::string>& foodNames() const { return foodNames_; }
    const std::vector<std::string>& nutrientNames() const { return nutrientNames_; }
    int nFoods() const { return nFoods_; }
    int nNutrients() const { return nNutrients_; }
    const std::vector<std::vector<double>>& nutrients() const { return nutrients_; }
    const std::vector<double>& minIntake() const { return minIntake_; }
    const std::vector<double>& maxIntake() const { return maxIntake_; }
    
    // Get constraint groups
    const dsl::IndexedConstraintSet& minConstraints() const {
        return constraints().get(Cons::NutrientMin).asIndexed();
    }
    
    const dsl::IndexedConstraintSet& maxConstraints() const {
        return constraints().get(Cons::NutrientMax).asIndexed();
    }
};

// ============================================================================
// MAIN PROGRAM
// ============================================================================
int main() {
    std::cout << "================================================================\n";
    std::cout << "EXAMPLE 06: Diet Problem - Nutritional Optimization\n";
    std::cout << "================================================================\n\n";

    try {
        // ====================================================================
        // PROBLEM DATA
        // ====================================================================
        std::vector<std::string> foodNames = {
            "Bread", "Milk", "Cheese", "Potato", "Fish", "Yogurt"
        };
        
        std::vector<std::string> nutrientNames = {
            "Calories", "Protein(g)", "Calcium(mg)", "VitaminA(IU)", "VitaminC(mg)"
        };
        
        // Cost per serving ($)
        std::vector<double> cost = {0.50, 0.75, 1.20, 0.40, 2.50, 0.80};
        
        // Nutrient content per serving [food][nutrient]
        //                  Calories  Protein  Calcium  VitA    VitC
        std::vector<std::vector<double>> nutrients = {
            {100,     4,      30,      0,       0},    // Bread
            {160,     8,      285,     500,     2},    // Milk
            {110,     7,      200,     300,     0},    // Cheese
            {160,     4,      10,      0,       20},   // Potato
            {200,     25,     20,      100,     0},    // Fish
            {120,     8,      300,     100,     4}     // Yogurt
        };
        
        // Daily requirements (min and max) - adjusted for feasibility
        std::vector<double> minIntake = {2000, 55, 800, 1500, 50};
        std::vector<double> maxIntake = {2500, 100, 1600, 10000, 200};
        
        // Maximum servings per food (for variety)
        std::vector<double> maxServings = {10, 8, 6, 8, 4, 6};

        const int nF = static_cast<int>(foodNames.size());
        const int nN = static_cast<int>(nutrientNames.size());

        // ====================================================================
        // PRINT PROBLEM DESCRIPTION
        // ====================================================================
        std::cout << "PROBLEM DATA\n";
        std::cout << "------------\n";
        
        std::cout << "Available Foods:\n";
        std::cout << std::setw(12) << "Food" << std::setw(10) << "Cost" 
                  << std::setw(12) << "Max Serv" << "\n";
        std::cout << std::string(34, '-') << "\n";
        for (int f = 0; f < nF; ++f) {
            std::cout << std::setw(12) << foodNames[f] 
                      << std::setw(10) << ("$" + std::to_string(cost[f]).substr(0, 4))
                      << std::setw(12) << maxServings[f] << "\n";
        }
        std::cout << "\n";
        
        std::cout << "Nutrient Content per Serving:\n";
        std::cout << std::setw(12) << "";
        for (int n = 0; n < nN; ++n) {
            std::cout << std::setw(12) << nutrientNames[n].substr(0, 10);
        }
        std::cout << "\n" << std::string(12 + nN * 12, '-') << "\n";
        
        for (int f = 0; f < nF; ++f) {
            std::cout << std::setw(12) << foodNames[f];
            for (int n = 0; n < nN; ++n) {
                std::cout << std::setw(12) << static_cast<int>(nutrients[f][n]);
            }
            std::cout << "\n";
        }
        std::cout << "\n";
        
        std::cout << "Daily Requirements:\n";
        std::cout << std::setw(15) << "Nutrient" << std::setw(12) << "Minimum" 
                  << std::setw(12) << "Maximum" << "\n";
        std::cout << std::string(39, '-') << "\n";
        for (int n = 0; n < nN; ++n) {
            std::cout << std::setw(15) << nutrientNames[n] 
                      << std::setw(12) << minIntake[n]
                      << std::setw(12) << maxIntake[n] << "\n";
        }
        std::cout << "\n";

        // ====================================================================
        // BUILD AND SOLVE MODEL
        // ====================================================================
        std::cout << "SOLVING...\n";
        std::cout << "----------\n";
        
        DietBuilder builder(foodNames, nutrientNames, cost, nutrients,
                           minIntake, maxIntake, maxServings);
        builder.optimize();

        std::cout << "Status: " << dsl::statusString(builder.status()) << "\n";
        std::cout << dsl::modelSummary(builder.model()) << "\n";

        // ====================================================================
        // DISPLAY RESULTS
        // ====================================================================
        if (builder.hasSolution()) {
            std::cout << std::fixed << std::setprecision(2);
            
            std::cout << "\nOPTIMAL DIET PLAN\n";
            std::cout << "-----------------\n";
            std::cout << "Daily Cost: $" << builder.objVal() << "\n";
            std::cout << "Foods Used: " << builder.store()["foods_used"].get<int>() 
                      << " / " << nF << "\n\n";

            // Diet composition
            std::cout << "Recommended Servings:\n";
            std::cout << std::setw(12) << "Food" << std::setw(12) << "Servings" 
                      << std::setw(10) << "Cost" << std::setw(10) << "Max\n";
            std::cout << std::string(44, '-') << "\n";
            
            auto& X = builder.variables().get(Vars::X);
            double totalCost = 0;
            
            for (int f = 0; f < nF; ++f) {
                double servings = dsl::value(X.at(f));
                double foodCost = cost[f] * servings;
                totalCost += foodCost;
                
                if (servings > 0.01) {
                    std::cout << std::setw(12) << foodNames[f]
                              << std::setw(12) << servings
                              << std::setw(10) << ("$" + std::to_string(foodCost).substr(0, 5))
                              << std::setw(10) << maxServings[f];
                    
                    // Flag if at max
                    if (std::abs(servings - maxServings[f]) < 0.01) {
                        std::cout << " [AT MAX]";
                    }
                    std::cout << "\n";
                }
            }
            std::cout << std::string(44, '-') << "\n";
            std::cout << std::setw(12) << "TOTAL" << std::setw(12) << "" 
                      << std::setw(10) << ("$" + std::to_string(totalCost).substr(0, 5)) << "\n\n";

            // Nutrient intake analysis
            std::cout << "Nutrient Intake Analysis:\n";
            std::cout << std::setw(15) << "Nutrient" << std::setw(12) << "Intake"
                      << std::setw(12) << "Min" << std::setw(12) << "Max" 
                      << std::setw(15) << "Status\n";
            std::cout << std::string(66, '-') << "\n";
            
            for (int n = 0; n < nN; ++n) {
                double intake = 0;
                for (int f = 0; f < nF; ++f) {
                    intake += nutrients[f][n] * dsl::value(X.at(f));
                }
                
                std::string status;
                if (std::abs(intake - minIntake[n]) < 0.01) {
                    status = "AT MINIMUM";
                } else if (std::abs(intake - maxIntake[n]) < 0.01) {
                    status = "AT MAXIMUM";
                } else {
                    status = "OK";
                }
                
                std::cout << std::setw(15) << nutrientNames[n]
                          << std::setw(12) << static_cast<int>(intake)
                          << std::setw(12) << static_cast<int>(minIntake[n])
                          << std::setw(12) << static_cast<int>(maxIntake[n])
                          << std::setw(15) << status << "\n";
            }
            std::cout << "\n";

            // Shadow prices interpretation
            std::cout << "SENSITIVITY ANALYSIS\n";
            std::cout << "--------------------\n";
            std::cout << "Shadow prices show the marginal cost of tightening constraints:\n\n";
            
            std::cout << "Minimum Nutrient Requirements (cost to increase minimum by 1 unit):\n";
            auto& minCons = builder.minConstraints();
            bool anyMinBinding = false;
            for (int n = 0; n < nN; ++n) {
                double pi = dsl::dual(minCons.at(n));
                // For >= constraints, positive dual means constraint is binding
                if (std::abs(pi) > 0.0001) {
                    anyMinBinding = true;
                    std::cout << "  " << std::setw(15) << nutrientNames[n] 
                              << ": +$" << std::abs(pi) << " per unit increase\n";
                }
            }
            if (!anyMinBinding) {
                std::cout << "  (No minimum constraints are binding)\n";
            }
            
            std::cout << "\nMaximum Nutrient Limits (cost to decrease maximum by 1 unit):\n";
            auto& maxCons = builder.maxConstraints();
            bool anyMaxBinding = false;
            for (int n = 0; n < nN; ++n) {
                double pi = dsl::dual(maxCons.at(n));
                // For <= constraints, negative dual means constraint is binding
                if (std::abs(pi) > 0.0001) {
                    anyMaxBinding = true;
                    std::cout << "  " << std::setw(15) << nutrientNames[n] 
                              << ": +$" << std::abs(pi) << " per unit decrease\n";
                }
            }
            if (!anyMaxBinding) {
                std::cout << "  (No maximum constraints are binding)\n";
            }
            
            std::cout << "\nNote: Binding constraints limit the optimal solution.\n";
            std::cout << "Relaxing binding constraints would reduce cost.\n";
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
