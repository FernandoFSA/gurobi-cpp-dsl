/*
================================================================================
EXAMPLE 01: SIMPLE LINEAR PROGRAMMING - Production Planning
================================================================================
DIFFICULTY: Beginner
PROBLEM TYPE: Linear Programming (LP)

PROBLEM DESCRIPTION
-------------------
A factory produces 3 products (A, B, C) using 2 machines. Each product requires
different processing hours on each machine. The factory wants to maximize total
profit subject to machine capacity constraints.

MATHEMATICAL MODEL
------------------
Sets:
    P = {0, 1, 2}       Products (A, B, C)
    M = {0, 1}          Machines

Parameters:
    profit[p]           Profit per unit of product p
    hours[m,p]          Hours required on machine m to produce one unit of p
    capacity[m]         Available hours on machine m

Variables:
    x[p] >= 0           Units of product p to produce (continuous)

Objective:
    max  sum_{p in P} profit[p] * x[p]

Constraints:
    Capacity[m]:  sum_{p in P} hours[m,p] * x[p] <= capacity[m]   for all m in M

DSL FEATURES DEMONSTRATED
-------------------------
- dsl::range()                    Create index domains
- dsl::sum(domain, lambda)        Mathematical summation notation
- VariableFactory::add()          Create variable arrays
- ConstraintFactory::addIndexed() Domain-based constraint creation
- ModelBuilder pattern            Structured model construction
- maximize() helper               Set objective with sense
- valuesWithIndex()               Solution extraction with indices
- slack(), dual()                 LP sensitivity analysis
- statusString(), modelSummary()  Diagnostics

================================================================================
*/

#include <iostream>
#include <iomanip>
#include <gurobi_dsl/dsl.h>

// ============================================================================
// TYPE-SAFE ENUM KEYS FOR VARIABLES AND CONSTRAINTS
// ============================================================================
DECLARE_ENUM_WITH_COUNT(Vars, X);
DECLARE_ENUM_WITH_COUNT(Cons, Capacity);

// ============================================================================
// PRODUCTION BUILDER - Implements the ModelBuilder Pattern
// ============================================================================
class ProductionBuilder : public dsl::ModelBuilder<Vars, Cons> {
private:
    // Problem data
    std::vector<std::string> productNames_;
    std::vector<double> profit_;
    std::vector<std::vector<double>> hours_;  // hours[machine][product]
    std::vector<double> capacity_;
    
    int nProducts_;
    int nMachines_;

public:
    ProductionBuilder(
        const std::vector<std::string>& productNames,
        const std::vector<double>& profit,
        const std::vector<std::vector<double>>& hours,
        const std::vector<double>& capacity
    ) : productNames_(productNames), profit_(profit), 
        hours_(hours), capacity_(capacity),
        nProducts_(static_cast<int>(profit.size())),
        nMachines_(static_cast<int>(capacity.size()))
    {
        // Store problem metadata for later access
        store()["n_products"] = nProducts_;
        store()["n_machines"] = nMachines_;
    }

protected:
    // ========================================================================
    // ModelBuilder Template Method Hooks
    // ========================================================================
    
    void addParameters() override {
        quiet();  // Suppress Gurobi output
    }

    void addVariables() override {
        // Create continuous variables x[p] >= 0 for each product
        auto X = dsl::VariableFactory::add(
            model(), 
            GRB_CONTINUOUS,     // Variable type
            0.0,                // Lower bound
            GRB_INFINITY,       // Upper bound (no limit)
            "x",                // Base name
            nProducts_          // Size
        );
        
        variables().set(Vars::X, std::move(X));
    }

    void addConstraints() override {
        auto& X = variables().get(Vars::X);
        auto P = dsl::range(0, nProducts_);
        auto M = dsl::range(0, nMachines_);

        // Capacity[m]: sum_{p in P} hours[m,p] * x[p] <= capacity[m]
        auto capConstrs = dsl::ConstraintFactory::addIndexed(
            model(), 
            "capacity",    // Constraint name
            M,             // Index domain
            [&](int m) {   // Constraint generator
                return dsl::sum(P, [&](int p) { 
                    return hours_[m][p] * X.at(p); 
                }) <= capacity_[m];
            }
        );
        
        constraints().set(Cons::Capacity, std::move(capConstrs));
    }

    void addObjective() override {
        auto& X = variables().get(Vars::X);
        auto P = dsl::range(0, nProducts_);

        // max sum_{p in P} profit[p] * x[p]
        maximize(dsl::sum(P, [&](int p) { 
            return profit_[p] * X.at(p); 
        }));
    }

    void afterOptimize() override {
        if (hasSolution()) {
            store()["objective"] = objVal();
            store()["runtime"] = runtime();
        }
    }

public:
    // ========================================================================
    // Public Accessors for Results
    // ========================================================================
    
    const std::vector<std::string>& productNames() const { return productNames_; }
    const std::vector<double>& profit() const { return profit_; }
    
    // Access capacity constraints for sensitivity analysis
    const dsl::IndexedConstraintSet& capacityConstraints() const {
        return constraints().get(Cons::Capacity).asIndexed();
    }
};

// ============================================================================
// MAIN PROGRAM
// ============================================================================
int main() {
    std::cout << "================================================================\n";
    std::cout << "EXAMPLE 01: Production Planning LP\n";
    std::cout << "================================================================\n\n";

    try {
        // ====================================================================
        // PROBLEM DATA
        // ====================================================================
        std::vector<std::string> productNames = {"Product A", "Product B", "Product C"};
        std::vector<double> profit = {30.0, 50.0, 40.0};  // $/unit
        
        // Hours required on each machine per unit of product
        //                    Prod A  Prod B  Prod C
        std::vector<std::vector<double>> hours = {
            {1.0, 2.0, 3.0},  // Machine 0 (Cutting)
            {2.0, 1.0, 3.0}   // Machine 1 (Assembly)
        };
        
        std::vector<double> capacity = {100.0, 80.0};  // hours available
        std::vector<std::string> machineNames = {"Cutting", "Assembly"};

        const int nProducts = 3;
        const int nMachines = 2;

        // ====================================================================
        // PRINT PROBLEM DESCRIPTION
        // ====================================================================
        std::cout << "PROBLEM DATA\n";
        std::cout << "------------\n";
        std::cout << "Products: ";
        for (int p = 0; p < nProducts; ++p) {
            std::cout << productNames[p];
            if (p < nProducts - 1) std::cout << ", ";
        }
        std::cout << "\n\n";

        std::cout << "Profit per unit:\n";
        for (int p = 0; p < nProducts; ++p) {
            std::cout << "  " << productNames[p] << ": $" << profit[p] << "\n";
        }
        std::cout << "\n";

        std::cout << "Hours required per unit:\n";
        std::cout << std::setw(12) << "";
        for (int p = 0; p < nProducts; ++p) {
            std::cout << std::setw(12) << productNames[p];
        }
        std::cout << std::setw(12) << "Capacity\n";
        
        for (int m = 0; m < nMachines; ++m) {
            std::cout << std::setw(12) << machineNames[m];
            for (int p = 0; p < nProducts; ++p) {
                std::cout << std::setw(12) << hours[m][p];
            }
            std::cout << std::setw(12) << capacity[m] << " hrs\n";
        }
        std::cout << "\n";

        // ====================================================================
        // BUILD AND SOLVE MODEL
        // ====================================================================
        std::cout << "SOLVING...\n";
        std::cout << "----------\n";
        
        ProductionBuilder builder(productNames, profit, hours, capacity);
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
            std::cout << "Maximum Profit: $" << builder.objVal() << "\n";
            std::cout << "Runtime: " << builder.runtime() << " seconds\n\n";

            // Production plan
            std::cout << "Production Plan:\n";
            auto& X = builder.variables().get(Vars::X);
            double totalProfit = 0;
            
            for (auto& [idx, val] : dsl::valuesWithIndex(X)) {
                int p = idx[0];
                double prodProfit = profit[p] * val;
                totalProfit += prodProfit;
                std::cout << "  " << std::setw(12) << productNames[p] 
                          << ": " << std::setw(8) << val << " units"
                          << " (contributes $" << prodProfit << ")\n";
            }

            // Sensitivity analysis - machine utilization
            std::cout << "\nMachine Utilization & Sensitivity:\n";
            auto& capCons = builder.capacityConstraints();
            
            for (int m = 0; m < nMachines; ++m) {
                double slk = dsl::slack(capCons.at(m));
                double pi = dsl::dual(capCons.at(m));  // Shadow price
                double used = capacity[m] - slk;
                double utilization = (used / capacity[m]) * 100.0;
                
                std::cout << "  " << std::setw(12) << machineNames[m] << ": "
                          << std::setw(6) << used << "/" << capacity[m] << " hrs"
                          << " (" << std::setw(5) << utilization << "% utilized)"
                          << ", Shadow Price: $" << pi << "/hr\n";
            }

            std::cout << "\nINTERPRETATION\n";
            std::cout << "--------------\n";
            std::cout << "Shadow prices indicate the marginal value of additional capacity.\n";
            std::cout << "A shadow price of $X/hr means profit would increase by $X\n";
            std::cout << "for each additional hour of capacity on that machine.\n";
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
