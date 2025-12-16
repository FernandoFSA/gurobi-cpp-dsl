/*
================================================================================
EXAMPLE 03: TRANSPORTATION PROBLEM - Classic Supply Chain
================================================================================
DIFFICULTY: Beginner
PROBLEM TYPE: Linear Programming (LP)

PROBLEM DESCRIPTION
-------------------
A company has multiple warehouses (supply points) and retail stores (demand 
points). Each warehouse has a limited supply, and each store has a specific
demand. The goal is to minimize total shipping cost while satisfying all
demands without exceeding warehouse supplies.

MATHEMATICAL MODEL
------------------
Sets:
    W = {0, 1, ..., m-1}    Warehouses (supply points)
    S = {0, 1, ..., n-1}    Stores (demand points)

Parameters:
    supply[w]               Units available at warehouse w
    demand[s]               Units required at store s
    cost[w,s]               Cost to ship one unit from w to s

Variables:
    x[w,s] >= 0             Units shipped from warehouse w to store s

Objective:
    min  sum_{w,s} cost[w,s] * x[w,s]

Constraints:
    Supply[w]:  sum_{s} x[w,s] <= supply[w]    for all w in W
    Demand[s]:  sum_{w} x[w,s] >= demand[s]    for all s in S

DSL FEATURES DEMONSTRATED
-------------------------
- Two-dimensional variables          X(w, s) access pattern
- Cartesian product domains          W * S
- Multiple constraint families       Supply and Demand constraints
- minimize() helper                  Objective with GRB_MINIMIZE
- sum() with 2D domain               Nested summation
- Constraint iteration               forEach() on constraint groups
- ConstraintContainer                Accessing constraints from table

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
DECLARE_ENUM_WITH_COUNT(Cons, Supply, Demand);

// ============================================================================
// TRANSPORTATION BUILDER
// ============================================================================
class TransportationBuilder : public dsl::ModelBuilder<Vars, Cons> {
private:
    std::vector<std::string> warehouseNames_;
    std::vector<std::string> storeNames_;
    std::vector<double> supply_;
    std::vector<double> demand_;
    std::vector<std::vector<double>> cost_;  // cost[warehouse][store]
    
    int nWarehouses_;
    int nStores_;

public:
    TransportationBuilder(
        const std::vector<std::string>& warehouseNames,
        const std::vector<std::string>& storeNames,
        const std::vector<double>& supply,
        const std::vector<double>& demand,
        const std::vector<std::vector<double>>& cost
    ) : warehouseNames_(warehouseNames), storeNames_(storeNames),
        supply_(supply), demand_(demand), cost_(cost),
        nWarehouses_(static_cast<int>(supply.size())),
        nStores_(static_cast<int>(demand.size()))
    {
        store()["n_warehouses"] = nWarehouses_;
        store()["n_stores"] = nStores_;
        store()["total_supply"] = 0.0;
        store()["total_demand"] = 0.0;
        
        for (double s : supply) store()["total_supply"] = store()["total_supply"].get<double>() + s;
        for (double d : demand) store()["total_demand"] = store()["total_demand"].get<double>() + d;
    }

protected:
    void addParameters() override {
        quiet();
    }

    void addVariables() override {
        // Shipping quantities x[w,s] >= 0
        auto X = dsl::VariableFactory::add(
            model(), GRB_CONTINUOUS, 0.0, GRB_INFINITY, "x", 
            nWarehouses_, nStores_  // 2D array
        );
        variables().set(Vars::X, std::move(X));
    }

    void addConstraints() override {
        auto& X = variables().get(Vars::X);
        auto W = dsl::range(0, nWarehouses_);
        auto S = dsl::range(0, nStores_);

        // Supply[w]: sum_{s} x[w,s] <= supply[w]
        auto supplyConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "supply", W,
            [&](int w) {
                return dsl::sum(S, [&](int s) { 
                    return X.at(w, s); 
                }) <= supply_[w];
            }
        );
        constraints().set(Cons::Supply, std::move(supplyConstrs));

        // Demand[s]: sum_{w} x[w,s] >= demand[s]
        auto demandConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "demand", S,
            [&](int s) {
                return dsl::sum(W, [&](int w) { 
                    return X.at(w, s); 
                }) >= demand_[s];
            }
        );
        constraints().set(Cons::Demand, std::move(demandConstrs));
    }

    void addObjective() override {
        auto& X = variables().get(Vars::X);
        auto W = dsl::range(0, nWarehouses_);
        auto S = dsl::range(0, nStores_);

        // min sum_{w,s} cost[w,s] * x[w,s]
        minimize(dsl::sum(W * S, [&](int w, int s) {
            return cost_[w][s] * X.at(w, s);
        }));
    }

    void afterOptimize() override {
        if (hasSolution()) {
            store()["total_cost"] = objVal();
            store()["runtime"] = runtime();
        }
    }

public:
    // Accessors
    const std::vector<std::string>& warehouseNames() const { return warehouseNames_; }
    const std::vector<std::string>& storeNames() const { return storeNames_; }
    int nWarehouses() const { return nWarehouses_; }
    int nStores() const { return nStores_; }
    
    const dsl::IndexedConstraintSet& supplyConstraints() const {
        return constraints().get(Cons::Supply).asIndexed();
    }
    
    const dsl::IndexedConstraintSet& demandConstraints() const {
        return constraints().get(Cons::Demand).asIndexed();
    }
};

// ============================================================================
// MAIN PROGRAM
// ============================================================================
int main() {
    std::cout << "================================================================\n";
    std::cout << "EXAMPLE 03: Transportation Problem\n";
    std::cout << "================================================================\n\n";

    try {
        // ====================================================================
        // PROBLEM DATA
        // ====================================================================
        std::vector<std::string> warehouseNames = {"Chicago", "Denver", "Atlanta"};
        std::vector<std::string> storeNames = {"Boston", "Dallas", "Phoenix", "Seattle"};
        
        // Supply at each warehouse
        std::vector<double> supply = {400, 300, 350};
        
        // Demand at each store
        std::vector<double> demand = {250, 200, 300, 200};
        
        // Shipping cost per unit from warehouse to store
        //                     Boston  Dallas  Phoenix  Seattle
        std::vector<std::vector<double>> cost = {
            {8.0, 6.0, 10.0, 9.0},     // Chicago
            {9.0, 5.0,  4.0, 7.0},     // Denver
            {7.0, 8.0,  6.0, 11.0}     // Atlanta
        };

        const int nW = static_cast<int>(supply.size());
        const int nS = static_cast<int>(demand.size());

        // ====================================================================
        // PRINT PROBLEM DESCRIPTION
        // ====================================================================
        std::cout << "PROBLEM DATA\n";
        std::cout << "------------\n";
        
        // Supply information
        std::cout << "Warehouse Supply:\n";
        double totalSupply = 0;
        for (int w = 0; w < nW; ++w) {
            std::cout << "  " << std::setw(10) << warehouseNames[w] 
                      << ": " << supply[w] << " units\n";
            totalSupply += supply[w];
        }
        std::cout << "  " << std::setw(10) << "Total" << ": " << totalSupply << " units\n\n";
        
        // Demand information
        std::cout << "Store Demand:\n";
        double totalDemand = 0;
        for (int s = 0; s < nS; ++s) {
            std::cout << "  " << std::setw(10) << storeNames[s] 
                      << ": " << demand[s] << " units\n";
            totalDemand += demand[s];
        }
        std::cout << "  " << std::setw(10) << "Total" << ": " << totalDemand << " units\n\n";
        
        // Cost matrix
        std::cout << "Shipping Cost ($/unit):\n";
        std::cout << std::setw(12) << "";
        for (int s = 0; s < nS; ++s) {
            std::cout << std::setw(10) << storeNames[s];
        }
        std::cout << "\n" << std::string(12 + nS * 10, '-') << "\n";
        
        for (int w = 0; w < nW; ++w) {
            std::cout << std::setw(12) << warehouseNames[w];
            for (int s = 0; s < nS; ++s) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(1) << cost[w][s];
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        // ====================================================================
        // BUILD AND SOLVE MODEL
        // ====================================================================
        std::cout << "SOLVING...\n";
        std::cout << "----------\n";
        
        TransportationBuilder builder(warehouseNames, storeNames, supply, demand, cost);
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
            std::cout << "Total Shipping Cost: $" << builder.objVal() << "\n\n";

            // Shipping plan
            std::cout << "Optimal Shipping Plan (units):\n";
            std::cout << std::setw(12) << "";
            for (int s = 0; s < nS; ++s) {
                std::cout << std::setw(10) << storeNames[s];
            }
            std::cout << std::setw(10) << "Shipped" << std::setw(10) << "Supply\n";
            std::cout << std::string(12 + (nS + 2) * 10, '-') << "\n";
            
            auto& X = builder.variables().get(Vars::X);
            for (int w = 0; w < nW; ++w) {
                std::cout << std::setw(12) << warehouseNames[w];
                double shipped = 0;
                for (int s = 0; s < nS; ++s) {
                    double val = dsl::value(X.at(w, s));
                    shipped += val;
                    if (val > 0.01) {
                        std::cout << std::setw(10) << val;
                    } else {
                        std::cout << std::setw(10) << "-";
                    }
                }
                std::cout << std::setw(10) << shipped << std::setw(10) << supply[w] << "\n";
            }
            
            // Demand satisfaction row
            std::cout << std::string(12 + (nS + 2) * 10, '-') << "\n";
            std::cout << std::setw(12) << "Received";
            for (int s = 0; s < nS; ++s) {
                double received = 0;
                for (int w = 0; w < nW; ++w) {
                    received += dsl::value(X.at(w, s));
                }
                std::cout << std::setw(10) << received;
            }
            std::cout << "\n";
            
            std::cout << std::setw(12) << "Demand";
            for (int s = 0; s < nS; ++s) {
                std::cout << std::setw(10) << demand[s];
            }
            std::cout << "\n\n";

            // Sensitivity analysis
            std::cout << "SENSITIVITY ANALYSIS (Shadow Prices)\n";
            std::cout << "------------------------------------\n";
            
            std::cout << "Supply Constraints (value of extra unit at warehouse):\n";
            auto& supCons = builder.supplyConstraints();
            for (int w = 0; w < nW; ++w) {
                double pi = dsl::dual(supCons.at(w));
                double slack = dsl::slack(supCons.at(w));
                std::cout << "  " << std::setw(10) << warehouseNames[w] 
                          << ": Shadow Price = $" << std::setw(6) << -pi  // Negative for <= constraint
                          << ", Unused = " << slack << " units\n";
            }
            
            std::cout << "\nDemand Constraints (cost of extra unit to store):\n";
            auto& demCons = builder.demandConstraints();
            for (int s = 0; s < nS; ++s) {
                double pi = dsl::dual(demCons.at(s));
                std::cout << "  " << std::setw(10) << storeNames[s] 
                          << ": Shadow Price = $" << std::setw(6) << pi << "\n";
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
