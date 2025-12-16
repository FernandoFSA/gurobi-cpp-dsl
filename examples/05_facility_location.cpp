/*
================================================================================
EXAMPLE 05: FACILITY LOCATION - Capacitated Facility Location Problem
================================================================================
DIFFICULTY: Intermediate
PROBLEM TYPE: Mixed-Integer Programming (MIP)

PROBLEM DESCRIPTION
-------------------
A company wants to decide which potential facility locations to open and how
to assign customers to facilities. Each facility has a fixed opening cost and
a capacity limit. Transportation costs depend on the distance from facility
to customer. The goal is to minimize total cost (opening + transportation)
while meeting all customer demands.

MATHEMATICAL MODEL
------------------
Sets:
    F = {0, 1, ..., m-1}    Potential facility locations
    C = {0, 1, ..., n-1}    Customers

Parameters:
    fixedCost[f]        Cost to open facility f
    capacity[f]         Maximum demand facility f can serve
    demand[c]           Demand of customer c
    transCost[f,c]      Cost to serve one unit of demand from f to c

Variables:
    y[f] in {0,1}       1 if facility f is opened
    x[f,c] >= 0         Fraction of customer c's demand served by facility f

Objective:
    min  sum_f fixedCost[f]*y[f] + sum_{f,c} transCost[f,c]*demand[c]*x[f,c]

Constraints:
    Demand[c]:     sum_f x[f,c] = 1                      for all c in C
    Capacity[f]:   sum_c demand[c]*x[f,c] <= capacity[f]*y[f]    for all f in F
    Linking[f,c]:  x[f,c] <= y[f]                        for all f,c

DSL FEATURES DEMONSTRATED
-------------------------
- Multiple variable groups        Binary y[f] and continuous x[f,c]
- Linking constraints             x <= y (big-M alternative)
- Capacity with binary variables  capacity * y relationship
- setStart()                      Warm-start hints
- Callback integration ready      MIPCallback potential
- Solution pool access            Multiple near-optimal solutions

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
DECLARE_ENUM_WITH_COUNT(Vars, Y, X);
DECLARE_ENUM_WITH_COUNT(Cons, Demand, Capacity, Linking);

// ============================================================================
// FACILITY LOCATION BUILDER
// ============================================================================
class FacilityLocationBuilder : public dsl::ModelBuilder<Vars, Cons> {
private:
    std::vector<std::string> facilityNames_;
    std::vector<std::string> customerNames_;
    std::vector<double> fixedCost_;
    std::vector<double> capacity_;
    std::vector<double> demand_;
    std::vector<std::vector<double>> transCost_;  // transCost[facility][customer]
    
    int nFacilities_;
    int nCustomers_;

public:
    FacilityLocationBuilder(
        const std::vector<std::string>& facilityNames,
        const std::vector<std::string>& customerNames,
        const std::vector<double>& fixedCost,
        const std::vector<double>& capacity,
        const std::vector<double>& demand,
        const std::vector<std::vector<double>>& transCost
    ) : facilityNames_(facilityNames), customerNames_(customerNames),
        fixedCost_(fixedCost), capacity_(capacity), demand_(demand),
        transCost_(transCost),
        nFacilities_(static_cast<int>(facilityNames.size())),
        nCustomers_(static_cast<int>(customerNames.size()))
    {
        store()["n_facilities"] = nFacilities_;
        store()["n_customers"] = nCustomers_;
        store()["total_demand"] = 0.0;
        for (double d : demand) {
            store()["total_demand"] = store()["total_demand"].get<double>() + d;
        }
    }

protected:
    void addParameters() override {
        quiet();
        applyPreset(Preset::Fast);
        timeLimit(60.0);
    }

    void addVariables() override {
        // y[f] - binary, 1 if facility f is opened
        auto Y = dsl::VariableFactory::add(
            model(), GRB_BINARY, 0.0, 1.0, "y", nFacilities_
        );
        variables().set(Vars::Y, std::move(Y));

        // x[f,c] - continuous, fraction of customer c served by facility f
        auto X = dsl::VariableFactory::add(
            model(), GRB_CONTINUOUS, 0.0, 1.0, "x", nFacilities_, nCustomers_
        );
        variables().set(Vars::X, std::move(X));
    }

    void addConstraints() override {
        auto& Y = variables().get(Vars::Y);
        auto& X = variables().get(Vars::X);
        auto F = dsl::range(0, nFacilities_);
        auto C = dsl::range(0, nCustomers_);

        // Demand[c]: Each customer's demand must be fully satisfied
        auto demandConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "demand", C,
            [&](int c) {
                return dsl::sum(F, [&](int f) { 
                    return X.at(f, c); 
                }) == 1;
            }
        );
        constraints().set(Cons::Demand, std::move(demandConstrs));

        // Capacity[f]: Facility capacity constraint
        auto capacityConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "capacity", F,
            [&](int f) {
                return dsl::sum(C, [&](int c) { 
                    return demand_[c] * X.at(f, c); 
                }) <= capacity_[f] * Y.at(f);
            }
        );
        constraints().set(Cons::Capacity, std::move(capacityConstrs));

        // Linking[f,c]: Can only serve from open facilities
        auto linkingConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "linking", F * C,
            [&](int f, int c) {
                return X.at(f, c) <= Y.at(f);
            }
        );
        constraints().set(Cons::Linking, std::move(linkingConstrs));
    }

    void addObjective() override {
        auto& Y = variables().get(Vars::Y);
        auto& X = variables().get(Vars::X);
        auto F = dsl::range(0, nFacilities_);
        auto C = dsl::range(0, nCustomers_);

        // Fixed costs
        GRBLinExpr fixedCosts = dsl::sum(F, [&](int f) {
            return fixedCost_[f] * Y.at(f);
        });

        // Transportation costs
        GRBLinExpr transCosts = dsl::sum(F * C, [&](int f, int c) {
            return transCost_[f][c] * demand_[c] * X.at(f, c);
        });

        minimize(fixedCosts + transCosts);
    }

    void afterOptimize() override {
        if (hasSolution()) {
            store()["total_cost"] = objVal();
            store()["runtime"] = runtime();
            
            // Compute cost breakdown
            auto& Y = variables().get(Vars::Y);
            auto& X = variables().get(Vars::X);
            
            double fixedTotal = 0;
            int openCount = 0;
            for (int f = 0; f < nFacilities_; ++f) {
                if (dsl::value(Y.at(f)) > 0.5) {
                    fixedTotal += fixedCost_[f];
                    openCount++;
                }
            }
            
            store()["fixed_cost"] = fixedTotal;
            store()["transport_cost"] = objVal() - fixedTotal;
            store()["facilities_opened"] = openCount;
        }
    }

public:
    // Accessors
    const std::vector<std::string>& facilityNames() const { return facilityNames_; }
    const std::vector<std::string>& customerNames() const { return customerNames_; }
    int nFacilities() const { return nFacilities_; }
    int nCustomers() const { return nCustomers_; }
    
    // Get opened facilities
    std::vector<int> getOpenFacilities() const {
        std::vector<int> open;
        const auto& Y = variables().get(Vars::Y);
        for (int f = 0; f < nFacilities_; ++f) {
            if (dsl::value(Y.at(f)) > 0.5) {
                open.push_back(f);
            }
        }
        return open;
    }
    
    // Get assignments
    std::vector<std::tuple<int, int, double>> getAssignments() const {
        std::vector<std::tuple<int, int, double>> assignments;
        const auto& X = variables().get(Vars::X);
        
        for (int f = 0; f < nFacilities_; ++f) {
            for (int c = 0; c < nCustomers_; ++c) {
                double val = dsl::value(X.at(f, c));
                if (val > 0.01) {
                    assignments.emplace_back(f, c, val);
                }
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
    std::cout << "EXAMPLE 05: Capacitated Facility Location Problem\n";
    std::cout << "================================================================\n\n";

    try {
        // ====================================================================
        // PROBLEM DATA
        // ====================================================================
        std::vector<std::string> facilityNames = {
            "Atlanta", "Boston", "Chicago", "Denver"
        };
        
        std::vector<std::string> customerNames = {
            "Customer1", "Customer2", "Customer3", "Customer4", 
            "Customer5", "Customer6"
        };

        // Fixed cost to open each facility
        std::vector<double> fixedCost = {1000, 1200, 1100, 900};
        
        // Capacity of each facility
        std::vector<double> capacity = {500, 400, 600, 450};
        
        // Demand of each customer
        std::vector<double> demand = {100, 150, 120, 80, 200, 130};
        
        // Transportation cost per unit from facility to customer
        //                     C1    C2    C3    C4    C5    C6
        std::vector<std::vector<double>> transCost = {
            {4.0, 5.0, 6.0, 3.0, 7.0, 5.0},   // Atlanta
            {6.0, 3.0, 4.0, 7.0, 5.0, 6.0},   // Boston
            {5.0, 4.0, 3.0, 5.0, 4.0, 4.0},   // Chicago
            {7.0, 6.0, 5.0, 4.0, 3.0, 5.0}    // Denver
        };

        const int nF = static_cast<int>(facilityNames.size());
        const int nC = static_cast<int>(customerNames.size());

        // ====================================================================
        // PRINT PROBLEM DESCRIPTION
        // ====================================================================
        std::cout << "PROBLEM DATA\n";
        std::cout << "------------\n";
        
        std::cout << "Potential Facilities:\n";
        std::cout << std::setw(12) << "Location" << std::setw(15) << "Fixed Cost" 
                  << std::setw(12) << "Capacity\n";
        std::cout << std::string(39, '-') << "\n";
        for (int f = 0; f < nF; ++f) {
            std::cout << std::setw(12) << facilityNames[f] 
                      << std::setw(15) << ("$" + std::to_string(static_cast<int>(fixedCost[f])))
                      << std::setw(12) << capacity[f] << "\n";
        }
        std::cout << "\n";
        
        std::cout << "Customer Demands:\n";
        double totalDemand = 0;
        for (int c = 0; c < nC; ++c) {
            std::cout << "  " << std::setw(12) << customerNames[c] << ": " << demand[c] << " units\n";
            totalDemand += demand[c];
        }
        std::cout << "  " << std::setw(12) << "Total" << ": " << totalDemand << " units\n\n";
        
        std::cout << "Transportation Cost ($/unit):\n";
        std::cout << std::setw(12) << "";
        for (int c = 0; c < nC; ++c) {
            std::cout << std::setw(10) << customerNames[c].substr(0, 8);
        }
        std::cout << "\n" << std::string(12 + nC * 10, '-') << "\n";
        for (int f = 0; f < nF; ++f) {
            std::cout << std::setw(12) << facilityNames[f];
            for (int c = 0; c < nC; ++c) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(1) << transCost[f][c];
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        // ====================================================================
        // BUILD AND SOLVE MODEL
        // ====================================================================
        std::cout << "SOLVING...\n";
        std::cout << "----------\n";
        
        FacilityLocationBuilder builder(facilityNames, customerNames, 
                                        fixedCost, capacity, demand, transCost);
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
            std::cout << "Total Cost: $" << builder.objVal() << "\n";
            std::cout << "  Fixed Costs: $" << builder.store()["fixed_cost"].get<double>() << "\n";
            std::cout << "  Transport Costs: $" << builder.store()["transport_cost"].get<double>() << "\n";
            std::cout << "Facilities Opened: " << builder.store()["facilities_opened"].get<int>() 
                      << " / " << nF << "\n\n";

            // Show opened facilities
            std::cout << "Opened Facilities:\n";
            for (int f : builder.getOpenFacilities()) {
                std::cout << "  [OPEN] " << std::setw(12) << facilityNames[f] 
                          << " (Fixed Cost: $" << fixedCost[f] 
                          << ", Capacity: " << capacity[f] << ")\n";
            }
            std::cout << "\n";
            
            // Show closed facilities
            auto openFacs = builder.getOpenFacilities();
            for (int f = 0; f < nF; ++f) {
                if (std::find(openFacs.begin(), openFacs.end(), f) == openFacs.end()) {
                    std::cout << "  [CLOSED] " << facilityNames[f] << "\n";
                }
            }
            std::cout << "\n";

            // Show customer assignments
            std::cout << "Customer Assignments:\n";
            auto assignments = builder.getAssignments();
            
            // Group by customer
            for (int c = 0; c < nC; ++c) {
                std::cout << "  " << std::setw(12) << customerNames[c] << " (demand=" << demand[c] << "):\n";
                for (auto [f, cust, frac] : assignments) {
                    if (cust == c) {
                        double served = frac * demand[c];
                        double cost = transCost[f][c] * served;
                        std::cout << "      <- " << std::setw(10) << facilityNames[f]
                                  << ": " << std::setw(6) << served << " units"
                                  << " (transport: $" << cost << ")\n";
                    }
                }
            }
            std::cout << "\n";

            // Facility utilization
            std::cout << "Facility Utilization:\n";
            auto& X = builder.variables().get(Vars::X);
            for (int f : builder.getOpenFacilities()) {
                double used = 0;
                for (int c = 0; c < nC; ++c) {
                    used += demand[c] * dsl::value(X.at(f, c));
                }
                double utilization = (used / capacity[f]) * 100;
                std::cout << "  " << std::setw(12) << facilityNames[f] << ": "
                          << std::setw(6) << used << " / " << capacity[f] << " units"
                          << " (" << std::setw(5) << utilization << "% utilized)\n";
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
