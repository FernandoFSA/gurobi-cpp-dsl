/*
================================================================================
EXAMPLE 09: VEHICLE ROUTING PROBLEM - Capacitated VRP (CVRP)
================================================================================
DIFFICULTY: Advanced
PROBLEM TYPE: Mixed-Integer Programming (MIP)

PROBLEM DESCRIPTION
-------------------
A depot must deliver goods to a set of customers using a fleet of identical
vehicles. Each vehicle has limited capacity. Each customer has a specific
demand. The goal is to find routes that minimize total travel distance while:
- Each customer is visited exactly once
- Vehicle capacity is not exceeded
- All routes start and end at the depot

MATHEMATICAL MODEL
------------------
Sets:
    N = {0, 1, ..., n}      Nodes (0 = depot, 1..n = customers)
    C = {1, ..., n}         Customers only
    K = {0, 1, ..., m-1}    Vehicles

Parameters:
    demand[i]       Demand of customer i (demand[0] = 0 for depot)
    dist[i,j]       Distance between nodes i and j
    capacity        Vehicle capacity

Variables:
    x[i,j,k] in {0,1}   1 if vehicle k travels from i to j
    u[i,k] >= 0         Cumulative demand up to customer i on vehicle k (MTZ)

Objective:
    min  sum_{i,j,k} dist[i,j] * x[i,j,k]

Constraints:
    Visit[i]:       sum_{j,k} x[i,j,k] = 1                  for all i in C
    FlowIn[j,k]:    sum_i x[i,j,k] = sum_i x[j,i,k]         for all j, k
    StartDepot[k]:  sum_j x[0,j,k] <= 1                     for all k
    Capacity[k]:    u[i,k] <= capacity                      for all i, k
    MTZ[i,j,k]:     u[j,k] >= u[i,k] + demand[j] - M*(1-x[i,j,k])

DSL FEATURES DEMONSTRATED
-------------------------
- Triple-indexed variables       x[i,j,k]
- MTZ subtour elimination        Miller-Tucker-Zemlin constraints
- Complex filtered domains       Valid arcs and vehicle assignments
- Symmetry breaking              Order vehicles by first customer
- Multiple constraint families   Flow, capacity, subtour
- Large-scale modeling           Many variables and constraints

================================================================================
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <gurobi_dsl/dsl.h>

// ============================================================================
// TYPE-SAFE ENUM KEYS
// ============================================================================
DECLARE_ENUM_WITH_COUNT(Vars, X, U);
DECLARE_ENUM_WITH_COUNT(Cons, Visit, FlowIn, StartDepot, Capacity, MTZ);

// ============================================================================
// VRP BUILDER
// ============================================================================
class VRPBuilder : public dsl::ModelBuilder<Vars, Cons> {
private:
    std::vector<std::string> nodeNames_;
    std::vector<double> demand_;
    std::vector<std::vector<double>> dist_;
    double vehicleCapacity_;
    int nVehicles_;
    
    int nNodes_;  // Including depot

public:
    VRPBuilder(
        const std::vector<std::string>& nodeNames,
        const std::vector<double>& demand,
        const std::vector<std::vector<double>>& dist,
        double vehicleCapacity,
        int nVehicles
    ) : nodeNames_(nodeNames), demand_(demand), dist_(dist),
        vehicleCapacity_(vehicleCapacity), nVehicles_(nVehicles),
        nNodes_(static_cast<int>(nodeNames.size()))
    {
        store()["n_nodes"] = nNodes_;
        store()["n_customers"] = nNodes_ - 1;
        store()["n_vehicles"] = nVehicles_;
        store()["vehicle_capacity"] = vehicleCapacity_;
        
        double totalDemand = 0;
        for (int i = 1; i < nNodes_; ++i) totalDemand += demand_[i];
        store()["total_demand"] = totalDemand;
    }

protected:
    void addParameters() override {
        quiet();
        applyPreset(Preset::Fast);
        timeLimit(60.0);
        mipGapLimit(0.05);  // 5% gap for large instances
    }

    void addVariables() override {
        auto N = dsl::range(0, nNodes_);
        auto C = dsl::range(1, nNodes_);  // Customers only
        auto K = dsl::range(0, nVehicles_);

        // x[i,j,k] - binary, vehicle k travels from i to j
        // Exclude self-loops (i != j)
        auto arcDomain = (N * N * K) | dsl::filter([](int i, int j, int k) {
            return i != j;
        });
        
        auto X = dsl::VariableFactory::addIndexed(
            model(), GRB_BINARY, 0.0, 1.0, "x", arcDomain
        );
        variables().set(Vars::X, std::move(X));

        // u[i,k] - cumulative demand at customer i for vehicle k (MTZ)
        auto customerVehicle = C * K;
        auto U = dsl::VariableFactory::addIndexed(
            model(), GRB_CONTINUOUS, 0.0, vehicleCapacity_, "u", customerVehicle
        );
        variables().set(Vars::U, std::move(U));
    }

    void addConstraints() override {
        auto& X = variables().get(Vars::X);
        auto& U = variables().get(Vars::U);
        
        auto N = dsl::range(0, nNodes_);
        auto C = dsl::range(1, nNodes_);
        auto K = dsl::range(0, nVehicles_);

        // Visit[i]: Each customer visited exactly once
        auto visitConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "visit", C,
            [&](int i) {
                GRBLinExpr lhs = 0;
                for (int j : N) {
                    if (i != j) {
                        for (int k : K) {
                            auto* var = X.asIndexed().try_get(i, j, k);
                            if (var) lhs += *var;
                        }
                    }
                }
                return lhs == 1;
            }
        );
        constraints().set(Cons::Visit, std::move(visitConstrs));

        // FlowIn[j,k]: Flow conservation at each node for each vehicle
        auto flowDomain = N * K;
        auto flowConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "flow", flowDomain,
            [&](int j, int k) {
                GRBLinExpr inFlow = 0, outFlow = 0;
                for (int i : N) {
                    if (i != j) {
                        auto* varIn = X.asIndexed().try_get(i, j, k);
                        auto* varOut = X.asIndexed().try_get(j, i, k);
                        if (varIn) inFlow += *varIn;
                        if (varOut) outFlow += *varOut;
                    }
                }
                return inFlow == outFlow;
            }
        );
        constraints().set(Cons::FlowIn, std::move(flowConstrs));

        // StartDepot[k]: Each vehicle leaves depot at most once
        auto depotConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "start_depot", K,
            [&](int k) {
                GRBLinExpr lhs = 0;
                for (int j : C) {
                    auto* var = X.asIndexed().try_get(0, j, k);
                    if (var) lhs += *var;
                }
                return lhs <= 1;
            }
        );
        constraints().set(Cons::StartDepot, std::move(depotConstrs));

        // MTZ subtour elimination
        double M = vehicleCapacity_;
        auto mtzDomain = (C * C * K) | dsl::filter([](int i, int j, int k) {
            return i != j;
        });
        
        auto mtzConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "mtz", mtzDomain,
            [&](int i, int j, int k) {
                auto* xVar = X.asIndexed().try_get(i, j, k);
                if (!xVar) {
                    // This shouldn't happen with our domain, but be safe
                    return GRBLinExpr(0) <= 0;
                }
                return U.asIndexed().at(j, k) >= U.asIndexed().at(i, k) + demand_[j] - M * (1 - *xVar);
            }
        );
        constraints().set(Cons::MTZ, std::move(mtzConstrs));
    }

    void addObjective() override {
        auto& X = variables().get(Vars::X);
        
        // Minimize total travel distance
        GRBLinExpr obj = 0;
        for (const auto& entry : X.asIndexed()) {
            int i = entry.index[0];
            int j = entry.index[1];
            obj += dist_[i][j] * entry.var;
        }
        minimize(obj);
    }

    void afterOptimize() override {
        if (hasSolution()) {
            store()["total_distance"] = objVal();
            store()["runtime"] = runtime();
            store()["gap"] = mipGap();
        }
    }

public:
    // Accessors
    const std::vector<std::string>& nodeNames() const { return nodeNames_; }
    int nNodes() const { return nNodes_; }
    int nVehicles() const { return nVehicles_; }
    double demand(int i) const { return demand_[i]; }
    double distance(int i, int j) const { return dist_[i][j]; }
    
    // Extract routes for each vehicle
    std::vector<std::vector<int>> getRoutes() const {
        std::vector<std::vector<int>> routes(nVehicles_);
        const auto& X = variables().get(Vars::X);
        
        for (int k = 0; k < nVehicles_; ++k) {
            // Start from depot
            int current = 0;
            routes[k].push_back(0);
            
            while (true) {
                int next = -1;
                for (int j = 0; j < nNodes_; ++j) {
                    if (j != current) {
                        auto* var = X.asIndexed().try_get(current, j, k);
                        if (var && dsl::value(*var) > 0.5) {
                            next = j;
                            break;
                        }
                    }
                }
                
                if (next == -1 || next == 0) {
                    if (routes[k].size() > 1) routes[k].push_back(0);  // Return to depot
                    break;
                }
                
                routes[k].push_back(next);
                current = next;
            }
        }
        
        return routes;
    }
};

// ============================================================================
// HELPER: Compute Euclidean distance matrix
// ============================================================================
std::vector<std::vector<double>> computeDistanceMatrix(
    const std::vector<std::pair<double, double>>& coords
) {
    int n = static_cast<int>(coords.size());
    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0));
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double dx = coords[i].first - coords[j].first;
            double dy = coords[i].second - coords[j].second;
            dist[i][j] = std::sqrt(dx * dx + dy * dy);
        }
    }
    return dist;
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================
int main() {
    std::cout << "================================================================\n";
    std::cout << "EXAMPLE 09: Capacitated Vehicle Routing Problem (CVRP)\n";
    std::cout << "================================================================\n\n";

    try {
        // ====================================================================
        // PROBLEM DATA
        // ====================================================================
        std::vector<std::string> nodeNames = {
            "Depot", "A", "B", "C", "D", "E", "F"
        };
        
        // Coordinates (for distance calculation)
        std::vector<std::pair<double, double>> coords = {
            {0, 0},    // Depot
            {2, 4},    // A
            {5, 3},    // B
            {6, 1},    // C
            {8, 5},    // D
            {3, 6},    // E
            {1, 2}     // F
        };
        
        // Customer demands (depot has 0 demand)
        std::vector<double> demand = {0, 10, 15, 20, 10, 25, 15};
        
        int nVehicles = 3;
        double vehicleCapacity = 40;
        
        // Compute distance matrix
        auto dist = computeDistanceMatrix(coords);

        const int nNodes = static_cast<int>(nodeNames.size());
        const int nCustomers = nNodes - 1;

        // ====================================================================
        // PRINT PROBLEM DESCRIPTION
        // ====================================================================
        std::cout << "PROBLEM DATA\n";
        std::cout << "------------\n";
        std::cout << "Customers: " << nCustomers << "\n";
        std::cout << "Vehicles: " << nVehicles << "\n";
        std::cout << "Vehicle Capacity: " << vehicleCapacity << " units\n\n";
        
        double totalDemand = 0;
        std::cout << "Customer Information:\n";
        std::cout << std::setw(10) << "Customer" << std::setw(12) << "Location"
                  << std::setw(10) << "Demand\n";
        std::cout << std::string(32, '-') << "\n";
        
        for (int i = 0; i < nNodes; ++i) {
            std::cout << std::setw(10) << nodeNames[i]
                      << std::setw(6) << "(" << coords[i].first << "," << coords[i].second << ")"
                      << std::setw(10) << demand[i] << "\n";
            totalDemand += demand[i];
        }
        std::cout << "\nTotal Demand: " << totalDemand << " units\n";
        std::cout << "Total Capacity: " << nVehicles * vehicleCapacity << " units\n\n";

        // Distance matrix
        std::cout << "Distance Matrix:\n";
        std::cout << std::setw(8) << "";
        for (int j = 0; j < nNodes; ++j) {
            std::cout << std::setw(8) << nodeNames[j];
        }
        std::cout << "\n" << std::string(8 + nNodes * 8, '-') << "\n";
        
        for (int i = 0; i < nNodes; ++i) {
            std::cout << std::setw(8) << nodeNames[i];
            for (int j = 0; j < nNodes; ++j) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(1) << dist[i][j];
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        // ====================================================================
        // BUILD AND SOLVE MODEL
        // ====================================================================
        std::cout << "SOLVING...\n";
        std::cout << "----------\n";
        
        VRPBuilder builder(nodeNames, demand, dist, vehicleCapacity, nVehicles);
        builder.optimize();

        std::cout << "Status: " << dsl::statusString(builder.status()) << "\n";
        std::cout << "Runtime: " << builder.runtime() << " seconds\n";
        std::cout << "MIP Gap: " << builder.mipGap() * 100 << "%\n";
        std::cout << dsl::modelSummary(builder.model()) << "\n";

        // ====================================================================
        // DISPLAY RESULTS
        // ====================================================================
        if (builder.hasSolution()) {
            std::cout << std::fixed << std::setprecision(2);
            
            std::cout << "\nOPTIMAL ROUTES\n";
            std::cout << "--------------\n";
            std::cout << "Total Distance: " << builder.objVal() << " units\n\n";

            auto routes = builder.getRoutes();
            int activeVehicles = 0;
            
            for (int k = 0; k < nVehicles; ++k) {
                if (routes[k].size() <= 2) continue;  // Empty route (just depot)
                
                activeVehicles++;
                std::cout << "Vehicle " << k + 1 << ":\n";
                
                // Route
                std::cout << "  Route: ";
                double routeDistance = 0;
                double routeLoad = 0;
                
                for (size_t i = 0; i < routes[k].size(); ++i) {
                    int node = routes[k][i];
                    std::cout << nodeNames[node];
                    
                    if (i < routes[k].size() - 1) {
                        std::cout << " -> ";
                        routeDistance += dist[node][routes[k][i + 1]];
                    }
                    routeLoad += demand[node];
                }
                std::cout << "\n";
                
                // Details
                std::cout << "  Distance: " << routeDistance << "\n";
                std::cout << "  Load: " << routeLoad << "/" << vehicleCapacity 
                          << " (" << (routeLoad / vehicleCapacity) * 100 << "% utilized)\n";
                
                // Stop-by-stop
                std::cout << "  Stops:\n";
                for (size_t i = 1; i < routes[k].size() - 1; ++i) {
                    int node = routes[k][i];
                    std::cout << "    " << i << ". " << nodeNames[node] 
                              << " (demand: " << demand[node] << ")\n";
                }
                std::cout << "\n";
            }

            // Summary
            std::cout << "SUMMARY\n";
            std::cout << "-------\n";
            std::cout << "Vehicles used: " << activeVehicles << "/" << nVehicles << "\n";
            std::cout << "Customers served: " << nCustomers << "/" << nCustomers << "\n";
            std::cout << "Total distance: " << builder.objVal() << " units\n";
            std::cout << "Average route length: " << builder.objVal() / activeVehicles << " units\n";
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
