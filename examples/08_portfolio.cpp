/*
================================================================================
EXAMPLE 08: PORTFOLIO OPTIMIZATION - Mean-Variance with Quadratic Programming
================================================================================
DIFFICULTY: Advanced
PROBLEM TYPE: Quadratic Programming (QP)

PROBLEM DESCRIPTION
-------------------
An investor wants to allocate capital across multiple assets to maximize
expected return while limiting portfolio risk (variance). This is the classic
Markowitz mean-variance portfolio optimization problem.

The problem includes:
- Minimum return requirement
- Maximum allocation per asset (diversification)
- No short selling (all allocations >= 0)
- Budget constraint (allocations sum to 1)

MATHEMATICAL MODEL
------------------
Sets:
    A = {0, 1, ..., n-1}    Assets

Parameters:
    mu[a]               Expected return of asset a
    sigma[a,b]          Covariance between assets a and b
    minReturn           Minimum required expected return
    maxAllocation       Maximum fraction in any single asset

Variables:
    x[a] >= 0           Fraction of portfolio in asset a

Objective:
    min  sum_{a,b} sigma[a,b] * x[a] * x[b]   (minimize variance)

Constraints:
    Budget:         sum_a x[a] = 1
    MinReturn:      sum_a mu[a] * x[a] >= minReturn
    MaxAllocation:  x[a] <= maxAllocation      for all a

DSL FEATURES DEMONSTRATED
-------------------------
- Quadratic objective          GRBQuadExpr construction
- Covariance matrix handling   Symmetric matrix operations
- Parameter sensitivity        Efficient frontier exploration
- Risk-return trade-offs       Multiple runs with varying targets
- addQConstr() quadratic       Quadratic constraint capability

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
DECLARE_ENUM_WITH_COUNT(Cons, Budget, MinReturn, MaxAlloc);

// ============================================================================
// PORTFOLIO BUILDER
// ============================================================================
class PortfolioBuilder : public dsl::ModelBuilder<Vars, Cons> {
private:
    std::vector<std::string> assetNames_;
    std::vector<double> expectedReturn_;
    std::vector<std::vector<double>> covariance_;
    double minReturn_;
    double maxAllocation_;
    
    int nAssets_;

public:
    PortfolioBuilder(
        const std::vector<std::string>& assetNames,
        const std::vector<double>& expectedReturn,
        const std::vector<std::vector<double>>& covariance,
        double minReturn = 0.0,
        double maxAllocation = 1.0
    ) : assetNames_(assetNames), expectedReturn_(expectedReturn),
        covariance_(covariance), minReturn_(minReturn), maxAllocation_(maxAllocation),
        nAssets_(static_cast<int>(assetNames.size()))
    {
        store()["n_assets"] = nAssets_;
        store()["min_return_target"] = minReturn;
        store()["max_allocation"] = maxAllocation;
    }

protected:
    void addParameters() override {
        quiet();
    }

    void addVariables() override {
        // Portfolio weights x[a] in [0, maxAllocation]
        auto X = dsl::VariableFactory::add(
            model(), GRB_CONTINUOUS, 0.0, maxAllocation_, "x", nAssets_
        );
        variables().set(Vars::X, std::move(X));
    }

    void addConstraints() override {
        auto& X = variables().get(Vars::X);
        auto A = dsl::range(0, nAssets_);

        // Budget: sum_a x[a] = 1 (fully invested)
        auto budgetConstr = dsl::ConstraintFactory::add(
            model(), "budget",
            [&](const std::vector<int>&) {
                return dsl::sum(A, [&](int a) { return X.at(a); }) == 1;
            }
        );
        constraints().set(Cons::Budget, std::move(budgetConstr));

        // MinReturn: sum_a mu[a] * x[a] >= minReturn
        if (minReturn_ > 0) {
            auto returnConstr = dsl::ConstraintFactory::add(
                model(), "min_return",
                [&](const std::vector<int>&) {
                    return dsl::sum(A, [&](int a) { 
                        return expectedReturn_[a] * X.at(a); 
                    }) >= minReturn_;
                }
            );
            constraints().set(Cons::MinReturn, std::move(returnConstr));
        }
    }

    void addObjective() override {
        auto& X = variables().get(Vars::X);
        auto A = dsl::range(0, nAssets_);
        
        // Minimize portfolio variance: sum_{a,b} sigma[a,b] * x[a] * x[b]
        // Using quadSum for QP objectives
        auto variance = dsl::quadSum(A * A, [&](int a, int b) {
            return covariance_[a][b] * X.at(a) * X.at(b);
        });
        
        model().setObjective(variance, GRB_MINIMIZE);
    }

    void afterOptimize() override {
        if (hasSolution()) {
            store()["variance"] = objVal();
            store()["std_dev"] = std::sqrt(objVal());
            store()["runtime"] = runtime();
            
            // Compute expected return
            auto& X = variables().get(Vars::X);
            double expReturn = 0;
            for (int a = 0; a < nAssets_; ++a) {
                expReturn += expectedReturn_[a] * dsl::value(X.at(a));
            }
            store()["expected_return"] = expReturn;
            
            // Sharpe ratio (assuming risk-free rate = 0)
            double sharpe = expReturn / std::sqrt(objVal());
            store()["sharpe_ratio"] = sharpe;
        }
    }

public:
    // Accessors
    const std::vector<std::string>& assetNames() const { return assetNames_; }
    int nAssets() const { return nAssets_; }
    double expectedReturn(int a) const { return expectedReturn_[a]; }
    
    // Get portfolio allocation
    std::vector<double> getAllocation() const {
        std::vector<double> alloc;
        const auto& X = variables().get(Vars::X);
        for (int a = 0; a < nAssets_; ++a) {
            alloc.push_back(dsl::value(X.at(a)));
        }
        return alloc;
    }
    
    // Change minimum return target and re-solve (for efficient frontier)
    void setMinReturn(double target) {
        minReturn_ = target;
        store()["min_return_target"] = target;
    }
};

// ============================================================================
// HELPER: Compute efficient frontier
// ============================================================================
struct FrontierPoint {
    double targetReturn;
    double expectedReturn;
    double variance;
    double stdDev;
    double sharpeRatio;
    std::vector<double> allocation;
};

std::vector<FrontierPoint> computeEfficientFrontier(
    const std::vector<std::string>& assetNames,
    const std::vector<double>& expectedReturn,
    const std::vector<std::vector<double>>& covariance,
    double maxAllocation,
    int numPoints = 10
) {
    std::vector<FrontierPoint> frontier;
    
    // Find min and max achievable returns
    double minRet = *std::min_element(expectedReturn.begin(), expectedReturn.end());
    double maxRet = *std::max_element(expectedReturn.begin(), expectedReturn.end());
    
    // Adjust for diversification constraint
    maxRet = std::min(maxRet, maxAllocation * maxRet + (1 - maxAllocation) * minRet);
    
    for (int i = 0; i < numPoints; ++i) {
        double target = minRet + (maxRet - minRet) * i / (numPoints - 1);
        
        PortfolioBuilder builder(assetNames, expectedReturn, covariance, target, maxAllocation);
        builder.optimize();
        
        if (builder.hasSolution()) {
            FrontierPoint point;
            point.targetReturn = target;
            point.expectedReturn = builder.store()["expected_return"].get<double>();
            point.variance = builder.store()["variance"].get<double>();
            point.stdDev = builder.store()["std_dev"].get<double>();
            point.sharpeRatio = builder.store()["sharpe_ratio"].get<double>();
            point.allocation = builder.getAllocation();
            frontier.push_back(point);
        }
    }
    
    return frontier;
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================
int main() {
    std::cout << "================================================================\n";
    std::cout << "EXAMPLE 08: Portfolio Optimization (Markowitz Mean-Variance)\n";
    std::cout << "================================================================\n\n";

    try {
        // ====================================================================
        // PROBLEM DATA
        // ====================================================================
        std::vector<std::string> assetNames = {
            "Stocks", "Bonds", "Real Estate", "Gold", "Cash"
        };
        
        // Expected annual returns (as decimals)
        std::vector<double> expectedReturn = {0.10, 0.04, 0.08, 0.05, 0.02};
        
        // Covariance matrix (annualized)
        //                   Stocks   Bonds  RealEst   Gold    Cash
        std::vector<std::vector<double>> covariance = {
            {0.0400, 0.0010, 0.0200, 0.0005, 0.0000},  // Stocks
            {0.0010, 0.0025, 0.0005, 0.0010, 0.0001},  // Bonds
            {0.0200, 0.0005, 0.0300, 0.0010, 0.0000},  // Real Estate
            {0.0005, 0.0010, 0.0010, 0.0100, 0.0000},  // Gold
            {0.0000, 0.0001, 0.0000, 0.0000, 0.0001}   // Cash
        };
        
        double maxAllocation = 0.40;  // Max 40% in any single asset

        const int nA = static_cast<int>(assetNames.size());

        // ====================================================================
        // PRINT PROBLEM DESCRIPTION
        // ====================================================================
        std::cout << "PROBLEM DATA\n";
        std::cout << "------------\n";
        std::cout << "Number of Assets: " << nA << "\n";
        std::cout << "Max Allocation per Asset: " << maxAllocation * 100 << "%\n\n";
        
        std::cout << "Asset Expected Returns & Volatility:\n";
        std::cout << std::setw(15) << "Asset" << std::setw(15) << "Exp. Return"
                  << std::setw(15) << "Volatility\n";
        std::cout << std::string(45, '-') << "\n";
        
        for (int a = 0; a < nA; ++a) {
            double vol = std::sqrt(covariance[a][a]) * 100;
            std::cout << std::setw(15) << assetNames[a]
                      << std::setw(14) << std::fixed << std::setprecision(1) 
                      << expectedReturn[a] * 100 << "%"
                      << std::setw(14) << vol << "%\n";
        }
        std::cout << "\n";
        
        std::cout << "Correlation Matrix:\n";
        std::cout << std::setw(15) << "";
        for (int a = 0; a < nA; ++a) {
            std::cout << std::setw(10) << assetNames[a].substr(0, 8);
        }
        std::cout << "\n" << std::string(15 + nA * 10, '-') << "\n";
        
        for (int a = 0; a < nA; ++a) {
            std::cout << std::setw(15) << assetNames[a];
            for (int b = 0; b < nA; ++b) {
                double corr = covariance[a][b] / 
                              (std::sqrt(covariance[a][a]) * std::sqrt(covariance[b][b]));
                std::cout << std::setw(10) << std::fixed << std::setprecision(2) << corr;
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        // ====================================================================
        // SINGLE OPTIMIZATION: Minimum Variance Portfolio
        // ====================================================================
        std::cout << "MINIMUM VARIANCE PORTFOLIO\n";
        std::cout << "--------------------------\n";
        
        PortfolioBuilder minVarBuilder(assetNames, expectedReturn, covariance, 0.0, maxAllocation);
        minVarBuilder.optimize();

        if (minVarBuilder.hasSolution()) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Expected Return: " << minVarBuilder.store()["expected_return"].get<double>() * 100 << "%\n";
            std::cout << "Portfolio Std Dev: " << minVarBuilder.store()["std_dev"].get<double>() * 100 << "%\n";
            std::cout << "Sharpe Ratio: " << minVarBuilder.store()["sharpe_ratio"].get<double>() << "\n\n";
            
            std::cout << "Allocation:\n";
            auto alloc = minVarBuilder.getAllocation();
            for (int a = 0; a < nA; ++a) {
                if (alloc[a] > 0.001) {
                    std::cout << "  " << std::setw(15) << assetNames[a] 
                              << ": " << std::setw(6) << alloc[a] * 100 << "%\n";
                }
            }
        }
        std::cout << "\n";

        // ====================================================================
        // EFFICIENT FRONTIER
        // ====================================================================
        std::cout << "EFFICIENT FRONTIER\n";
        std::cout << "------------------\n";
        std::cout << "Computing efficient frontier with 10 points...\n\n";
        
        auto frontier = computeEfficientFrontier(assetNames, expectedReturn, covariance, 
                                                  maxAllocation, 10);
        
        std::cout << std::setw(12) << "Return" << std::setw(12) << "Risk"
                  << std::setw(12) << "Sharpe";
        for (int a = 0; a < nA; ++a) {
            std::cout << std::setw(10) << assetNames[a].substr(0, 8);
        }
        std::cout << "\n" << std::string(36 + nA * 10, '-') << "\n";
        
        for (const auto& point : frontier) {
            std::cout << std::setw(11) << point.expectedReturn * 100 << "%"
                      << std::setw(11) << point.stdDev * 100 << "%"
                      << std::setw(12) << std::fixed << std::setprecision(2) << point.sharpeRatio;
            
            for (int a = 0; a < nA; ++a) {
                if (point.allocation[a] > 0.001) {
                    std::cout << std::setw(9) << static_cast<int>(point.allocation[a] * 100) << "%";
                } else {
                    std::cout << std::setw(10) << "-";
                }
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        // ====================================================================
        // OPTIMAL SHARPE RATIO PORTFOLIO
        // ====================================================================
        std::cout << "OPTIMAL PORTFOLIO (Maximum Sharpe Ratio)\n";
        std::cout << "-----------------------------------------\n";
        
        // Find the portfolio with highest Sharpe ratio
        auto bestIt = std::max_element(frontier.begin(), frontier.end(),
            [](const auto& a, const auto& b) { return a.sharpeRatio < b.sharpeRatio; });
        
        if (bestIt != frontier.end()) {
            std::cout << "Expected Return: " << bestIt->expectedReturn * 100 << "%\n";
            std::cout << "Portfolio Risk: " << bestIt->stdDev * 100 << "%\n";
            std::cout << "Sharpe Ratio: " << bestIt->sharpeRatio << "\n\n";
            
            std::cout << "Optimal Allocation:\n";
            for (int a = 0; a < nA; ++a) {
                double pct = bestIt->allocation[a] * 100;
                if (pct > 0.1) {
                    std::cout << "  " << std::setw(15) << assetNames[a] << ": ";
                    
                    // Visual bar
                    int barLen = static_cast<int>(pct / 2);
                    std::cout << "[";
                    for (int i = 0; i < barLen; ++i) std::cout << "=";
                    for (int i = barLen; i < 20; ++i) std::cout << " ";
                    std::cout << "] " << std::setw(5) << pct << "%\n";
                }
            }
        }
        
        std::cout << "\n";
        std::cout << "INTERPRETATION\n";
        std::cout << "--------------\n";
        std::cout << "The efficient frontier shows the trade-off between risk and return.\n";
        std::cout << "Moving right along the frontier increases both return and risk.\n";
        std::cout << "The Sharpe ratio measures risk-adjusted return (higher is better).\n";

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
