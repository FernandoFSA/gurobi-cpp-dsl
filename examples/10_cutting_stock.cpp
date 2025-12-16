/*
================================================================================
EXAMPLE 10: CUTTING STOCK PROBLEM - Pattern-Based Formulation
================================================================================
DIFFICULTY: Advanced
PROBLEM TYPE: Integer Programming with Pattern Generation

PROBLEM DESCRIPTION
-------------------
A paper company has stock rolls of fixed width. Customers order smaller rolls
of various widths. The company must cut the stock rolls to fulfill all orders
while minimizing the number of stock rolls used (waste minimization).

This example demonstrates a pattern-based formulation where we enumerate
feasible cutting patterns and select the minimum number of rolls/patterns.

MATHEMATICAL MODEL
------------------
Sets:
    I = {0, 1, ..., m-1}    Item types (ordered widths)
    P = {0, 1, ..., n-1}    Cutting patterns (generated)

Parameters:
    demand[i]       Number of items of type i needed
    pattern[p,i]    Number of items of type i cut in pattern p
    waste[p]        Waste produced by pattern p

Variables:
    x[p] >= 0, integer     Number of times pattern p is used

Objective:
    min  sum_p x[p]   (minimize total rolls used)
    or:  min  sum_p waste[p] * x[p]  (minimize waste)

Constraints:
    Demand[i]:  sum_p pattern[p,i] * x[p] >= demand[i]    for all i

DSL FEATURES DEMONSTRATED
-------------------------
- Pattern generation             Pre-solve enumeration
- Sparse pattern matrices        Only store non-zero entries
- Integer programming            GRB_INTEGER variables
- Objective switching            Compare roll count vs waste
- Solution interpretation        Map back to physical cuts
- Model modification             Change objective and re-solve

================================================================================
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <gurobi_dsl/dsl.h>

// ============================================================================
// TYPE-SAFE ENUM KEYS
// ============================================================================
DECLARE_ENUM_WITH_COUNT(Vars, X);
DECLARE_ENUM_WITH_COUNT(Cons, Demand);

// ============================================================================
// PATTERN STRUCTURE
// ============================================================================
struct CuttingPattern {
    std::vector<int> cuts;  // cuts[i] = number of item type i in this pattern
    int waste;              // waste = stockWidth - sum(cuts[i] * itemWidth[i])
    
    std::string toString(const std::vector<int>& itemWidths) const {
        std::string s;
        for (size_t i = 0; i < cuts.size(); ++i) {
            if (cuts[i] > 0) {
                if (!s.empty()) s += " + ";
                s += std::to_string(cuts[i]) + "x" + std::to_string(itemWidths[i]);
            }
        }
        s += " (waste: " + std::to_string(waste) + ")";
        return s;
    }
};

// ============================================================================
// PATTERN GENERATOR
// ============================================================================
std::vector<CuttingPattern> generatePatterns(
    int stockWidth,
    const std::vector<int>& itemWidths,
    int maxPatternsPerItem = 100
) {
    std::vector<CuttingPattern> patterns;
    int nItems = static_cast<int>(itemWidths.size());
    
    // Generate patterns using bounded enumeration
    // For simplicity, we generate patterns by iterating over possible counts
    std::vector<int> maxCounts(nItems);
    for (int i = 0; i < nItems; ++i) {
        maxCounts[i] = stockWidth / itemWidths[i];
    }
    
    // Recursive enumeration (bounded depth-first)
    std::function<void(int, std::vector<int>&, int)> enumerate = 
        [&](int idx, std::vector<int>& counts, int usedWidth) {
            if (usedWidth > stockWidth) return;
            
            if (idx == nItems) {
                if (usedWidth > 0) {  // Non-empty pattern
                    CuttingPattern p;
                    p.cuts = counts;
                    p.waste = stockWidth - usedWidth;
                    patterns.push_back(p);
                }
                return;
            }
            
            // Limit patterns per item type
            int maxCount = std::min(maxCounts[idx], 
                                   (stockWidth - usedWidth) / itemWidths[idx]);
            
            for (int c = 0; c <= maxCount; ++c) {
                counts[idx] = c;
                enumerate(idx + 1, counts, usedWidth + c * itemWidths[idx]);
                
                // Limit total patterns
                if (patterns.size() > 1000) return;
            }
            counts[idx] = 0;
        };
    
    std::vector<int> counts(nItems, 0);
    enumerate(0, counts, 0);
    
    return patterns;
}

// ============================================================================
// CUTTING STOCK BUILDER
// ============================================================================
class CuttingStockBuilder : public dsl::ModelBuilder<Vars, Cons> {
private:
    std::vector<int> itemWidths_;
    std::vector<int> demand_;
    std::vector<CuttingPattern> patterns_;
    int stockWidth_;
    bool minimizeWaste_;
    
    int nItems_;
    int nPatterns_;

public:
    CuttingStockBuilder(
        const std::vector<int>& itemWidths,
        const std::vector<int>& demand,
        int stockWidth,
        bool minimizeWaste = false
    ) : itemWidths_(itemWidths), demand_(demand), stockWidth_(stockWidth),
        minimizeWaste_(minimizeWaste),
        nItems_(static_cast<int>(itemWidths.size()))
    {
        // Generate cutting patterns
        patterns_ = generatePatterns(stockWidth, itemWidths);
        nPatterns_ = static_cast<int>(patterns_.size());
        
        store()["n_items"] = nItems_;
        store()["n_patterns"] = nPatterns_;
        store()["stock_width"] = stockWidth_;
        
        int totalDemand = std::accumulate(demand.begin(), demand.end(), 0);
        store()["total_demand"] = totalDemand;
    }

protected:
    void addParameters() override {
        quiet();
        applyPreset(Preset::Fast);
    }

    void addVariables() override {
        // x[p] - integer, number of times pattern p is used
        auto X = dsl::VariableFactory::add(
            model(), GRB_INTEGER, 0.0, GRB_INFINITY, "x", nPatterns_
        );
        variables().set(Vars::X, std::move(X));
    }

    void addConstraints() override {
        auto& X = variables().get(Vars::X);
        auto I = dsl::range(0, nItems_);
        auto P = dsl::range(0, nPatterns_);

        // Demand[i]: Must satisfy demand for each item type
        auto demandConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "demand", I,
            [&](int i) {
                GRBLinExpr lhs = 0;
                for (int p : P) {
                    if (patterns_[p].cuts[i] > 0) {
                        lhs += patterns_[p].cuts[i] * X.at(p);
                    }
                }
                return lhs >= demand_[i];
            }
        );
        constraints().set(Cons::Demand, std::move(demandConstrs));
    }

    void addObjective() override {
        auto& X = variables().get(Vars::X);
        auto P = dsl::range(0, nPatterns_);

        if (minimizeWaste_) {
            // Minimize total waste
            minimize(dsl::sum(P, [&](int p) {
                return patterns_[p].waste * X.at(p);
            }));
        } else {
            // Minimize number of rolls used
            minimize(dsl::sum(P, [&](int p) {
                return X.at(p);
            }));
        }
    }

    void afterOptimize() override {
        if (hasSolution()) {
            auto& X = variables().get(Vars::X);
            
            int rollsUsed = 0;
            int totalWaste = 0;
            for (int p = 0; p < nPatterns_; ++p) {
                int count = static_cast<int>(std::round(dsl::value(X.at(p))));
                if (count > 0) {
                    rollsUsed += count;
                    totalWaste += count * patterns_[p].waste;
                }
            }
            
            store()["rolls_used"] = rollsUsed;
            store()["total_waste"] = totalWaste;
            store()["runtime"] = runtime();
        }
    }

public:
    // Accessors
    const std::vector<int>& itemWidths() const { return itemWidths_; }
    const std::vector<int>& demand() const { return demand_; }
    const std::vector<CuttingPattern>& patterns() const { return patterns_; }
    int stockWidth() const { return stockWidth_; }
    int nItems() const { return nItems_; }
    int nPatterns() const { return nPatterns_; }
    
    // Get solution as (pattern index, count) pairs
    std::vector<std::pair<int, int>> getSolution() const {
        std::vector<std::pair<int, int>> solution;
        const auto& X = variables().get(Vars::X);
        
        for (int p = 0; p < nPatterns_; ++p) {
            int count = static_cast<int>(std::round(dsl::value(X.at(p))));
            if (count > 0) {
                solution.emplace_back(p, count);
            }
        }
        
        // Sort by count (most used first)
        std::sort(solution.begin(), solution.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        return solution;
    }
};

// ============================================================================
// MAIN PROGRAM
// ============================================================================
int main() {
    std::cout << "================================================================\n";
    std::cout << "EXAMPLE 10: Cutting Stock Problem\n";
    std::cout << "================================================================\n\n";

    try {
        // ====================================================================
        // PROBLEM DATA
        // ====================================================================
        int stockWidth = 100;  // Stock roll width
        
        // Item types with their widths and demands
        std::vector<int> itemWidths = {45, 36, 31, 14};
        std::vector<int> demand = {97, 610, 395, 211};

        const int nItems = static_cast<int>(itemWidths.size());

        // ====================================================================
        // PRINT PROBLEM DESCRIPTION
        // ====================================================================
        std::cout << "PROBLEM DATA\n";
        std::cout << "------------\n";
        std::cout << "Stock Roll Width: " << stockWidth << " units\n\n";
        
        std::cout << "Items to Cut:\n";
        std::cout << std::setw(10) << "Item" << std::setw(10) << "Width"
                  << std::setw(10) << "Demand\n";
        std::cout << std::string(30, '-') << "\n";
        
        int totalPieces = 0;
        for (int i = 0; i < nItems; ++i) {
            std::cout << std::setw(10) << ("Type " + std::to_string(i + 1))
                      << std::setw(10) << itemWidths[i]
                      << std::setw(10) << demand[i] << "\n";
            totalPieces += demand[i];
        }
        std::cout << "\nTotal pieces needed: " << totalPieces << "\n\n";

        // ====================================================================
        // PATTERN GENERATION
        // ====================================================================
        std::cout << "PATTERN GENERATION\n";
        std::cout << "------------------\n";
        
        auto patterns = generatePatterns(stockWidth, itemWidths);
        std::cout << "Generated " << patterns.size() << " feasible cutting patterns.\n\n";
        
        // Show some example patterns
        std::cout << "Sample Patterns:\n";
        int showCount = std::min(10, static_cast<int>(patterns.size()));
        for (int p = 0; p < showCount; ++p) {
            std::cout << "  Pattern " << std::setw(3) << p << ": " 
                      << patterns[p].toString(itemWidths) << "\n";
        }
        if (patterns.size() > 10) {
            std::cout << "  ... and " << patterns.size() - 10 << " more patterns\n";
        }
        std::cout << "\n";

        // ====================================================================
        // SOLVE: MINIMIZE ROLLS
        // ====================================================================
        std::cout << "SOLVING (Minimize Rolls Used)\n";
        std::cout << "-----------------------------\n";
        
        CuttingStockBuilder rollBuilder(itemWidths, demand, stockWidth, false);
        rollBuilder.optimize();

        std::cout << "Status: " << dsl::statusString(rollBuilder.status()) << "\n";
        std::cout << dsl::modelSummary(rollBuilder.model()) << "\n";

        if (rollBuilder.hasSolution()) {
            std::cout << std::fixed << std::setprecision(0);
            
            int rollsUsed = rollBuilder.store()["rolls_used"];
            int totalWaste = rollBuilder.store()["total_waste"];
            double wastePercent = 100.0 * totalWaste / (rollsUsed * stockWidth);
            
            std::cout << "\nRESULTS (Minimize Rolls)\n";
            std::cout << "------------------------\n";
            std::cout << "Rolls Used: " << rollsUsed << "\n";
            std::cout << "Total Waste: " << totalWaste << " units ("
                      << std::setprecision(1) << wastePercent << "%)\n\n";
            
            std::cout << "Cutting Plan:\n";
            std::cout << std::setw(8) << "Count" << std::setw(8) << "Pattern" << "  Description\n";
            std::cout << std::string(60, '-') << "\n";
            
            for (auto [patternIdx, count] : rollBuilder.getSolution()) {
                std::cout << std::setw(8) << count
                          << std::setw(8) << patternIdx
                          << "  " << patterns[patternIdx].toString(itemWidths) << "\n";
            }
            std::cout << "\n";

            // Verify demand satisfaction
            std::cout << "Demand Verification:\n";
            std::cout << std::setw(10) << "Item" << std::setw(12) << "Demanded"
                      << std::setw(12) << "Produced" << std::setw(12) << "Excess\n";
            std::cout << std::string(46, '-') << "\n";
            
            std::vector<int> produced(nItems, 0);
            for (auto [patternIdx, count] : rollBuilder.getSolution()) {
                for (int i = 0; i < nItems; ++i) {
                    produced[i] += count * patterns[patternIdx].cuts[i];
                }
            }
            
            for (int i = 0; i < nItems; ++i) {
                std::cout << std::setw(10) << ("Type " + std::to_string(i + 1))
                          << std::setw(12) << demand[i]
                          << std::setw(12) << produced[i]
                          << std::setw(12) << (produced[i] - demand[i]) << "\n";
            }
        }
        std::cout << "\n";

        // ====================================================================
        // SOLVE: MINIMIZE WASTE
        // ====================================================================
        std::cout << "SOLVING (Minimize Waste)\n";
        std::cout << "------------------------\n";
        
        CuttingStockBuilder wasteBuilder(itemWidths, demand, stockWidth, true);
        wasteBuilder.optimize();

        if (wasteBuilder.hasSolution()) {
            int rollsUsed = wasteBuilder.store()["rolls_used"].get<int>();
            int totalWaste = wasteBuilder.store()["total_waste"].get<int>();
            double wastePercent = 100.0 * totalWaste / (rollsUsed * stockWidth);
            
            std::cout << "\nRESULTS (Minimize Waste)\n";
            std::cout << "------------------------\n";
            std::cout << "Rolls Used: " << rollsUsed << "\n";
            std::cout << "Total Waste: " << totalWaste << " units ("
                      << std::setprecision(1) << wastePercent << "%)\n\n";
            
            std::cout << "Cutting Plan:\n";
            for (auto [patternIdx, count] : wasteBuilder.getSolution()) {
                std::cout << "  " << std::setw(4) << count << " x "
                          << patterns[patternIdx].toString(itemWidths) << "\n";
            }
        }
        std::cout << "\n";

        // ====================================================================
        // COMPARISON
        // ====================================================================
        if (rollBuilder.hasSolution() && wasteBuilder.hasSolution()) {
            std::cout << "COMPARISON\n";
            std::cout << "----------\n";
            std::cout << std::setw(25) << "Objective" << std::setw(15) << "Min Rolls"
                      << std::setw(15) << "Min Waste\n";
            std::cout << std::string(55, '-') << "\n";
            
            int rollsRoll = rollBuilder.store()["rolls_used"].get<int>();
            int wasteRoll = rollBuilder.store()["total_waste"].get<int>();
            int rollsWaste = wasteBuilder.store()["rolls_used"].get<int>();
            int wasteWaste = wasteBuilder.store()["total_waste"].get<int>();
            
            std::cout << std::setw(25) << "Rolls Used"
                      << std::setw(15) << rollsRoll
                      << std::setw(15) << rollsWaste << "\n";
            std::cout << std::setw(25) << "Total Waste"
                      << std::setw(15) << wasteRoll
                      << std::setw(15) << wasteWaste << "\n";
            
            std::cout << "\nNote: Different objectives can lead to different optimal solutions.\n";
            std::cout << "Minimizing rolls may produce more waste (overproduction).\n";
            std::cout << "Minimizing waste may use more rolls.\n";
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
