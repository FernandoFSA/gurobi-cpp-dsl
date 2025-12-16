/*
================================================================================
EXAMPLE 07: JOB SCHEDULING - Single Machine with Due Dates
================================================================================
DIFFICULTY: Intermediate
PROBLEM TYPE: Mixed-Integer Programming (MIP)

PROBLEM DESCRIPTION
-------------------
A machine must process a set of jobs. Each job has a processing time, a due
date, and a tardiness penalty (cost per unit time late). Jobs cannot overlap
and the machine can process only one job at a time. The goal is to schedule
jobs to minimize total weighted tardiness.

MATHEMATICAL MODEL
------------------
Sets:
    J = {0, 1, ..., n-1}    Jobs

Parameters:
    processing[j]   Processing time for job j
    due[j]          Due date for job j
    penalty[j]      Penalty per unit of tardiness for job j
    M               Big-M constant (sum of all processing times)

Variables:
    start[j] >= 0           Start time of job j
    tardy[j] >= 0           Tardiness of job j (max(0, completion - due))
    before[i,j] in {0,1}    1 if job i is scheduled before job j

Objective:
    min  sum_j penalty[j] * tardy[j]

Constraints:
    Tardiness[j]:   tardy[j] >= start[j] + processing[j] - due[j]     for all j
    NoOverlap[i,j]: start[j] >= start[i] + processing[i] - M*(1-before[i,j])
                    start[i] >= start[j] + processing[j] - M*before[i,j]
                                                          for all i < j

DSL FEATURES DEMONSTRATED
-------------------------
- Big-M constraints          Disjunctive scheduling
- Auxiliary variables        Tardiness linearization
- Filtered pair domain       i < j for no-overlap constraints
- setStart() warm start      Provide initial solution
- Time limit management      Handle long solve times
- Multiple solutions         Access solution pool

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
DECLARE_ENUM_WITH_COUNT(Vars, Start, Tardy, Before);
DECLARE_ENUM_WITH_COUNT(Cons, Tardiness, NoOverlap1, NoOverlap2);

// ============================================================================
// SCHEDULING BUILDER
// ============================================================================
class SchedulingBuilder : public dsl::ModelBuilder<Vars, Cons> {
private:
    std::vector<std::string> jobNames_;
    std::vector<double> processing_;
    std::vector<double> due_;
    std::vector<double> penalty_;
    double bigM_;
    
    int nJobs_;

public:
    SchedulingBuilder(
        const std::vector<std::string>& jobNames,
        const std::vector<double>& processing,
        const std::vector<double>& due,
        const std::vector<double>& penalty
    ) : jobNames_(jobNames), processing_(processing), due_(due), penalty_(penalty),
        nJobs_(static_cast<int>(jobNames.size()))
    {
        // Big-M = sum of all processing times (makespan upper bound)
        bigM_ = std::accumulate(processing.begin(), processing.end(), 0.0);
        
        store()["n_jobs"] = nJobs_;
        store()["big_m"] = bigM_;
        store()["total_processing"] = bigM_;
    }

protected:
    void addParameters() override {
        quiet();
        applyPreset(Preset::Fast);
        timeLimit(30.0);        // Cap solve time
        mipGapLimit(0.01);      // 1% gap tolerance
    }

    void addVariables() override {
        auto J = dsl::range(0, nJobs_);
        
        // start[j] - start time of job j
        auto Start = dsl::VariableFactory::add(
            model(), GRB_CONTINUOUS, 0.0, bigM_, "start", nJobs_
        );
        variables().set(Vars::Start, std::move(Start));

        // tardy[j] - tardiness of job j
        auto Tardy = dsl::VariableFactory::add(
            model(), GRB_CONTINUOUS, 0.0, bigM_, "tardy", nJobs_
        );
        variables().set(Vars::Tardy, std::move(Tardy));

        // before[i,j] - 1 if job i is before job j (only for i < j)
        auto pairDomain = (J * J) | dsl::filter([](int i, int j) { return i < j; });
        auto Before = dsl::VariableFactory::addIndexed(
            model(), GRB_BINARY, 0.0, 1.0, "before", pairDomain
        );
        variables().set(Vars::Before, std::move(Before));
    }

    void addConstraints() override {
        auto& Start = variables().get(Vars::Start);
        auto& Tardy = variables().get(Vars::Tardy);
        auto& Before = variables().get(Vars::Before);
        auto J = dsl::range(0, nJobs_);

        // Tardiness[j]: tardy[j] >= completion[j] - due[j]
        auto tardyConstrs = dsl::ConstraintFactory::addIndexed(
            model(), "tardiness", J,
            [&](int j) {
                // tardy[j] >= start[j] + processing[j] - due[j]
                return Tardy.at(j) >= Start.at(j) + processing_[j] - due_[j];
            }
        );
        constraints().set(Cons::Tardiness, std::move(tardyConstrs));

        // No-overlap constraints for each pair i < j
        auto pairDomain = (J * J) | dsl::filter([](int i, int j) { return i < j; });
        
        // If before[i,j] = 1: job i before j, so start[j] >= start[i] + processing[i]
        auto noOverlap1 = dsl::ConstraintFactory::addIndexed(
            model(), "no_overlap1", pairDomain,
            [&](int i, int j) {
                return Start.at(j) >= Start.at(i) + processing_[i] - bigM_ * (1 - Before.asIndexed().at(i, j));
            }
        );
        constraints().set(Cons::NoOverlap1, std::move(noOverlap1));

        // If before[i,j] = 0: job j before i, so start[i] >= start[j] + processing[j]
        auto noOverlap2 = dsl::ConstraintFactory::addIndexed(
            model(), "no_overlap2", pairDomain,
            [&](int i, int j) {
                return Start.at(i) >= Start.at(j) + processing_[j] - bigM_ * Before.asIndexed().at(i, j);
            }
        );
        constraints().set(Cons::NoOverlap2, std::move(noOverlap2));
    }

    void addObjective() override {
        auto& Tardy = variables().get(Vars::Tardy);
        auto J = dsl::range(0, nJobs_);

        // Minimize total weighted tardiness
        minimize(dsl::sum(J, [&](int j) {
            return penalty_[j] * Tardy.at(j);
        }));
    }

    void afterOptimize() override {
        if (hasSolution()) {
            store()["total_penalty"] = objVal();
            store()["runtime"] = runtime();
            store()["gap"] = mipGap();
            store()["solutions_found"] = solutionCount();
        }
    }
    
    // Override beforeOptimize to set warm start after variables are created
    void beforeOptimize() override {
        if (store().contains("use_warm_start") && store()["use_warm_start"].get<bool>()) {
            setEDDWarmStartInternal();
        }
    }

private:
    // Internal warm start (called after variables exist)
    void setEDDWarmStartInternal() {
        auto& Start = variables().get(Vars::Start);
        
        // Sort jobs by due date (Earliest Due Date heuristic)
        std::vector<int> order(nJobs_);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), 
                  [this](int a, int b) { return due_[a] < due_[b]; });
        
        // Set start times
        double time = 0;
        for (int j : order) {
            dsl::setStart(Start.at(j), time);
            time += processing_[j];
        }
    }

public:
    // Accessors
    const std::vector<std::string>& jobNames() const { return jobNames_; }
    int nJobs() const { return nJobs_; }
    double processing(int j) const { return processing_[j]; }
    double due(int j) const { return due_[j]; }
    double penalty(int j) const { return penalty_[j]; }
    
    // Enable warm start (must be called before optimize)
    void enableWarmStart() {
        store()["use_warm_start"] = true;
    }
    
    // Get schedule as sorted list of (job, start, completion, tardiness)
    std::vector<std::tuple<int, double, double, double>> getSchedule() const {
        std::vector<std::tuple<int, double, double, double>> schedule;
        const auto& Start = variables().get(Vars::Start);
        const auto& Tardy = variables().get(Vars::Tardy);
        
        for (int j = 0; j < nJobs_; ++j) {
            double start = dsl::value(Start.at(j));
            double completion = start + processing_[j];
            double tardiness = dsl::value(Tardy.at(j));
            schedule.emplace_back(j, start, completion, tardiness);
        }
        
        // Sort by start time
        std::sort(schedule.begin(), schedule.end(),
                  [](const auto& a, const auto& b) { return std::get<1>(a) < std::get<1>(b); });
        
        return schedule;
    }
};

// ============================================================================
// MAIN PROGRAM
// ============================================================================
int main() {
    std::cout << "================================================================\n";
    std::cout << "EXAMPLE 07: Single Machine Scheduling with Tardiness\n";
    std::cout << "================================================================\n\n";

    try {
        // ====================================================================
        // PROBLEM DATA
        // ====================================================================
        std::vector<std::string> jobNames = {
            "Job_A", "Job_B", "Job_C", "Job_D", "Job_E", "Job_F"
        };
        
        // Processing time for each job
        std::vector<double> processing = {4, 3, 6, 2, 5, 3};
        
        // Due date for each job
        std::vector<double> due = {10, 6, 15, 8, 20, 12};
        
        // Penalty per unit tardiness
        std::vector<double> penalty = {5, 8, 3, 10, 2, 6};

        const int nJ = static_cast<int>(jobNames.size());
        double totalProcessing = std::accumulate(processing.begin(), processing.end(), 0.0);

        // ====================================================================
        // PRINT PROBLEM DESCRIPTION
        // ====================================================================
        std::cout << "PROBLEM DATA\n";
        std::cout << "------------\n";
        std::cout << "Number of Jobs: " << nJ << "\n";
        std::cout << "Total Processing Time: " << totalProcessing << " hours\n\n";
        
        std::cout << "Job Details:\n";
        std::cout << std::setw(10) << "Job" << std::setw(12) << "Processing"
                  << std::setw(10) << "Due" << std::setw(12) << "Penalty/hr\n";
        std::cout << std::string(44, '-') << "\n";
        
        for (int j = 0; j < nJ; ++j) {
            std::cout << std::setw(10) << jobNames[j]
                      << std::setw(12) << processing[j]
                      << std::setw(10) << due[j]
                      << std::setw(12) << ("$" + std::to_string(static_cast<int>(penalty[j]))) << "\n";
        }
        std::cout << "\n";

        // ====================================================================
        // BUILD AND SOLVE MODEL
        // ====================================================================
        std::cout << "SOLVING...\n";
        std::cout << "----------\n";
        
        SchedulingBuilder builder(jobNames, processing, due, penalty);
        
        // Enable warm start using EDD heuristic (applied during optimize)
        std::cout << "Enabling Earliest Due Date warm start...\n";
        builder.enableWarmStart();
        
        builder.optimize();

        std::cout << "Status: " << dsl::statusString(builder.status()) << "\n";
        std::cout << "Runtime: " << builder.runtime() << " seconds\n";
        std::cout << "MIP Gap: " << builder.mipGap() * 100 << "%\n";
        std::cout << "Solutions Found: " << builder.solutionCount() << "\n";
        std::cout << dsl::modelSummary(builder.model()) << "\n";

        // ====================================================================
        // DISPLAY RESULTS
        // ====================================================================
        if (builder.hasSolution()) {
            std::cout << std::fixed << std::setprecision(2);
            
            std::cout << "\nOPTIMAL SCHEDULE\n";
            std::cout << "----------------\n";
            std::cout << "Total Tardiness Penalty: $" << builder.objVal() << "\n\n";

            // Gantt-style schedule
            std::cout << "Schedule (sorted by start time):\n";
            std::cout << std::setw(10) << "Job" << std::setw(10) << "Start"
                      << std::setw(10) << "End" << std::setw(10) << "Due"
                      << std::setw(12) << "Tardiness" << std::setw(12) << "Penalty\n";
            std::cout << std::string(64, '-') << "\n";
            
            double totalPenalty = 0;
            for (auto [j, start, completion, tardiness] : builder.getSchedule()) {
                double jobPenalty = penalty[j] * tardiness;
                totalPenalty += jobPenalty;
                
                std::cout << std::setw(10) << jobNames[j]
                          << std::setw(10) << start
                          << std::setw(10) << completion
                          << std::setw(10) << due[j]
                          << std::setw(12) << tardiness
                          << std::setw(12) << ("$" + std::to_string(static_cast<int>(jobPenalty)));
                
                if (tardiness > 0.01) {
                    std::cout << " [LATE]";
                } else if (completion <= due[j]) {
                    std::cout << " [ON TIME]";
                }
                std::cout << "\n";
            }
            std::cout << std::string(64, '-') << "\n";
            std::cout << std::setw(52) << "Total: " 
                      << std::setw(12) << ("$" + std::to_string(static_cast<int>(totalPenalty))) << "\n\n";

            // Visual timeline
            std::cout << "Visual Timeline:\n";
            std::cout << "Time:  ";
            for (int t = 0; t <= static_cast<int>(totalProcessing); t += 2) {
                std::cout << std::setw(4) << t;
            }
            std::cout << "\n       ";
            for (int t = 0; t <= static_cast<int>(totalProcessing); t += 2) {
                std::cout << "----";
            }
            std::cout << "\n";
            
            for (auto [j, start, completion, tardiness] : builder.getSchedule()) {
                std::cout << std::setw(7) << jobNames[j] << ":";
                
                int startPos = static_cast<int>(start / 2);
                int endPos = static_cast<int>(completion / 2);
                
                for (int p = 0; p < startPos; ++p) std::cout << "    ";
                std::cout << "[";
                for (int p = startPos; p < endPos - 1; ++p) std::cout << "====";
                std::cout << "===]";
                std::cout << "\n";
            }
            
            std::cout << "\n";

            // Summary statistics
            std::cout << "SUMMARY\n";
            std::cout << "-------\n";
            int onTime = 0, late = 0;
            double maxTardiness = 0;
            for (auto [j, start, completion, tardiness] : builder.getSchedule()) {
                if (tardiness > 0.01) {
                    late++;
                    maxTardiness = std::max(maxTardiness, tardiness);
                } else {
                    onTime++;
                }
            }
            std::cout << "Jobs on time: " << onTime << " / " << nJ << "\n";
            std::cout << "Jobs late: " << late << " / " << nJ << "\n";
            std::cout << "Maximum tardiness: " << maxTardiness << " hours\n";
            std::cout << "Schedule makespan: " << totalProcessing << " hours\n";
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
