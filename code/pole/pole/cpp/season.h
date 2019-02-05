//
// Created by elvieto on 24-11-18.
//

#ifndef POLE_ANALYSIS_H
#define POLE_ANALYSIS_H

#include "agent.h"
#include "pole.h"
#include <vector>
#include <map>


class Season {
public:
    Season(Agent *agent);

    void run(long n_episodes, long n_record, long max_steps);

    std::map<long, std::map<std::string, std::vector<double>*>*> *get_data_log();

    std::map<std::string, std::vector<double>*> *get_scalar_data();

    void clear_all_logs();
private:
    Agent *agent;
    Pole *pole;

    std::map<long, std::map<std::string, std::vector<double>*>*> data_log;
    std::map<std::string, std::vector<double>*> scalar_data;
};

#endif //POLE_ANALYSIS_H
