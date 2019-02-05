//
// Created by elvieto on 24-11-18.
//

#ifndef POLE_PID_AGENT_H
#define POLE_PID_AGENT_H

#include "agent.h"
#include "pole.h"
#include <map>

class PIDAgent : Agent{
public:
    PIDAgent(double k_p, double k_i, double k_d);
    ~PIDAgent();
    bool run_step() override;
    void run_episode(long max_steps) override;
    void begin_episode() override;

    inline void set_environment(Pole *pole) override {
        this->pole = pole;
    }

    inline Pole *get_environment() override {
        return this->pole;
    }

    std::map<std::string, std::vector<double>*>* get_data() override;
    std::map<std::string, double> get_scalar_data() override;

private:
    double k_p;
    double k_i;
    double k_d;
    Pole *pole;

    double previous_error;
    double integral;

    std::map<std::string, std::vector<double>*> data_map;
};


#endif //POLE_PID_AGENT_H
