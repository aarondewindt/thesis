//
// Created by elvieto on 24-11-18.
//

#ifndef POLE_AGENT_H
#define POLE_AGENT_H

#include <map>
#include <vector>
#include <string>
#include "pole.h"

class Agent {
public:
    virtual std::map<std::string, std::vector<double>*>* get_data() = 0;
    virtual std::map<std::string, double> get_scalar_data() = 0;
    virtual void begin_episode() = 0;
    virtual void end_episode() = 0;
    virtual void set_environment(Pole *pole) = 0;
    virtual Pole *get_environment() = 0;

    virtual bool run_step() = 0;
    virtual void run_episode(long max_steps) = 0;


};


#endif //POLE_AGENT_H
