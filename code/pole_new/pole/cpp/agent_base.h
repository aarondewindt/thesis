//
// Created by adewindt on 6/7/20.
//

#ifndef POLE_NEW_AGENT_BASE_H
#define POLE_NEW_AGENT_BASE_H

#include <map>
#include <vector>

#include "types.h"
#include "environment.h"

namespace pole {
    class AgentBase {
    public:
        virtual bool step() = 0;
        virtual i64 run_episode(i64 max_steps) = 0;

        virtual std::map<std::string, std::vector<f64>> get_data() = 0;
        virtual std::map<std::string, f64> get_scalar_data() = 0;

        virtual f64 get_reward_sum() = 0;
    };

}

#endif //POLE_NEW_AGENT_BASE_H
