
#include "season.h"
#include "rand.h"
#include <printf.h>

Season::Season(Agent *agent) : agent(agent), pole(agent->get_environment()) {

}

void Season::run(long n_episodes, long n_record, long max_steps) {
    // Clear season data
    clear_all_logs();

    bool log_whole;
    for (int i = 0; i < n_episodes; i++) {
        if (i % 1 == 0) {
            std::printf("E %d\n", i);
        }
        // log the episode results if neccesary or if its the last one.
        log_whole = ((i % ((n_episodes / n_record) + 1)) == 0) || (i == (n_episodes - 1));

        // Reset pole
        pole->theta = frand(-3.14, 3.14);  // 0.0872
        pole->theta_dot = frand(-5, 5);
        pole->time = 0;

//        std::printf("%f %f\n", pole->theta, pole->theta_dot);

        // Run episode
        agent->run_episode(max_steps);

        // Log results
        std::map<std::string, std::vector<double>*> *episode_data = agent->get_data();
        if (log_whole) {
            // Log episode state data if necessary.
            data_log[i] = episode_data;
        } else {
            // Delete the logged data if not.
            for (auto const& item : *episode_data) {
                delete item.second;
            }
            delete episode_data;
        }

        std::map<std::string, double> episode_scalar_data = agent->get_scalar_data();
        for (auto const& item : episode_scalar_data) {
            if (!scalar_data.count(item.first)){
                scalar_data[item.first] = new std::vector<double>();
            }
            scalar_data[item.first]->push_back(item.second);
        }

    }
}

std::map<long, std::map<std::string, std::vector<double>*>*> *Season::get_data_log() {
    return &data_log;
}

std::map<std::string, std::vector<double>*> *Season::get_scalar_data() {
    return &scalar_data;
}

void Season::clear_all_logs() {
    for (auto const& episode_log : data_log) {
        for (auto const& episode_value : *(episode_log.second)) {
            std::vector<double>* values = episode_value.second;
            delete values;
        }
        delete episode_log.second;
    }

    for (auto const& value_log : scalar_data) {
        value_log.second->clear();
    }

    data_log.clear();
    scalar_data.clear();
}
