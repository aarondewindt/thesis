use std::collections::HashMap;
use environment::{Environment, State};
use point::Point;
use rand::prelude::*;
use std::f64;
use std::collections::hash_map::Entry;

pub struct StateInfo {
    pub value: f64,
    pub reward: f64,
    pub visit_count: f64
}

pub struct Agent {
    pub states_info: HashMap<State, StateInfo>,
    pub env: Environment,
    pub eps: f64,
    pub gamma: f64,
    pub alpha: f64,
    rng: ThreadRng,
}

impl Agent {
    pub fn new(env: Environment, eps: f64, gamma: f64, alpha: f64) -> Agent {
        Agent {
            states_info: HashMap::new(),
            env,
            eps,
            gamma,
            alpha,
            rng: thread_rng()
        }
    }

    pub fn step(&mut self) -> f64 {
        // Choose direction
        let mut direction = -1;
        let rnd_val: f64 = self.rng.gen();
        if rnd_val > self.eps {
            // Be greedy, look for the next state with the most value.
            let mut best_value = f64::NEG_INFINITY;
            for dir in 0 ..= 7 {
                let state = State {
                    location: self.env.next_location(dir),
                    memory: self.env.memory
                };
                {
                    let value = match self.states_info.entry(state) {
                        Entry::Occupied(mut o) => {
                            if o.get().value.is_nan() {
                                0.0
                            } else {
                                o.get().value
                            }
                        },
                        Entry::Vacant(_) => 0.0
                    };
                    if value > best_value {
                        best_value = value;
                        direction = dir;
                    }
                }
            }
        } else {
            // Explore, choose a random direction
            direction = self.rng.gen_range(0, 8)
        }

        // Get current state
        let state_0 = self.env.state();
        let value_0 ;
        let reward_0 ;
        {
            let info_0 = {
                self.states_info.entry(state_0).or_insert_with(
                    || StateInfo {
                        value: f64::NAN,
                        reward: f64::NAN,
                        visit_count: 0.0
                    }
                )
            };
            // Update visit count
            info_0.visit_count += 1.0;
            value_0 = info_0.value;
            reward_0 = info_0.reward;
        }

        // Get next state
        let state_1 = State {
            location: self.env.next_location(direction),
            memory: self.env.memory
        };

        // Perform action and get reward from the next state.
        let reward_1 = self.env.act(direction);

        // Set the reward of the new state in the table and get its current value.
        let value_1;
        {
            let info_1 = {
                self.states_info.entry(state_1).or_insert_with(
                    || StateInfo {
                        value: f64::NAN,
                        reward: f64::NAN,
                        visit_count: 0.0
                    }
                )
            };
            // If this is the first time we arrive to this state it still
            // Doesn't have a value. So set to value to the reward received.
            if info_1.value.is_nan() {
                info_1.value = reward_1;
            }

            // Get the value of the new_state and set the reward.
            value_1 = info_1.value;
            info_1.reward = reward_1;
        }

        // Calculate and update old state value
        // The value and reward of the initial state are not known.
        // So we skip this step in this case.
        if !value_0.is_nan() {
            let info_0 = self.states_info.get_mut(&state_0).unwrap();
            // let alpha = 1.0 / info_0.visit_count;
            info_0.value += self.alpha * (reward_0 + self.gamma * value_1 - info_0.value);
        }

        // This will either move the agent back into the board if it's
        // out of bounds.
        // To a random location if at the station or building.
        // Nothing in any other case.
        self.env.finish(state_0);

        return reward_1;
    }

    pub fn _print_values_for_memory(&mut self, memory: i8) {
        for y in (-1..(self.env.field_size.1 + 1)).rev() {
            for x in -1..(self.env.field_size.0+1) {
                let value = match self.states_info.entry(State{
                    location: Point(x, y),
                    memory
                }) {
                    Entry::Occupied(mut o) => o.get().value,
                    Entry::Vacant(_o) => 0.0
                };
                print!("\t{:5.2}", value)
            }
            print!("\n")
        }
        print!("\n")
    }

    pub fn print_greedy_move(&mut self, memory: i8) {
        for y in (-1..(self.env.field_size.1 + 1)).rev() {
            for x in -1..(self.env.field_size.0+1) {
                let location = Point(x, y);
                let mut best_value = f64::NEG_INFINITY;
                let mut direction = -1;
                for dir in 0 ..= 7 {
                    let state = State {
                        location: location + match dir {
                            0 => Point(0, 1),
                            1 => Point(1, 1),
                            2 => Point(1, 0),
                            3 => Point(1, -1),
                            4 => Point(0, -1),
                            5 => Point(-1, -1),
                            6 => Point(-1, 0),
                            7 => Point(-1, 1),
                            _ => panic!("Invalid direction '{}'", dir)
                        },
                        memory
                    };
                    {
                        let value = match self.states_info.entry(state) {
                            Entry::Occupied(mut o) => {
                                if o.get().value.is_nan() {
                                    0.0
                                } else {
                                    o.get().value
                                }
                            },
                            Entry::Vacant(_) => 0.0
                        };
                        if value > best_value {
                            best_value = value;
                            direction = dir;
                        }
                    }
                }

                let dir_char = match direction {
                    0 => "↑",
                    1 => "↗",
                    2 => "→",
                    3 => "↘",
                    4 => "↓",
                    5 => "↙",
                    6 => "←",
                    7 => "↖",
                    _ => "n"
                };
                if self.env.is_building(location) {
                    print!(" B");
                } else if self.env.station_location == location  {
                    print!(" S");
                } else {
                    print!(" {}", dir_char)
                }

            }
            print!("\n")
        }
        print!("\n")
    }
}
