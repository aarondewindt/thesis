use point::Point;
use environment::Environment;
use agent::Agent;

use csv::Writer;

const N_AGENTS: usize = 1000;
const N_ITERATIONS: usize = 5000;

pub fn experiment_1() {
    // run_batch(0.1, 0.9, "./results.csv");

    // run_batch(1., 0.9, "./results_10_9.csv");
    // run_batch(0.1, 0.9, "./results_1_9.csv");
    // run_batch(0.01, 0.9, "./results_01_9.csv");
    // run_batch(0.001, 0.9, "./results_001_9.csv");

    // run_batch(0.01, 0.0, "./results_01_00.csv");
    run_batch(0.01, 0.1, "./results_01_01.csv");
    // run_batch(0.01, 0.5, "./results_01_05.csv");
    // run_batch(0.01, 0.9, "./results_01_09.csv");
    // run_batch(0.01, 1.0, "./results_01_10.csv");
}

fn run_batch(alpha: f64, gamma: f64, result_path: &str) {
    let mut rewards = Vec::new();

    let mut agents = Vec::new();
    for _ in 0..=N_AGENTS {
        agents.push(Agent::new(
            Environment::new(
                Point(0, 0),
                1,
                Point(5,5),
                vec![
                    Point(0, 2),
                    Point(5, 0),
                    Point(3, 4),
                ],
                (6, 6)
            ),
            0.1,
            gamma,
            alpha
        ));
    }

    for _ in 0..N_ITERATIONS {
        let mut iteration_rewards = Vec::new();
        for i_agent in 0..N_AGENTS {
            iteration_rewards.push(agents[i_agent].step().to_string());
        }
        rewards.push(iteration_rewards)
    }


    // Print results
//     for i in 1..=6 {
//         println!("mem: {}", i);
// //        agent._print_values_for_memory(i);
//         agents[0].print_greedy_move(i);
//     }

    let mut wrt = csv::Writer::from_path(result_path).unwrap();
    let mut header = Vec::new();
    for i_agent in 0..N_AGENTS {
        header.push(format!("agent_{}", i_agent));
    }
    wrt.write_record(header);
    for record in rewards {
        wrt.write_record(record);
    }

    println!("Results written to: {}", result_path);

}