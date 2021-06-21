extern crate rand;
extern crate csv;

mod point;
mod environment;
mod agent;
mod experiment_1;

use point::Point;
use environment::Environment;
use agent::Agent;
use experiment_1::experiment_1;


fn main() {
    experiment_1::experiment_1();
    return;


    // Original map
    let mut agent = Agent::new(
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
        0.2,
        0.9,
        0.001,
    );

    // Larger map
    // let mut agent = Agent::new(
    //     Environment::new(
    //         Point(0, 0),
    //         1,
    //         Point(9,9),
    //         vec![
    //             Point(0, 2),
    //             Point(5, 0),
    //             Point(3, 4),
    //             Point(9, 0),
    //             Point(2, 7),
    //             Point(8, 4),
    //         ],
    //         (10, 10)
    //     ),
    //     0.2,
    //     0.9
    // );

    agent.env._print_map();

    // Run simulation
    for _ in 0..=1000000 {
        agent.step();
    }

    // Print results
    for i in 1..=6 {
        println!("mem: {}", i);
//        agent._print_values_for_memory(i);
        agent.print_greedy_move(i);
    }


}






































