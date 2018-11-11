extern crate rand;

mod point;
mod environment;
mod agent;

use point::Point;
use environment::Environment;
use agent::Agent;




fn main() {
    // Original map
    // let mut agent = Agent::new(
    //     Environment::new(
    //         Point(0, 0),
    //         1,
    //         Point(5,5),
    //         vec![
    //             Point(0, 2),
    //             Point(5, 0),
    //             Point(3, 4),
    //         ],
    //         (6, 6)
    //     ),
    //     0.2,
    //     0.9
    // );

    // Larger map
    let mut agent = Agent::new(
        Environment::new(
            Point(0, 0),
            1,
            Point(9,9),
            vec![
                Point(0, 2),
                Point(5, 0),
                Point(3, 4),
                Point(9, 0),
                Point(2, 7),
                Point(8, 4),
            ],
            (10, 10)
        ),
        0.2,
        0.9
    );

    agent.env._print_map();

    // Run simulation
    for _ in 0..=100000 {
        agent.step();
    }

    // Print results
    for i in 1..=6 {
        println!("mem: {}", i);
//        agent._print_values_for_memory(i);
        agent.print_greedy_move(i);
    }


}






































