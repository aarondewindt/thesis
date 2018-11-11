extern crate rand;

use point::Point;
use std::fmt;
use rand::prelude::*;


#[derive(Eq, PartialEq, Hash, Copy, Clone)]
pub struct State {
    pub location: Point,
    pub memory: i8
}
impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "State({}, {})", self.location, self.memory)
    }
}

pub struct Environment {
    pub location: Point,
    pub memory: i8,
    pub station_location: Point,
    pub building_locations: Vec<Point>,
    pub field_size: (i8, i8),
    rng: ThreadRng,
}

impl Environment {
    pub fn new(location: Point, memory: i8, station_location: Point,
               building_locations: Vec<Point>, field_size: (i8, i8)) -> Environment{
        Environment {
            location,
            memory: match memory {
                1 ... 6 => memory,
                _ => panic!("Invalid memory value")
            },
            station_location,
            building_locations,
            field_size,
            rng: thread_rng()
        }
    }

    pub fn next_location(&self, direction: i8) -> Point {
        self.location + match direction {
            0 => Point(0, 1),
            1 => Point(1, 1),
            2 => Point(1, 0),
            3 => Point(1, -1),
            4 => Point(0, -1),
            5 => Point(-1, -1),
            6 => Point(-1, 0),
            7 => Point(-1, 1),
            _ => panic!("Invalid direction '{}'", direction)
        }
    }

    pub fn act(&mut self, direction: i8) -> f64 {
        self.location = self.next_location(direction);

        if self.out_of_bounds() {
            return -1.0;

        } else if self.is_building(self.location) {
            return match self.memory {
                1 ... 5 => {
                    self.memory += 1;
                    8.0
                },
                6 => 0.0,
                _ => panic!("Invalid memory value")
            };

        }else if self.location == self.station_location {
            let reward: f64 = (self.memory - 1).pow(2).into();
            self.memory = 1;
            return reward;
        } else {
            return 0.0;
        }
    }

    pub fn finish(&mut self, prev_state: State) {
        if self.out_of_bounds() {
            self.location = prev_state.location

        } else if self.is_building(self.location) {
            self.location = Point(
                self.rng.gen_range(0, self.field_size.0),
                self.rng.gen_range(0, self.field_size.1),
            );
        }else if self.location == self.station_location {
            self.location = Point(
                self.rng.gen_range(0, self.field_size.0),
                self.rng.gen_range(0, self.field_size.1),
            );
        }
    }

    pub fn is_building(&self, location: Point) -> bool {
        self.building_locations.iter().any(|x| *x == location)
    }

    pub fn out_of_bounds(&self) -> bool {
        (self.location.0 < 0)
            | (self.location.1 < 0)
            | (self.location.0 >= self.field_size.0)
            | (self.location.1 >= self.field_size.1)
    }

    pub fn state(&self) -> State {
        State{
            location: self.location.clone(),
            memory: self.memory,
        }
    }

    pub fn _print_map(&self) {
        println!("mem: {}", self.memory);
        for y in (0..self.field_size.1).rev() {
            for x in 0..self.field_size.0 {
                let point = Point(x, y);
                print!("\t{}", match (
                    point == self.location,
                    point == self.station_location,
                    self.is_building(point)
                ) {
                    (false, false, false) => "___",
                    (true, false, false) => "Q__",
                    (false, true, false) => "_S_",
                    (false, false, true) => "__B",
                    (true, false, true) => "Q_B",
                    (true, true, false) => "QS_",
                    (true, true, true) => "QSB",
                    (false, true, true) => "_SB",
                })
            }
            print!("\n")
        }
        print!("\n")
    }
}
