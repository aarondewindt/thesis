use std::ops;
use std::fmt;

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
pub struct Point(pub i8, pub i8);

impl Point {
    pub fn clone(&self) -> Point {
        Point(self.0, self.1)
    }
}

impl ops::Add for Point {
    type Output = Point;
    fn add(self, other: Point) -> Point {
        Point(
            self.0 + other.0,
            self.1 + other.1
        )
    }
}

impl ops::Add<(i8, i8)> for Point {
    type Output = Point;
    fn add(self, other: (i8, i8)) -> Point {
        Point(
            self.0 + other.0,
            self.1 + other.1
        )
    }
}

impl ops::AddAssign for Point {
    fn add_assign(&mut self, other: Point) {
        *self = Point(
            self.0 + other.0,
            self.1 + other.1,
        );
    }
}

impl ops::AddAssign<(i8, i8)> for Point {
    fn add_assign(&mut self, other: (i8, i8)) {
        *self = Point(
            self.0 + other.0,
            self.1 + other.1,
        );
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Point({}, {})", self.0, self.1)
    }
}
