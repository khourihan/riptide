use std::collections::HashMap;

use crate::Vector;

#[cfg(feature = "d2")]
mod circle;

#[cfg(feature = "d2")]
pub use circle::Circle;

/// An obstacle for a fluid that is unaffected by buoyancy forces.
pub trait Obstacle {
    fn sdf(&self, p: Vector) -> Sdf;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sdf {
    pub distance: f32,
    pub gradient: Vector,
}

impl Sdf {
    pub fn new(distance: f32, gradient: Vector) -> Sdf {
        Sdf { distance, gradient }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ObstacleId(pub usize);

#[derive(Default)]
pub struct ObstacleSet {
    pub obstacles: HashMap<usize, Box<dyn Obstacle>>,
}

impl ObstacleSet {
    pub fn new(obstacles: HashMap<usize, Box<dyn Obstacle>>) -> Self {
        ObstacleSet {
            obstacles,
        }
    }
}

impl Obstacle for ObstacleSet {
    fn sdf(&self, p: Vector) -> Sdf {
        let mut dist = f32::MAX;
        let mut gradient = Vector::ZERO;

        for obstacle in self.obstacles.values() {
            let sd = obstacle.sdf(p);
            if dist > sd.distance {
                dist = sd.distance;
                gradient = sd.gradient;
            }
        }

        Sdf::new(dist, gradient)
    }
}
