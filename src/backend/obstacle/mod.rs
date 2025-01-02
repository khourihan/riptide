use std::collections::HashMap;

use glam::Vec2;

pub mod circle;

/// An obstacle for the fluid that is unaffected by buoyancy forces.
pub trait Obstacle {
    fn sdf(&self, p: Vec2) -> Sdf;
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sdf {
    pub distance: f32,
    pub gradient: Vec2,
}

impl Sdf {
    pub fn new(distance: f32, gradient: Vec2) -> Sdf {
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
    fn sdf(&self, p: Vec2) -> Sdf {
        let mut dist = f32::MAX;
        let mut gradient = Vec2::ZERO;

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
