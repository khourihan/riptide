use std::collections::HashMap;

pub mod circle;

/// An obstacle for a fluid that is unaffected by buoyancy forces.
pub trait Obstacle<const D: usize> {
    fn sdf(&self, p: [f32; D]) -> Sdf<D>;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sdf<const D: usize> {
    pub distance: f32,
    pub gradient: [f32; D],
}

impl<const D: usize> Sdf<D> {
    pub fn new(distance: f32, gradient: [f32; D]) -> Sdf<D> {
        Sdf { distance, gradient }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ObstacleId(pub usize);

#[derive(Default)]
pub struct ObstacleSet<const D: usize> {
    pub obstacles: HashMap<usize, Box<dyn Obstacle<D>>>,
}

impl<const D: usize> ObstacleSet<D> {
    pub fn new(obstacles: HashMap<usize, Box<dyn Obstacle<D>>>) -> Self {
        ObstacleSet {
            obstacles,
        }
    }
}

impl<const D: usize> Obstacle<D> for ObstacleSet<D> {
    fn sdf(&self, p: [f32; D]) -> Sdf<D> {
        let mut dist = f32::MAX;
        let mut gradient = [0.0; D];

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
