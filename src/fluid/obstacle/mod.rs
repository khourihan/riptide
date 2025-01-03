use std::collections::HashMap;

use glam::{Vec2, Vec3};

pub mod circle;

/// A 2D obstacle for a fluid that is unaffected by buoyancy forces.
pub trait Obstacle2D {
    fn sdf(&self, p: Vec2) -> Sdf2D;
}

/// A 3D obstacle for a fluid that is unaffected by buoyancy forces.
pub trait Obstacle3D {
    fn sdf(&self, p: Vec3) -> Sdf3D;
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sdf2D {
    pub distance: f32,
    pub gradient: Vec2,
}

impl Sdf2D {
    pub fn new(distance: f32, gradient: Vec2) -> Sdf2D {
        Sdf2D { distance, gradient }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sdf3D {
    pub distance: f32,
    pub gradient: Vec3,
}

impl Sdf3D {
    pub fn new(distance: f32, gradient: Vec3) -> Sdf3D {
        Sdf3D { distance, gradient }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ObstacleId(pub usize);

#[derive(Default)]
pub struct ObstacleSet2D {
    pub obstacles: HashMap<usize, Box<dyn Obstacle2D>>,
}

impl ObstacleSet2D {
    pub fn new(obstacles: HashMap<usize, Box<dyn Obstacle2D>>) -> Self {
        ObstacleSet2D {
            obstacles,
        }
    }
}

impl Obstacle2D for ObstacleSet2D {
    fn sdf(&self, p: Vec2) -> Sdf2D {
        let mut dist = f32::MAX;
        let mut gradient = Vec2::ZERO;

        for obstacle in self.obstacles.values() {
            let sd = obstacle.sdf(p);
            if dist > sd.distance {
                dist = sd.distance;
                gradient = sd.gradient;
            }
        }

        Sdf2D::new(dist, gradient)
    }
}

#[derive(Default)]
pub struct ObstacleSet3D {
    pub obstacles: HashMap<usize, Box<dyn Obstacle3D>>,
}

impl ObstacleSet3D {
    pub fn new(obstacles: HashMap<usize, Box<dyn Obstacle3D>>) -> Self {
        ObstacleSet3D {
            obstacles,
        }
    }
}

impl Obstacle3D for ObstacleSet3D {
    fn sdf(&self, p: Vec3) -> Sdf3D {
        let mut dist = f32::MAX;
        let mut gradient = Vec3::ZERO;

        for obstacle in self.obstacles.values() {
            let sd = obstacle.sdf(p);
            if dist > sd.distance {
                dist = sd.distance;
                gradient = sd.gradient;
            }
        }

        Sdf3D::new(dist, gradient)
    }
}
