use glam::Vec2;

use super::{Obstacle, Sdf};

#[derive(Debug, Clone, Copy)]
pub struct Circle {
    pub position: Vec2,
    pub radius: f32,
}

impl Circle {
    pub fn new(pos: Vec2, radius: f32) -> Self {
        Circle {
            position: pos,
            radius,
        }
    }

    /// Sets the position of the circle. Should be called every time step.
    pub fn set_position(&mut self, pos: Vec2) {
        self.position = pos;
    }
}

impl Obstacle<2> for Circle {
    fn sdf(&self, p: [f32; 2]) -> Sdf<2> {
        let p: Vec2 = p.into();
        let d = (p - self.position).length();

        Sdf {
            distance: d - self.radius,
            gradient: ((p - self.position) / d).into(),
        }
    }
}
