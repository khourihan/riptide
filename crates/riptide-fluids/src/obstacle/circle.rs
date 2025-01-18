use super::{Obstacle, Sdf, Vector};

#[derive(Debug, Clone, Copy)]
pub struct Circle {
    pub position: Vector,
    pub radius: f32,
}

impl Circle {
    pub fn new(pos: Vector, radius: f32) -> Self {
        Circle {
            position: pos,
            radius,
        }
    }

    /// Sets the position of the circle. Should be called every time step.
    pub fn set_position(&mut self, pos: Vector) {
        self.position = pos;
    }
}

impl Obstacle for Circle {
    fn sdf(&self, p: Vector) -> Sdf {
        let d = (p - self.position).length();

        Sdf {
            distance: d - self.radius,
            gradient: (p - self.position) / d,
        }
    }
}
