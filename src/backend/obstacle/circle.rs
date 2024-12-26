use glam::Vec2;

use super::Obstacle;

#[derive(Debug, Clone, Copy)]
pub struct Circle {
    pub pos: Vec2,
    prev_pos: Vec2,
    r: f32,
}

impl Circle {
    pub fn new(pos: Vec2, radius: f32) -> Self {
        Circle {
            prev_pos: pos,
            pos,
            r: radius,
        }
    }

    /// Sets the position of the circle. Should be called every time step.
    pub fn set_position(&mut self, pos: Vec2) {
        self.prev_pos = self.pos;
        self.pos = pos;
    }
}

impl Obstacle for Circle {
    fn distance(&self, p: Vec2) -> f32 {
        (p - self.pos).length() - self.r
    }

    fn velocity(&self, _p: Vec2, dt: f32) -> Vec2 {
        (self.pos - self.prev_pos) / dt
    }
}
