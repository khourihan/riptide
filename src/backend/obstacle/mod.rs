use glam::Vec2;

pub mod circle;

pub trait Obstacle {
    fn distance(&self, p: Vec2) -> f32;

    fn velocity(&self, p: Vec2, dt: f32) -> Vec2;
}

pub struct ObstacleSet {
    pub obstacles: Vec<Box<dyn Obstacle>>,
}

impl ObstacleSet {
    pub fn new(obstacles: Vec<Box<dyn Obstacle>>) -> Self {
        ObstacleSet {
            obstacles
        }
    }
}

impl Obstacle for ObstacleSet {
    fn distance(&self, p: Vec2) -> f32 {
        let mut dist = f32::MAX;

        for obstacle in self.obstacles.iter() {
            dist = dist.min(obstacle.distance(p));
        }

        dist
    }

    fn velocity(&self, p: Vec2, dt: f32) -> Vec2 {
        let mut dist = f32::MAX;
        let mut index = 0;

        for (i, obstacle) in self.obstacles.iter().enumerate() {
            let d = obstacle.distance(p);
            if dist > d {
                dist = d;
                index = i;
            }
        }

        self.obstacles[index].velocity(p, dt)
    }
}
