use obstacle::ObstacleSet;

pub mod flip;
pub mod obstacle;
pub mod scene;

pub trait Fluid<const D: usize> {
    type Params;

    fn step(&mut self, dt: f32, params: &Self::Params, obstacles: &ObstacleSet<D>);
}
