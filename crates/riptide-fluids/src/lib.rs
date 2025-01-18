use obstacle::ObstacleSet;

pub mod flip;
pub mod obstacle;
pub mod scene;

#[cfg(all(feature = "d2", feature = "d3"))]
compile_error!("Features 'd2' and 'd3' are mutually exclusive and cannot be enabled together");
#[cfg(not(any(feature = "d2", feature = "d3")))]
compile_error!("Either feature 'd2' or 'd3' must be enabled");

#[cfg(feature = "d2")]
pub(crate) type Vector = glam::Vec2;
#[cfg(feature = "d3")]
pub(crate) type Vector = glam::Vec3;

pub trait Fluid {
    type Params;

    fn step(&mut self, dt: f32, params: &Self::Params, obstacles: &ObstacleSet);

    fn particle_radius(&self) -> f32;
}
