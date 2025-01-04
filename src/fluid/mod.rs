use std::io::Write;

use obstacle::ObstacleSet;

use crate::io::encode::{EncodingError, FluidDataEncoder};

pub mod flip;
pub mod obstacle;
pub mod scene;

pub trait Fluid<const D: usize> {
    type Params;

    fn step(&mut self, dt: f32, params: &Self::Params, obstacles: &ObstacleSet<D>);

    fn encode_state<W: Write>(&self, encoder: &mut FluidDataEncoder<W>) -> Result<(), EncodingError>;
}
