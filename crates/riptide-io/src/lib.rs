use std::io::Write;

use encode::{EncodingError, FluidFrameEncoder};
use riptide_fluids::flip::{d2::FlipFluid2D, d3::FlipFluid3D};

pub mod encode;
pub mod decode;
pub mod as_bytes;

pub trait EncodeFluid {
    fn encode_state<W: Write>(&self, encoder: &mut FluidFrameEncoder<W>) -> Result<(), EncodingError>;
}

impl EncodeFluid for FlipFluid2D {
    fn encode_state<W: std::io::Write>(&self, encoder: &mut FluidFrameEncoder<W>) -> Result<(), EncodingError> {
        encoder.encode_section(self.positions.len(), self.positions.as_slice().unwrap())?;

        Ok(())
    }
}

impl EncodeFluid for FlipFluid3D {
    fn encode_state<W: std::io::Write>(&self, encoder: &mut FluidFrameEncoder<W>) -> Result<(), EncodingError> {
        encoder.encode_section(self.positions.len(), self.positions.as_slice().unwrap())?;

        Ok(())
    }
}
