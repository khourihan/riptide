use std::io::Write;

use thiserror::Error;

use crate::fluid::scene::Scene;

pub struct FluidDataEncoder<W: Write> {
    writer: W,
    /// Whether or not to include velocity and roughness information in addition to the position of
    /// particles.
    verbose: bool,
}

impl<W: Write> FluidDataEncoder<W> {
    pub fn new(writer: W) -> FluidDataEncoder<W> {
        Self {
            writer,
            verbose: false,
        }
    }

    fn encode_header(&mut self) -> Result<(), EncodingError> {
        Ok(())
    }

    fn encode_step(&mut self, scene: &Scene) -> Result<(), EncodingError> {
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum EncodingError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
