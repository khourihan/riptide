use std::io::Write;

use thiserror::Error;

use crate::fluid::{scene::Scene, Fluid};

use super::as_bytes::AsBytes;

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

    pub fn encode_file_header<const D: usize, F, P>(&mut self, scene: &Scene<D, F, P>, fps: u32, num_steps: u64) -> Result<(), EncodingError>
    where 
        F: Fluid<D, Params = P>,
    {
        self.writer.write_all(&[D as u8])?;
        self.writer.write_all(&fps.to_ne_bytes())?;
        self.writer.write_all(&num_steps.to_ne_bytes())?;

        for i in 0..D {
            self.writer.write_all(&scene.size()[i].to_bytes())?
        }

        Ok(())
    }

    pub fn encode_section<const N: usize, T, I>(&mut self, len: usize, values: I) -> Result<(), EncodingError>
    where
        I: Iterator<Item = T>,
        T: AsBytes<N>,
    {
        self.writer.write_all(&(len as u64).to_ne_bytes())?;

        for v in values {
            self.writer.write_all(&v.to_bytes())?;
        }

        Ok(())
    }

    pub fn encode_step<const D: usize, F, P>(&mut self, scene: &Scene<D, F, P>) -> Result<(), EncodingError>
    where 
        F: Fluid<D, Params = P>,
    {
        scene.fluid.encode_state(self)?;

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum EncodingError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
