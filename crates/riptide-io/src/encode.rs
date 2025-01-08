use std::{fs::File, io::{BufWriter, Write}, path::PathBuf};

use thiserror::Error;

use riptide_fluids::{scene::Scene, Fluid};

use crate::EncodeFluid;

use super::as_bytes::AsBytes;

pub struct FluidDataEncoder {
    /// The path to the directory into which the fluid data will be placed.
    path: PathBuf,
    num_frames: u64,
    fps: u32,
    current_frame: u64,
}

impl FluidDataEncoder {
    pub fn new(path: PathBuf, num_frames: u64, fps: u32) -> Result<FluidDataEncoder, EncodingError> {
        std::fs::create_dir(&path)?;

        Ok(Self {
            path,
            num_frames,
            fps,
            current_frame: 0,
        })
    }

    fn frame_path(&self, frame: u64) -> PathBuf {
        let max_digits = (self.num_frames - 1).checked_ilog10().unwrap_or(0) + 1;
        let zeros = max_digits - (frame.checked_ilog10().unwrap_or(0) + 1);

        self.path.join(format!("{}{frame}.dat", "0".repeat(zeros as usize)))
    }

    pub fn encode_metadata<const D: usize, F, P>(&mut self, scene: &Scene<D, F, P>) -> Result<(), EncodingError>
    where 
        F: Fluid<D, Params = P>,
    {
        let path = self.path.join("_meta");
        let mut writer = File::create(path)?;

        writer.write_all(&[D as u8])?;
        writer.write_all(&self.fps.to_ne_bytes())?;
        writer.write_all(&self.num_frames.to_ne_bytes())?;

        writer.write_all(&scene.fluid.particle_radius().to_ne_bytes())?;

        for i in 0..D {
            writer.write_all(&scene.size()[i].to_bytes())?;
        }

        Ok(())
    }

    pub fn encode_frame<const D: usize, F, P>(&mut self, scene: &Scene<D, F, P>) -> Result<(), EncodingError>
    where 
        F: EncodeFluid,
    {
        let path = self.frame_path(self.current_frame);
        let writer = BufWriter::new(File::create(path)?);

        scene.fluid.encode_state(&mut FluidFrameEncoder { writer })?;

        self.current_frame += 1;

        Ok(())
    }
}

pub struct FluidFrameEncoder<W: Write> {
    writer: BufWriter<W>,
}

impl<W: Write> FluidFrameEncoder<W> {
    pub fn encode_section<const N: usize, T, I>(&mut self, len: usize, values: I) -> Result<(), EncodingError>
    where
        I: Iterator<Item = T>,
        T: AsBytes<N>,
    {
        self.writer.write_all(&(len as u64).to_ne_bytes())?;

        let bytes: Vec<_> = values.flat_map(|v| v.to_bytes()).collect();
        self.writer.write_all(&bytes)?;

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum EncodingError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
