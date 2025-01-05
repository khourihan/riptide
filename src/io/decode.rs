use std::{io::Read, mem::{self, MaybeUninit}};

use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use smallvec::SmallVec;
use thiserror::Error;

pub struct FluidDataDecoder<R: Read> {
    reader: R,
}

impl<R: Read> FluidDataDecoder<R> {
    pub fn new(reader: R) -> FluidDataDecoder<R> {
        Self {
            reader,
        }
    }

    fn read_value<T>(&mut self) -> Result<T, DecodingError> {
        let mut bytes = vec![0; mem::size_of::<T>()];
        self.reader.read_exact(&mut bytes)?;

        let mut to: MaybeUninit<T> = MaybeUninit::uninit();

        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), to.as_mut_ptr().cast::<u8>(), mem::size_of::<T>());
            Ok(to.assume_init())
        }
    }

    pub fn decode(&mut self) -> Result<FluidData, DecodingError> {
        let dim = self.read_value::<u8>()?;
        let fps = self.read_value::<u32>()?;
        let n_frames = self.read_value::<u64>()?;
        let mut size: SmallVec<[_; 4]> = SmallVec::new();

        for _ in 0..dim {
            let v = self.read_value::<f32>()?;
            size.push(v);
        }

        let mut frames = Vec::new();

        let bar_template = "Decoding Fluid Data {spinner:.green} [{elapsed}] [{bar:50.white/white}] {pos}/{len} ({eta})";
        let style = ProgressStyle::with_template(bar_template).unwrap()
            .progress_chars("=> ").tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");
        let progress = ProgressBar::new(n_frames).with_style(style);

        for _ in (0..n_frames).progress_with(progress) {
            let n_particles = dim as u64 * self.read_value::<u64>()?;
            let mut count = 0;
            let positions = std::iter::from_fn(|| {
                count += 1;

                if count <= n_particles {
                    Some(self.read_value::<f32>())
                } else {
                    None
                }
            }).collect::<Result<Vec<_>, _>>()?;

            frames.push(FluidDataFrame {
                positions: FluidDataArray(positions),
            })
        }

        Ok(FluidData {
            dim,
            fps,
            size,
            frames,
        })
    }
}

pub struct FluidData {
    pub dim: u8,
    pub fps: u32,
    size: SmallVec<[f32; 4]>,
    pub frames: Vec<FluidDataFrame>,
}

impl FluidData {
    pub fn size<const D: usize>(&self) -> [f32; D] {
        self.size[..D].try_into().unwrap()
    }
}

pub struct FluidDataFrame {
    pub positions: FluidDataArray,
}

pub struct FluidDataArray(Vec<f32>);

impl FluidDataArray {
    pub fn iter<const D: usize>(&self) -> impl Iterator<Item = [f32; D]> + use<'_, D> {
        self.0.chunks_exact(D).map(|chunk| <[f32; D]>::try_from(chunk).unwrap())
    }

    pub fn get<const D: usize>(&self, i: usize) -> [f32; D] {
        self.0[i..i + D].try_into().unwrap()
    }
}

#[derive(Debug, Error)]
pub enum DecodingError {
    #[error(transparent)]
    Io(#[from] std::io::Error)
}
