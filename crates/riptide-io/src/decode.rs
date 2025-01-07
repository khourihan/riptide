use std::{fs::File, io::{BufRead, BufReader}, mem::{self, MaybeUninit}, path::PathBuf};

use smallvec::SmallVec;
use thiserror::Error;

pub struct FluidDataDecoder {
    /// The path to the directory into which the fluid data resides.
    path: PathBuf,
    dim: u8,
    num_frames: u64,
    current_frame: u64,
}

impl FluidDataDecoder {
    pub fn new(path: PathBuf) -> FluidDataDecoder {
        Self {
            path,
            dim: 0,
            num_frames: 0,
            current_frame: 0,
        }
    }

    fn read_value<const N: usize, T, R: BufRead>(reader: &mut R) -> Result<T, DecodingError> {
        let mut bytes = [0; N];
        reader.read_exact(&mut bytes)?;

        let mut to: MaybeUninit<T> = MaybeUninit::uninit();

        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), to.as_mut_ptr().cast::<u8>(), N);
            Ok(to.assume_init())
        }
    }

    fn read_values<T, R: BufRead>(reader: &mut R, count: usize) -> Result<Vec<T>, DecodingError> {
        let mut bytes = vec![0; mem::size_of::<T>() * count];
        reader.read_exact(&mut bytes)?;

        Ok(bytes.chunks_exact(mem::size_of::<T>()).map(|b| {
            let mut to: MaybeUninit<T> = MaybeUninit::uninit();

            unsafe {
                std::ptr::copy_nonoverlapping(b.as_ptr(), to.as_mut_ptr().cast::<u8>(), mem::size_of::<T>());
                to.assume_init()
            }
        }).collect())
    }

    fn frame_path(&self, frame: u64) -> PathBuf {
        let max_digits = (self.num_frames - 1).checked_ilog10().unwrap_or(0) + 1;
        let zeros = max_digits - (frame.checked_ilog10().unwrap_or(0) + 1);

        self.path.join(format!("{}{frame}.dat", "0".repeat(zeros as usize)))
    }

    pub fn decode_metadata(&mut self) -> Result<FluidMetadata, DecodingError> {
        let path = self.path.join("_meta");
        let mut reader = BufReader::new(File::open(path)?);

        let dim = Self::read_value::<1, u8, _>(&mut reader)?;
        let fps = Self::read_value::<4, u32, _>(&mut reader)?;
        let num_frames = Self::read_value::<8, u64, _>(&mut reader)?;
        let mut size: SmallVec<[_; 4]> = SmallVec::new();

        for _ in 0..dim {
            let v = Self::read_value::<4, f32, _>(&mut reader)?;
            size.push(v);
        }

        self.dim = dim;
        self.num_frames = num_frames;

        Ok(FluidMetadata {
            dim,
            fps,
            num_frames,
            size,
        })
    }

    pub fn decode_frame(&mut self) -> Result<Option<FluidFrameData>, DecodingError> {
        if self.current_frame >= self.num_frames {
            return Ok(None)
        }

        let path = self.frame_path(self.current_frame);
        let mut reader = BufReader::new(File::open(path)?);

        let n_particles = self.dim as u64 * Self::read_value::<8, u64, _>(&mut reader)?;
        let positions = Self::read_values::<f32, _>(&mut reader, n_particles as usize)?;

        self.current_frame += 1;

        Ok(Some(FluidFrameData {
            positions: FluidDataArray(positions),
        }))
    }

    pub fn reset(&mut self) {
        self.current_frame = 0;
    }
}

pub struct FluidMetadata {
    pub dim: u8,
    pub fps: u32,
    pub num_frames: u64,
    pub size: SmallVec<[f32; 4]>,
}

impl FluidMetadata {
    pub fn size<const D: usize>(&self) -> [f32; D] {
        self.size[..D].try_into().unwrap()
    }
}

pub struct FluidFrameData {
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
