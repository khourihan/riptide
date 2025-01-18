use std::io::Write;

use encode::{EncodingError, FluidFrameEncoder};
use riptide_fluids::flip::FlipFluid;

#[cfg(feature = "d2")]
use glam::Vec2;
#[cfg(feature = "d3")]
use glam::Vec3;

pub mod encode;
pub mod decode;
pub mod as_bytes;

pub trait EncodeFluid {
    fn encode_state<W: Write>(&self, encoder: &mut FluidFrameEncoder<W>) -> Result<(), EncodingError>;
}


impl EncodeFluid for FlipFluid {
    fn encode_state<W: std::io::Write>(&self, encoder: &mut FluidFrameEncoder<W>) -> Result<(), EncodingError> {
        let delta = self.spacing();

        encoder.encode_section(self.positions.len(), self.positions.iter().copied())?;
        encoder.encode_section(self.positions.len(), self.positions.iter().map(|&p| {
            #[cfg(feature = "d2")]
            {
                let gx = (self.sample_density(p + Vec2::new(delta, 0.0))
                    - self.sample_density(p - Vec2::new(delta, 0.0))) / (2.0 * delta);
                let gy = (self.sample_density(p + Vec2::new(0.0, delta))
                    - self.sample_density(p - Vec2::new(0.0, delta))) / (2.0 * delta);

                Vec2::new(gx, gy)
            }
            #[cfg(feature = "d3")]
            {
                let gx = (self.sample_density(p + Vec3::new(delta, 0.0, 0.0))
                    - self.sample_density(p - Vec3::new(delta, 0.0, 0.0))) / (2.0 * delta);
                let gy = (self.sample_density(p + Vec3::new(0.0, delta, 0.0))
                    - self.sample_density(p - Vec3::new(0.0, delta, 0.0))) / (2.0 * delta);
                let gz = (self.sample_density(p + Vec3::new(0.0, 0.0, delta))
                    - self.sample_density(p - Vec3::new(0.0, 0.0, delta))) / (2.0 * delta);

                Vec3::new(gx, gy, gz)
            }
        }))?;

        Ok(())
    }
}
