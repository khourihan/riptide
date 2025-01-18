#[cfg(feature = "d2")]
mod flip_2d;
#[cfg(feature = "d3")]
mod flip_3d;
#[cfg(feature = "d2")]
mod mac_2d;
#[cfg(feature = "d3")]
mod mac_3d;

#[cfg(feature = "d2")]
pub use flip_2d::{FlipFluid, FlipFluidParams};
#[cfg(feature = "d3")]
pub use flip_3d::{FlipFluid, FlipFluidParams};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CellType {
    Fluid,
    Solid,
    Air,
}
