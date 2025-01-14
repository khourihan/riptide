pub mod flip_2d;
pub mod flip_3d;
mod mac_2d;
mod mac_3d;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CellType {
    Fluid,
    Solid,
    Air,
}
