pub mod d2;
pub mod d3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CellType {
    Fluid,
    Solid,
    Air,
}
