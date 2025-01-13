use glam::{UVec2, Vec2};
use ndarray::{Array2, ArrayView2};

use super::CellType;

#[derive(Debug, Clone)]
pub struct MacGrid2D {
    /// Size of the grid, in meters.
    pub size: Vec2,
    /// Size of the grid, in cells.
    pub grid_size: UVec2,
    /// Number of cells in the X direction.
    pub nx: usize,
    /// Number of cells in the Y direction.
    pub ny: usize,
    /// Cell size.
    pub spacing: f32,
    /// 1.0 / spacing.
    pub inv_spacing: f32,
    
    /// Grid velocities.
    pub u: Array2<f32>,
    pub v: Array2<f32>,
    /// Grid velocity weights.
    pub weight_u: Array2<f32>,
    pub weight_v: Array2<f32>,
    /// Intermediate grid velocities.
    pub u_star: Array2<f32>,
    pub v_star: Array2<f32>,
    /// Pressure of the grid.
    pub pressure: Array2<f32>,
    /// Solid grid cells.
    pub solid: Array2<bool>,
    /// Grid cell types (`Fluid`, `Solid`, or `Air`).
    pub cell_type: Array2<CellType>,
    /// Grid densities.
    pub densities: Array2<f32>,
}

impl MacGrid2D {
    pub fn new(
        size: Vec2,
        spacing: f32,
    ) -> Self {
        let grid_size = (size / spacing).floor().as_uvec2() + 1;
        let h = f32::max(size.x / grid_size.x as f32, size.y / grid_size.y as f32);
        let nx = grid_size.x as usize;
        let ny = grid_size.y as usize;

        let u = Array2::from_elem((nx + 1, ny), 0.0);
        let v = Array2::from_elem((nx, ny + 1), 0.0);
        let weight_u = Array2::from_elem((nx + 1, ny), 0.0);
        let weight_v = Array2::from_elem((nx, ny + 1), 0.0);
        let u_star = Array2::from_elem((nx + 1, ny), 0.0);
        let v_star = Array2::from_elem((nx, ny + 1), 0.0);
        let pressure = Array2::from_elem((nx, ny), 0.0);
        let mut solid = Array2::from_elem((nx, ny), false);
        let cell_type = Array2::from_elem((nx, ny), CellType::Air);
        let densities = Array2::from_elem((nx, ny), 0.0);

        for i in 0..nx {
            solid[(i, 0)] = true;
            solid[(i, ny - 1)] = true;
        }

        for j in 0..ny {
            solid[(0, j)] = true;
            solid[(nx - 1, j)] = true;
        }

        Self {
            size,
            grid_size,
            nx,
            ny,
            spacing: h,
            inv_spacing: h.recip(),
            u,
            v,
            weight_u,
            weight_v,
            u_star,
            v_star,
            pressure,
            solid,
            cell_type,
            densities,
        }
    }

    #[inline]
    pub fn get_bilerp_u(&self, pos: Vec2) -> f32 {
        get_bilerp_x(self.u.view(), pos, self.spacing, self.inv_spacing, self.grid_size, self.size)
    }

    #[inline]
    pub fn get_bilerp_v(&self, pos: Vec2) -> f32 {
        get_bilerp_y(self.v.view(), pos, self.spacing, self.inv_spacing, self.grid_size, self.size)
    }

    #[inline]
    pub fn get_bilerp_u_star(&self, pos: Vec2) -> f32 {
        get_bilerp_x(self.u_star.view(), pos, self.spacing, self.inv_spacing, self.grid_size, self.size)
    }

    #[inline]
    pub fn get_bilerp_v_star(&self, pos: Vec2) -> f32 {
        get_bilerp_y(self.v_star.view(), pos, self.spacing, self.inv_spacing, self.grid_size, self.size)
    }

    #[inline]
    pub fn assign_uv_star(&mut self) {
        self.u_star.assign(&self.u);
        self.v_star.assign(&self.v);
    }
}

fn get_bilerp_x(
    vals: ArrayView2<f32>,
    p: Vec2,
    spacing: f32,
    inv_spacing: f32,
    grid_size: UVec2,
    size: Vec2,
) -> f32 {
    let pi = (p * inv_spacing).floor().as_uvec2().clamp(UVec2::ZERO, grid_size - 1);

    if p.y >= 0.0 && p.y <= size.y - spacing {
        let ix1 = pi.x;
        let ix2 = ix1 + 1;
        let (iy1, iy2) = if p.y > pi.y as f32 * spacing {
            (pi.y, pi.y.min(grid_size.y - 2) + 1)
        } else {
            (pi.y.max(1) - 1, pi.y)
        };

        let x1 = (ix1 as f32 - 0.5) * spacing;
        let x2 = (ix2 as f32 - 0.5) * spacing;
        let y1 = iy1 as f32 * spacing;
        let y2 = iy2 as f32 * spacing;

        let u00 = vals[(ix1 as usize, iy1 as usize)];
        let u01 = vals[(ix1 as usize, iy2 as usize)];
        let u10 = vals[(ix2 as usize, iy1 as usize)];
        let u11 = vals[(ix2 as usize, iy2 as usize)];

        ((u00 * (x2 - p.x) * (y2 - p.y))
            + u10 * (p.x - x1) * (y2 - p.y)
            + u01 * (x2 - p.x) * (p.y - y1)
            + u11 * (p.x - x1) * (p.y - y1)) / ((x2 - x1) * (y2 - y1))
    } else if p.y < 0.0 {
        let ix1 = pi.x;
        let ix2 = ix1 + 1;
        let x1 = (ix1 as f32 - 0.5) * spacing;
        let x2 = (ix2 as f32 - 0.5) * spacing;
        let u00 = vals[(ix1 as usize, 0)];
        let u10 = vals[(ix2 as usize, 0)];
        u00 * (1.0 - (p.x - x1) / (x2 - x1)) + u10 * ((p.x - x1) / (x2 - x1))
    } else {
        let ix1 = pi.x;
        let ix2 = ix1 + 1;
        let x1 = (ix1 as f32 - 0.5) * spacing;
        let x2 = (ix2 as f32 - 0.5) * spacing;
        let u00 = vals[(ix1 as usize, grid_size.y as usize - 1)];
        let u10 = vals[(ix2 as usize, grid_size.y as usize - 1)];
        u00 * (1.0 - (p.x - x1) / (x2 - x1)) + u10 * ((p.x - x1) / (x2 - x1))
    }
}

fn get_bilerp_y(
    vals: ArrayView2<f32>,
    p: Vec2,
    spacing: f32,
    inv_spacing: f32,
    grid_size: UVec2,
    size: Vec2,
) -> f32 {
    let pi = (p * inv_spacing).floor().as_uvec2().clamp(UVec2::ZERO, grid_size - 1);

    if p.x >= 0.0 && p.x <= size.x - spacing {
        let iy1 = pi.y;
        let iy2 = iy1 + 1;
        let (ix1, ix2) = if p.x > pi.x as f32 * spacing {
            (pi.x, pi.x.min(grid_size.x - 2) + 1)
        } else {
            (pi.x.max(1) - 1, pi.x)
        };

        let y1 = (iy1 as f32 - 0.5) * spacing;
        let y2 = (iy2 as f32 - 0.5) * spacing;
        let x1 = ix1 as f32 * spacing;
        let x2 = ix2 as f32 * spacing;

        let v00 = vals[(ix1 as usize, iy1 as usize)];
        let v01 = vals[(ix1 as usize, iy2 as usize)];
        let v10 = vals[(ix2 as usize, iy1 as usize)];
        let v11 = vals[(ix2 as usize, iy2 as usize)];

        ((v00 * (x2 - p.x) * (y2 - p.y))
            + v10 * (p.x - x1) * (y2 - p.y)
            + v01 * (x2 - p.x) * (p.y - y1)
            + v11 * (p.x - x1) * (p.y - y1)) / ((x2 - x1) * (y2 - y1))
    } else if p.x < 0.0 {
        let iy1 = pi.y;
        let iy2 = iy1 + 1;
        let y1 = (iy1 as f32 - 0.5) * spacing;
        let y2 = (iy2 as f32 - 0.5) * spacing;
        let v00 = vals[(0, iy1 as usize)];
        let v10 = vals[(0, iy2 as usize)];
        v00 * (1.0 - (p.y - y1) / (y2 - y1)) + v10 * ((p.y - y1) / (y2 - y1))
    } else {
        let iy1 = pi.y;
        let iy2 = iy1 + 1;
        let y1 = (iy1 as f32 - 0.5) * spacing;
        let y2 = (iy2 as f32 - 0.5) * spacing;
        let v00 = vals[(grid_size.x as usize - 1, iy1 as usize)];
        let v10 = vals[(grid_size.x as usize - 1, iy2 as usize)];
        v00 * (1.0 - (p.y - y1) / (y2 - y1)) + v10 * ((p.y - y1) / (y2 - y1))
    }
}
