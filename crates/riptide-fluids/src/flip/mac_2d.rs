use glam::{UVec2, Vec2};

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
    
    /// Grid velocities in the X direction.
    pub u: Vec<f32>,
    /// Grid velocities in the Y direction.
    pub v: Vec<f32>,
    /// Grid velocity weights in the X direction.
    pub weight_u: Vec<f32>,
    /// Grid velocity weights in the Y direction.
    pub weight_v: Vec<f32>,
    /// Intermediate grid velocities in the X direction.
    pub u_star: Vec<f32>,
    /// Intermediate grid velocities in the Y direction.
    pub v_star: Vec<f32>,
    /// Pressure of the grid.
    pub pressure: Vec<f32>,
    /// Solid grid cells.
    pub solid: Vec<bool>,
    /// Grid cell types (`Fluid`, `Solid`, or `Air`).
    pub cell_type: Vec<CellType>,
    /// Grid densities.
    pub densities: Vec<f32>,
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

        let u = vec![0.0; (nx + 1) * ny];
        let v = vec![0.0; nx * (ny + 1)];
        let weight_u = vec![0.0; (nx + 1) * ny];
        let weight_v = vec![0.0; nx * (ny + 1)];
        let u_star = vec![0.0; (nx + 1) * ny];
        let v_star = vec![0.0; nx * (ny + 1)];
        let pressure = vec![0.0; nx * ny];
        let mut solid = vec![false; nx * ny];
        let cell_type = vec![CellType::Air; nx * ny];
        let densities = vec![0.0; nx * ny];

        for i in 0..nx {
            solid[i] = true;
            solid[i + nx * (ny - 1)] = true;
        }

        for j in 0..ny {
            solid[nx * j] = true;
            solid[(nx - 1) + nx * j] = true;
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
    pub fn idx(&self, i: usize, j: usize) -> usize {
        i + self.nx * j
    }

    #[inline]
    pub fn u_idx(&self, i: usize, j: usize) -> usize {
        i + (self.nx + 1) * j
    }

    #[inline]
    pub fn v_idx(&self, i: usize, j: usize) -> usize {
        self.idx(i, j)
    }

    #[inline]
    pub fn get_bilerp_u(&self, pos: Vec2) -> (f32, f32) {
        get_bilerp_x(
            &self.u,
            &self.u_star,
            pos,
            self.spacing,
            self.inv_spacing,
            self.grid_size,
            self.size,
            self.nx + 1,
            self.ny,
        )
    }

    #[inline]
    pub fn get_bilerp_v(&self, pos: Vec2) -> (f32, f32) {
        get_bilerp_y(
            &self.v,
            &self.v_star,
            pos,
            self.spacing,
            self.inv_spacing,
            self.grid_size,
            self.size,
            self.nx,
            self.ny + 1,
        )
    }

    #[inline]
    pub fn assign_uv_star(&mut self) {
        self.u_star = self.u.clone();
        self.v_star = self.v.clone();
    }
}

#[allow(clippy::too_many_arguments)]
fn get_bilerp_x(
    g: &[f32],
    g_star: &[f32],
    p: Vec2,
    spacing: f32,
    inv_spacing: f32,
    grid_size: UVec2,
    size: Vec2,
    nx: usize,
    ny: usize,
) -> (f32, f32) {
    let pi = (p * inv_spacing).floor().as_uvec2().clamp(UVec2::ZERO, grid_size - 1);

    if p.y >= 0.0 && p.y <= size.y - spacing {
        let ix1 = pi.x as usize;
        let ix2 = ix1 + 1;
        let (iy1, iy2) = if p.y > pi.y as f32 * spacing {
            (pi.y as usize, (pi.y as usize).min(ny - 2) + 1)
        } else {
            ((pi.y as usize).max(1) - 1, pi.y as usize)
        };

        let x1 = (ix1 as f32 - 0.5) * spacing;
        let x2 = (ix2 as f32 - 0.5) * spacing;
        let y1 = iy1 as f32 * spacing;
        let y2 = iy2 as f32 * spacing;

        let i00 = ix1 + nx * iy1;
        let i01 = ix1 + nx * iy2;
        let i10 = ix2 + nx * iy1;
        let i11 = ix2 + nx * iy2;

        (
            {
                let u00 = g[i00];
                let u01 = g[i01];
                let u10 = g[i10];
                let u11 = g[i11];

                ((u00 * (x2 - p.x) * (y2 - p.y))
                    + u10 * (p.x - x1) * (y2 - p.y)
                    + u01 * (x2 - p.x) * (p.y - y1)
                    + u11 * (p.x - x1) * (p.y - y1)) / ((x2 - x1) * (y2 - y1))

            },
            {
                let u00 = g_star[i00];
                let u01 = g_star[i01];
                let u10 = g_star[i10];
                let u11 = g_star[i11];

                ((u00 * (x2 - p.x) * (y2 - p.y))
                    + u10 * (p.x - x1) * (y2 - p.y)
                    + u01 * (x2 - p.x) * (p.y - y1)
                    + u11 * (p.x - x1) * (p.y - y1)) / ((x2 - x1) * (y2 - y1))
            }
        )
    } else if p.y < 0.0 {
        let ix1 = pi.x as usize;
        let ix2 = ix1 + 1;
        let x1 = (ix1 as f32 - 0.5) * spacing;
        let x2 = (ix2 as f32 - 0.5) * spacing;

        (
            {
                let u00 = g[ix1];
                let u10 = g[ix2];
                u00 * (1.0 - (p.x - x1) / (x2 - x1)) + u10 * ((p.x - x1) / (x2 - x1))
            },
            {
                let u00 = g_star[ix1];
                let u10 = g_star[ix2];
                u00 * (1.0 - (p.x - x1) / (x2 - x1)) + u10 * ((p.x - x1) / (x2 - x1))
            }
        )
    } else {
        let ix1 = pi.x as usize;
        let ix2 = ix1 + 1;
        let x1 = (ix1 as f32 - 0.5) * spacing;
        let x2 = (ix2 as f32 - 0.5) * spacing;

        let i00 = ix1 + nx * (ny - 1);
        let i10 = ix2 + nx * (ny - 1);

        (
            {
                let u00 = g[i00];
                let u10 = g[i10];
                u00 * (1.0 - (p.x - x1) / (x2 - x1)) + u10 * ((p.x - x1) / (x2 - x1))
            },
            {
                let u00 = g_star[i00];
                let u10 = g_star[i10];
                u00 * (1.0 - (p.x - x1) / (x2 - x1)) + u10 * ((p.x - x1) / (x2 - x1))
            }
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn get_bilerp_y(
    g: &[f32],
    g_star: &[f32],
    p: Vec2,
    spacing: f32,
    inv_spacing: f32,
    grid_size: UVec2,
    size: Vec2,
    nx: usize,
    ny: usize,
) -> (f32, f32) {
    let pi = (p * inv_spacing).floor().as_uvec2().clamp(UVec2::ZERO, grid_size - 1);

    if p.x >= 0.0 && p.x <= size.x - spacing {
        let iy1 = pi.y as usize;
        let iy2 = iy1 + 1;
        let (ix1, ix2) = if p.x > pi.x as f32 * spacing {
            (pi.x as usize, (pi.x as usize).min(nx - 2) + 1)
        } else {
            ((pi.x as usize).max(1) - 1, pi.x as usize)
        };

        let y1 = (iy1 as f32 - 0.5) * spacing;
        let y2 = (iy2 as f32 - 0.5) * spacing;
        let x1 = ix1 as f32 * spacing;
        let x2 = ix2 as f32 * spacing;

        let i00 = ix1 + nx * iy1;
        let i01 = ix1 + nx * iy2;
        let i10 = ix2 + nx * iy1;
        let i11 = ix2 + nx * iy2;

        (
            {
                let v00 = g[i00];
                let v01 = g[i01];
                let v10 = g[i10];
                let v11 = g[i11];

                ((v00 * (x2 - p.x) * (y2 - p.y))
                    + v10 * (p.x - x1) * (y2 - p.y)
                    + v01 * (x2 - p.x) * (p.y - y1)
                    + v11 * (p.x - x1) * (p.y - y1)) / ((x2 - x1) * (y2 - y1))
            },
            {
                let v00 = g_star[i00];
                let v01 = g_star[i01];
                let v10 = g_star[i10];
                let v11 = g_star[i11];

                ((v00 * (x2 - p.x) * (y2 - p.y))
                    + v10 * (p.x - x1) * (y2 - p.y)
                    + v01 * (x2 - p.x) * (p.y - y1)
                    + v11 * (p.x - x1) * (p.y - y1)) / ((x2 - x1) * (y2 - y1))
            }
        )
    } else if p.x < 0.0 {
        let iy1 = pi.y as usize;
        let iy2 = iy1 + 1;
        let y1 = (iy1 as f32 - 0.5) * spacing;
        let y2 = (iy2 as f32 - 0.5) * spacing;

        let i00 = nx * iy1;
        let i10 = nx * iy2;

        (
            {
                let v00 = g[i00];
                let v10 = g[i10];
                v00 * (1.0 - (p.y - y1) / (y2 - y1)) + v10 * ((p.y - y1) / (y2 - y1))
            },
            {
                let v00 = g_star[i00];
                let v10 = g_star[i10];
                v00 * (1.0 - (p.y - y1) / (y2 - y1)) + v10 * ((p.y - y1) / (y2 - y1))
            }
        )
    } else {
        let iy1 = pi.y as usize;
        let iy2 = iy1 + 1;
        let y1 = (iy1 as f32 - 0.5) * spacing;
        let y2 = (iy2 as f32 - 0.5) * spacing;

        let i00 = (nx - 1) + nx * iy1;
        let i10 = (nx - 1) + nx * iy2;

        (
            {
                let v00 = g[i00];
                let v10 = g[i10];
                v00 * (1.0 - (p.y - y1) / (y2 - y1)) + v10 * ((p.y - y1) / (y2 - y1))
            },
            {
                let v00 = g_star[i00];
                let v10 = g_star[i10];
                v00 * (1.0 - (p.y - y1) / (y2 - y1)) + v10 * ((p.y - y1) / (y2 - y1))
            }
        )
    }
}
