use glam::{UVec3, Vec2, Vec3};

use super::CellType;

#[derive(Debug, Clone)]
pub struct MacGrid3D {
    /// Size of the grid, in meters.
    pub size: Vec3,
    /// Size of the grid, in cells.
    pub grid_size: UVec3,
    /// Number of cells in the X direction.
    pub nx: usize,
    /// Number of cells in the Y direction.
    pub ny: usize,
    /// Number of cells in the Z direction.
    pub nz: usize,
    /// Cell size.
    pub spacing: f32,
    /// 1.0 / spacing
    pub inv_spacing: f32,

    /// Grid velocities in the X direction.
    pub u: Vec<f32>,
    /// Grid velocities in the Y direction.
    pub v: Vec<f32>,
    /// Grid velocities in the Z direction.
    pub w: Vec<f32>,
    /// Grid velocity weights in the X direction.
    pub weight_u: Vec<f32>,
    /// Grid velocity weights in the Y direction.
    pub weight_v: Vec<f32>,
    /// Grid velocity weights in the Z direction.
    pub weight_w: Vec<f32>,
    /// Intermediate grid velocities in the X direction.
    pub u_star: Vec<f32>,
    /// Intermediate grid velocities in the Y direction.
    pub v_star: Vec<f32>,
    /// Intermediate grid velocities in the Z direction.
    pub w_star: Vec<f32>,
    /// Pressure of the grid.
    pub pressure: Vec<f32>,
    /// Solid grid cells.
    pub solid: Vec<bool>,
    /// Grid cell types (`Fluid`, `Solid`, or `Air`).
    pub cell_type: Vec<CellType>,
    /// Grid densities.
    pub densities: Vec<f32>,
}

impl MacGrid3D {
    pub fn new(
        size: Vec3,
        spacing: f32,
    ) -> Self {
        let grid_size = (size / spacing).floor().as_uvec3() + 1;
        let h = f32::max(
            f32::max(size.x / grid_size.x as f32, size.y / grid_size.y as f32),
            size.z / grid_size.z as f32,
        );
        let nx = grid_size.x as usize;
        let ny = grid_size.y as usize;
        let nz = grid_size.z as usize;

        let u = vec![0.0; (nx + 1) * ny * nz];
        let v = vec![0.0; nx * (ny + 1) * nz];
        let w = vec![0.0; nx * ny * (nz + 1)];
        let weight_u = vec![0.0; (nx + 1) * ny * nz];
        let weight_v = vec![0.0; nx * (ny + 1) * nz];
        let weight_w = vec![0.0; nx * ny * (nz + 1)];
        let u_star = vec![0.0; (nx + 1) * ny * nz];
        let v_star = vec![0.0; nx * (ny + 1) * nz];
        let w_star = vec![0.0; nx * ny * (nz + 1)];
        let pressure = vec![0.0; nx * ny * nz];
        let mut solid = vec![false; nx * ny * nz];
        let cell_type = vec![CellType::Air; nx * ny * nz];
        let densities = vec![0.0; nx * ny * nz];

        for i in 0..nx {
            for j in 0..ny {
                solid[i + nx * j] = true;
                solid[i + nx * (j + ny * (nz - 1))] = true;
            }
        }

        for i in 0..nx {
            for k in 0..nz {
                solid[i + nx * ny * k] = true;
                solid[i + nx * ((ny - 1) + ny * k)] = true;
            }
        }

        for j in 0..ny {
            for k in 0..nz {
                solid[nx * (j + ny * k)] = true;
                solid[(nx - 1) + nx * (j + ny * k)] = true;
            }
        }

        Self {
            size,
            grid_size,
            nx,
            ny,
            nz,
            spacing: h,
            inv_spacing: h.recip(),
            u,
            v,
            w,
            weight_u,
            weight_v,
            weight_w,
            u_star,
            v_star,
            w_star,
            pressure,
            solid,
            cell_type,
            densities,
        }
    }

    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    #[inline]
    pub fn u_idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + (self.nx + 1) * (j + self.ny * k)
    }

    #[inline]
    pub fn v_idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + (self.ny + 1) * k)
    }

    #[inline]
    pub fn w_idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    #[inline]
    pub fn get_bilerp_u(&self, pos: Vec3) -> (f32, f32) {
        grid_interpolate(
            &self.u,
            &self.u_star,
            pos,
            self.spacing,
            self.inv_spacing,
            self.grid_size,
            self.nx + 1,
            self.ny,
            self.nz,
            Vec3::new(-0.5 * self.spacing, 0.0, 0.0),
            self.size,
        )
    }

    #[inline]
    pub fn get_bilerp_v(&self, pos: Vec3) -> (f32, f32) {
        grid_interpolate(
            &self.v,
            &self.v_star,
            pos,
            self.spacing,
            self.inv_spacing,
            self.grid_size,
            self.nx,
            self.ny + 1,
            self.nz,
            Vec3::new(0.0, -0.5 * self.spacing, 0.0),
            self.size
        )
    }

    #[inline]
    pub fn get_bilerp_w(&self, pos: Vec3) -> (f32, f32) {
        grid_interpolate(
            &self.w,
            &self.w_star,
            pos,
            self.spacing,
            self.inv_spacing,
            self.grid_size,
            self.nx,
            self.ny,
            self.nz + 1,
            Vec3::new(0.0, 0.0, -0.5 * self.spacing),
            self.size
        )
    }

    #[inline]
    pub fn assign_uvw_star(&mut self) {
        self.u_star = self.u.clone();
        self.v_star = self.v.clone();
        self.w_star = self.w.clone();
    }
}

#[allow(clippy::too_many_arguments)]
fn grid_interpolate(
    g: &[f32],
    g_star: &[f32],
    p: Vec3,
    spacing: f32,
    inv_spacing: f32,
    grid_size: UVec3,
    nx: usize,
    ny: usize,
    nz: usize,
    offset: Vec3,
    size: Vec3,
) -> (f32, f32) {
    let pi = (p * inv_spacing).floor().as_uvec3().clamp(UVec3::ZERO, grid_size - 1);
    let px = pi.x as usize;
    let py = pi.y as usize;
    let pz = pi.z as usize;

    let min = offset + spacing * pi.as_vec3();

    let inside_left = p.x >= 0.0;
    let inside_right = p.x <= size.x - spacing;
    let inside_bottom = p.y >= 0.0;
    let inside_top = p.y <= size.y - spacing;
    let inside_front = p.z >= 0.0;
    let inside_back = p.z <= size.z - spacing;

    if inside_left && inside_right && inside_bottom && inside_top && inside_front && inside_back {
        let i000 = px + nx * (py + ny * pz);
        let i001 = px + nx * (py + ny * (pz + 1));
        let i010 = px + nx * ((py + 1) + ny * pz);
        let i011 = px + nx * ((py + 1) + ny * (pz + 1));
        let i100 = i000 + 1;
        let i101 = i001 + 1;
        let i110 = i010 + 1;
        let i111 = i011 + 1;

        let delta = (p - min) * inv_spacing;
        (
            trilerp_norm(g[i000], g[i001], g[i010], g[i011], g[i100], g[i101], g[i110], g[i111], delta),
            trilerp_norm(g_star[i000], g_star[i001], g_star[i010], g_star[i011], g_star[i100], g_star[i101], g_star[i110], g_star[i111], delta),
        )
    } else if !inside_left {
        if !inside_top {
            let i0 = nx * ((ny - 1) + ny * pz);
            let i1 = i0 + nx * ny;
            do_lerp(i0, i1, min.z, inv_spacing, p.z, g, g_star)
        } else if !inside_bottom {
            let i0 = nx * ny * pz;
            let i1 = i0 + nx * ny;
            do_lerp(i0, i1, min.z, inv_spacing, p.z, g, g_star)
        } else if !inside_front {
            let i0 = nx * py;
            let i1 = i0 + nx;
            do_lerp(i0, i1, min.y, inv_spacing, p.y, g, g_star)
        } else if !inside_back {
            let i0 = nx * (py + ny * (nz - 1));
            let i1 = i0 + nx;
            do_lerp(i0, i1, min.y, inv_spacing, p.y, g, g_star)
        } else {
            let i00 = nx * (py + ny * pz);
            let i01 = i00 + nx * ny;
            let i10 = i00 + nx;
            let i11 = i00 + nx + nx * ny;
            do_bilerp(i00, i01, i10, i11, min.y, min.z, inv_spacing, p.y, p.z, g, g_star)
        }
    } else if !inside_right {
        if !inside_top {
            let i0 = (nx - 1) + nx * ((ny - 1) + ny * pz);
            let i1 = i0 + nx * ny;
            do_lerp(i0, i1, min.z, inv_spacing, p.z, g, g_star)
        } else if !inside_bottom {
            let i0 = (nx - 1) + nx * ny * pz;
            let i1 = i0 + nx * ny;
            do_lerp(i0, i1, min.z, inv_spacing, p.z, g, g_star)
        } else if !inside_front {
            let i0 = (nx - 1) + nx * py;
            let i1 = i0 + nx;
            do_lerp(i0, i1, min.y, inv_spacing, p.y, g, g_star)
        } else if !inside_back {
            let i0 = (nx - 1) + nx * (py + ny * (nz - 1));
            let i1 = i0 + nx;
            do_lerp(i0, i1, min.y, inv_spacing, p.y, g, g_star)
        } else {
            let i00 = (nx - 1) + nx * (py + ny * pz);
            let i01 = i00 + nx * ny;
            let i10 = i00 + nx;
            let i11 = i00 + nx + nx * ny;
            do_bilerp(i00, i01, i10, i11, min.y, min.z, inv_spacing, p.y, p.z, g, g_star)
        }
    } else if !inside_top {
        if !inside_front {
            let i0 = px + nx * (ny - 1);
            let i1 = i0 + 1;
            do_lerp(i0, i1, min.x, inv_spacing, p.x, g, g_star)
        } else if !inside_back {
            let i0 = px + nx * ((ny - 1) + ny * (nz - 1));
            let i1 = i0 + 1;
            do_lerp(i0, i1, min.x, inv_spacing, p.x, g, g_star)
        } else {
            let i00 = px + nx * ((ny - 1) + ny * pz);
            let i01 = i00 + 1;
            let i10 = i00 + nx * ny;
            let i11 = i10 + 1;
            do_bilerp(i00, i01, i10, i11, min.z, min.x, inv_spacing, p.z, p.x, g, g_star)
        }
    } else if !inside_bottom {
        if !inside_front {
            let i0 = px;
            let i1 = i0 + 1;
            do_lerp(i0, i1, min.x, inv_spacing, p.x, g, g_star)
        } else if !inside_back {
            let i0 = px + nx * ny * (nz - 1);
            let i1 = i0 + 1;
            do_lerp(i0, i1, min.x, inv_spacing, p.x, g, g_star)
        } else {
            let i00 = px + nx * ny * pz;
            let i01 = i00 + 1;
            let i10 = i00 + nx * ny;
            let i11 = i10 + 1;
            do_bilerp(i00, i01, i10, i11, min.z, min.x, inv_spacing, p.z, p.x, g, g_star)
        }
    } else if !inside_front {
        let i00 = px + nx * py;
        let i01 = i00 + nx;
        let i10 = i00 + 1;
        let i11 = i10 + nx;
        do_bilerp(i00, i01, i10, i11, min.x, min.y, inv_spacing, p.x, p.y, g, g_star)
    } else {
        let i00 = px + nx * (py + ny * (nz - 1));
        let i01 = i00 + nx;
        let i10 = i00 + 1;
        let i11 = i10 + nx;
        do_bilerp(i00, i01, i10, i11, min.x, min.y, inv_spacing, p.x, p.y, g, g_star)
    }
}

#[inline]
fn lerp_norm(v0: f32, v1: f32, p: f32) -> f32 {
    v0 + p * (v1 - v0)
}

#[inline]
fn bilerp_norm(v00: f32, v01: f32, v10: f32, v11: f32, p: Vec2) -> f32 {
    let v0 = v00 + p.x * (v10 - v00);
    let v1 = v11 + p.x * (v11 - v01);
    v0 + p.y * (v1 - v0)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn trilerp_norm(v000: f32, v001: f32, v010: f32, v011: f32, v100: f32, v101: f32, v110: f32, v111: f32, p: Vec3) -> f32 {
    let v00 = v000 + p.x * (v100 - v000);
    let v01 = v001 + p.x * (v101 - v001);
    let v10 = v010 + p.x * (v110 - v010);
    let v11 = v011 + p.x * (v111 - v011);

    let v0 = v00 + p.y * (v10 - v00);
    let v1 = v01 + p.y * (v11 - v01);

    v0 + p.z * (v1 - v0)
}

fn do_lerp(
    i0: usize,
    i1: usize,
    min: f32,
    inv_spacing: f32,
    p: f32,
    g: &[f32],
    g_star: &[f32],
) -> (f32, f32) {
    let alpha = (p - min) * inv_spacing;
    (
        lerp_norm(g[i0], g[i1], alpha),
        lerp_norm(g_star[i0], g[i1], alpha),
    )
}

#[allow(clippy::too_many_arguments)]
fn do_bilerp(
    i00: usize,
    i01: usize,
    i10: usize,
    i11: usize,
    min_x: f32,
    min_y: f32,
    inv_spacing: f32,
    p_x: f32,
    p_y: f32,
    g: &[f32],
    g_star: &[f32],
) -> (f32, f32) {
    let alpha = (p_x - min_x) * inv_spacing;
    let beta = (p_y - min_y) * inv_spacing;
    (
        bilerp_norm(g[i00], g[i01], g[i10], g[i11], Vec2::new(alpha, beta)),
        bilerp_norm(g_star[i00], g_star[i01], g_star[i10], g_star[i11], Vec2::new(alpha, beta)),
    )
}
