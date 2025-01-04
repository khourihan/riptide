use glam::{UVec2, Vec2};
use ndarray::{azip, Array0, Array1, Array2, Axis};

use crate::{fluid::{obstacle::{Obstacle, ObstacleSet}, Fluid}, io::encode::FluidDataEncoder};

use super::CellType;

#[derive(Debug, Clone)]
pub struct FlipFluid2D {
    /// The density of the fluid, in kg/m³.
    ///
    /// Air in `0` kg/m³ and water is `1000` kg/m³.
    density: f32,
    size: UVec2,
    /// Cell size.
    spacing: f32,

    rest_density: f32,
    particle_radius: f32,
    particle_spacing: f32,
    n_particles: usize,
    particle_resolution: UVec2,

    /// Grid velocities.
    uvs: Array2<Vec2>,
    /// Grid velocity deltas.
    dudvs: Array2<Vec2>,
    /// Previous grid velocities.
    prev_uvs: Array2<Vec2>,
    /// Pressure of the grid.
    pressure: Array2<f32>,
    /// Solid grid cells. `0.0` for completely solid and `1.0` for not solid.
    solid: Array2<f32>,
    /// Grid cell types (`Fluid`, `Solid` or `Air`).
    cell_type: Array2<CellType>,
    /// Grid densities.
    densities: Array2<f32>,

    /// Particle positions.
    positions: Array1<Vec2>,
    /// Particle velocities.
    velocities: Array1<Vec2>,
    /// Fluid roughness approximation per particle.
    roughness: Array1<f32>,
    cell_particle_indices: Array1<usize>,
}

impl FlipFluid2D {
    pub fn new(
        density: f32,
        width: u32,
        height: u32,
        spacing: f32,
        particle_radius: f32,
    ) -> Self {
        let size = UVec2::new((width as f32 / spacing).floor() as u32 + 1, (height as f32 / spacing).floor() as u32 + 1);
        let h = f32::max(width as f32 / size.x as f32, height as f32 / size.y as f32);

        let uvs = Array2::from_elem((size.x as usize, size.y as usize), Vec2::ZERO);
        let dudvs = Array2::from_elem((size.x as usize, size.y as usize), Vec2::ZERO);
        let prev_uvs = Array2::from_elem((size.x as usize, size.y as usize), Vec2::ZERO);
        let pressure = Array2::from_elem((size.x as usize, size.y as usize), 0.0);
        let solid = Array2::from_elem((size.x as usize, size.y as usize), 1.0);
        let cell_type = Array2::from_elem((size.x as usize, size.y as usize), CellType::Fluid);
        let densities = Array2::from_elem((size.x as usize, size.y as usize), 0.0);

        let positions = Array1::from_vec(vec![]);
        let velocities = Array1::from_vec(vec![]);
        let roughness = Array1::from_vec(vec![]);

        let particle_spacing = 2.2 * particle_radius;
        let particle_resolution = UVec2::new(
            (width as f32 / particle_spacing).floor() as u32 + 1,
            (height as f32 / particle_spacing).floor() as u32 + 1,
        );

        let cell_particle_indices = Array1::from_vec(vec![]);

        Self {
            density,
            size,
            spacing: h,
            rest_density: 0.0,
            particle_radius,
            particle_spacing,
            n_particles: 0,
            particle_resolution,
            uvs,
            dudvs,
            prev_uvs,
            pressure,
            solid,
            cell_type,
            densities,
            positions,
            velocities,
            roughness,
            cell_particle_indices,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32, spacing: f32) {
        let size = UVec2::new((width as f32 / spacing).floor() as u32 + 1, (height as f32 / spacing).floor() as u32 + 1);
        self.spacing = f32::max(width as f32 / size.x as f32, height as f32 / size.y as f32);

        match size.x.cmp(&self.size.x) {
            std::cmp::Ordering::Greater => {
                let _ = self.uvs.append(Axis(0), Array2::from_elem(((size.x - self.size.x) as usize, self.size.y as usize), Vec2::ZERO).view());
                let _ = self.dudvs.append(Axis(0), Array2::from_elem(((size.x - self.size.x) as usize, self.size.y as usize), Vec2::ZERO).view());
                let _ = self.prev_uvs.append(Axis(0), Array2::from_elem(((size.x - self.size.x) as usize, self.size.y as usize), Vec2::ZERO).view());
                let _ = self.pressure.append(Axis(0), Array2::from_elem(((size.x - self.size.x) as usize, self.size.y as usize), 0.0).view());
                let _ = self.solid.append(Axis(0), Array2::from_elem(((size.x - self.size.x) as usize, self.size.y as usize), 1.0).view());
                let _ = self.cell_type.append(Axis(0), Array2::from_elem(((size.x - self.size.x) as usize, self.size.y as usize), CellType::Fluid).view());
                let _ = self.densities.append(Axis(0), Array2::from_elem(((size.x - self.size.x) as usize, self.size.y as usize), 0.0).view());
            },
            std::cmp::Ordering::Less => {
                for _ in 0..(self.size.x - size.x) {
                    let i = self.uvs.len_of(Axis(0)) - 1;
                    self.uvs.remove_index(Axis(0), i);
                    self.dudvs.remove_index(Axis(0), i);
                    self.prev_uvs.remove_index(Axis(0), i);
                    self.pressure.remove_index(Axis(0), i);
                    self.solid.remove_index(Axis(0), i);
                    self.cell_type.remove_index(Axis(0), i);
                    self.densities.remove_index(Axis(0), i);
                }
            },
            std::cmp::Ordering::Equal => (),
        }

        match size.y.cmp(&self.size.y) {
            std::cmp::Ordering::Greater => {
                let _ = self.uvs.append(Axis(1), Array2::from_elem((size.x as usize, (size.y - self.size.y) as usize), Vec2::ZERO).view());
                let _ = self.dudvs.append(Axis(1), Array2::from_elem((size.x as usize, (size.y - self.size.y) as usize), Vec2::ZERO).view());
                let _ = self.prev_uvs.append(Axis(1), Array2::from_elem((size.x as usize, (size.y - self.size.y) as usize), Vec2::ZERO).view());
                let _ = self.pressure.append(Axis(1), Array2::from_elem((size.x as usize, (size.y - self.size.y) as usize), 0.0).view());
                let _ = self.solid.append(Axis(1), Array2::from_elem((size.x as usize, (size.y - self.size.y) as usize), 1.0).view());
                let _ = self.cell_type.append(Axis(1), Array2::from_elem((size.x as usize, (size.y - self.size.y) as usize), CellType::Fluid).view());
                let _ = self.densities.append(Axis(1), Array2::from_elem((size.x as usize, (size.y - self.size.y) as usize), 0.0).view());
            },
            std::cmp::Ordering::Less => {
                for _ in 0..(self.size.y - size.y) {
                    let i = self.uvs.len_of(Axis(1)) - 1;
                    self.uvs.remove_index(Axis(1), i);
                    self.dudvs.remove_index(Axis(1), i);
                    self.prev_uvs.remove_index(Axis(1), i);
                    self.pressure.remove_index(Axis(1), i);
                    self.solid.remove_index(Axis(1), i);
                    self.cell_type.remove_index(Axis(1), i);
                    self.densities.remove_index(Axis(1), i);
                }
            },
            std::cmp::Ordering::Equal => (),
        }

        self.size = size;
        self.particle_resolution = UVec2::new(
            (width as f32 / self.particle_spacing).floor() as u32 + 1,
            (height as f32 / self.particle_spacing).floor() as u32 + 1,
        );
    }

    pub fn insert_particle(&mut self, pos: Vec2) {
        let _ = self.positions.push(Axis(0), Array0::from_elem((), pos).view());
        let _ = self.velocities.push(Axis(0), Array0::from_elem((), Vec2::ZERO).view());
        let _ = self.roughness.push(Axis(0), Array0::from_elem((), 0.0).view());
        let _ = self.cell_particle_indices.push(Axis(0), Array0::from_elem((), 0).view());
        self.n_particles += 1;
    }

    pub fn set_solid(&mut self, i: usize, j: usize, v: f32) {
        self.solid[(i, j)] = v;
    }

    pub fn iter_positions(&self) -> impl Iterator<Item = &Vec2> {
        self.positions.iter()
    }

    pub fn iter_particles(&self) -> impl Iterator<Item = (&Vec2, &Vec2, &f32)> {
        self.positions.iter().zip(self.velocities.iter()).zip(self.roughness.iter()).map(|((p, v), r)| (p, v, r))
    }

    pub fn size(&self) -> UVec2 {
        self.size
    }

    pub fn bounds(&self) -> (Vec2, Vec2) {
        (
            Vec2::splat(self.spacing + self.particle_radius),
            (self.size - 1).as_vec2() * self.spacing - self.particle_radius,
        )
    }

    fn integrate_particles(&mut self, dt: f32, gravity: Vec2) {
        self.velocities.map_inplace(|v| *v += dt * gravity);

        azip!((p in &mut self.positions, vel in &self.velocities) {
            *p += vel * dt;
        });
    }

    fn push_particles_apart(&mut self, num_iters: usize) {
        const ROUGHNESS_DIFFUSION: f32 = 0.001;

        let cell_count = (self.particle_resolution.x * self.particle_resolution.y) as usize;
        let mut cell_particle_count = Array1::from_elem(cell_count, 0);
        let mut first_cell_particle = Array1::from_elem(cell_count + 1, 0);

        for p in self.positions.iter() {
            let pi = (p / self.particle_spacing).floor().as_uvec2()
                .clamp(UVec2::ZERO, self.particle_resolution - 1);
            let cell_nr = pi.x * self.particle_resolution.y + pi.y;
            cell_particle_count[cell_nr as usize] += 1;
        }

        let mut first = 0;

        for (count, first_cell) in cell_particle_count.iter().zip(first_cell_particle.iter_mut()) {
            first += count;
            *first_cell = first;
        }

        first_cell_particle[(self.particle_resolution.x * self.particle_resolution.y) as usize] = first;

        for (i, p) in self.positions.iter().enumerate() {
            let pi = (p / self.particle_spacing).floor().as_uvec2()
                .clamp(UVec2::ZERO, self.particle_resolution - 1);
            let cell_nr = (pi.x * self.particle_resolution.y + pi.y) as usize;
            first_cell_particle[cell_nr] -= 1;
            self.cell_particle_indices[first_cell_particle[cell_nr]] = i;
        }

        let min_dist = 2.0 * self.particle_radius;
        let min_dist2 = min_dist * min_dist;

        for _iter in 0..num_iters {
            for i in 0..self.n_particles {
                let p = self.positions[i];

                let pi = (p / self.particle_spacing).floor().as_uvec2();
                let p0 = pi.max(UVec2::ONE) - 1;
                let p1 = (pi + 1).min(self.particle_resolution - 1);

                for xi in p0.x..=p1.x {
                    for yi in p0.y..=p1.y {
                        let cell_nr = (xi * self.particle_resolution.y + yi) as usize;
                        let first = first_cell_particle[cell_nr];
                        let last = first_cell_particle[cell_nr + 1];

                        for j in first..last {
                            let id = self.cell_particle_indices[j];
                            if id == i {
                                continue;
                            }

                            let q = self.positions[id];
                            let mut delta = q - p;
                            let d2 = delta.length_squared();
                            if d2 > min_dist2 || d2 == 0.0 {
                                continue;
                            }
                            
                            let d = d2.sqrt();
                            let s = 0.5 * (min_dist - d) / d;
                            delta *= s;

                            self.positions[i] -= delta;
                            self.positions[id] += delta;

                            let r0 = self.roughness[i];
                            let r1 = self.roughness[id];
                            let rough = 0.5 * (r0 + r1);
                            self.roughness[i] = r0 + (rough - r0) * ROUGHNESS_DIFFUSION;
                            self.roughness[id] = r1 + (rough - r1) * ROUGHNESS_DIFFUSION;
                        }
                    }
                }
            }
        }
    }

    fn handle_particle_collisions(&mut self, obstacles: &ObstacleSet<2>, dt: f32) {
        let (min, max) = self.bounds();

        azip!((p in &mut self.positions, v in &mut self.velocities) {
            let sdf = obstacles.sdf((*p).into());
            if sdf.distance < 0.0 {
                // TODO: add velocity of obstacle to this.
                *v = -sdf.distance * Vec2::from(sdf.gradient) / dt;
            }

            if p.x < min.x {
                p.x = min.x;
                v.x = 0.0;
            }

            if p.x > max.x {
                p.x = max.x;
                v.x = 0.0;
            }

            if p.y < min.y {
                p.y = min.y;
                v.y = 0.0;
            }

            if p.y > max.y {
                p.y = max.y;
                v.y = 0.0;
            }
        });
    }

    fn update_particle_density(&mut self) {
        let h = self.spacing;
        let h1 = h.recip();
        let h2 = 0.5 * h;

        self.densities.fill(0.0);

        for p in self.positions.iter() {
            let pi = p.clamp(Vec2::splat(h), (self.size - 1).as_vec2() * h);

            let p0 = ((pi - h2) * h1).floor().as_uvec2();
            let t = ((pi - h2) - p0.as_vec2() * h) * h1;
            let p1 = (p0 + 1).min(self.size - 2);
            let s = 1.0 - t;

            if p0.x < self.size.x && p0.y < self.size.y {
                self.densities[(p0.x as usize, p0.y as usize)] += s.x * s.y;
            }

            if p1.x < self.size.x && p0.y < self.size.y {
                self.densities[(p1.x as usize, p0.y as usize)] += t.x * s.y;
            }

            if p1.x < self.size.x && p1.y < self.size.y {
                self.densities[(p1.x as usize, p1.y as usize)] += t.x * t.y;
            }

            if p0.x < self.size.x && p1.y < self.size.y {
                self.densities[(p0.x as usize, p1.y as usize)] += s.x * t.y;
            }
        }

        if self.rest_density == 0.0 {
            let mut sum: f32 = 0.0;
            let mut num_fluid_cells: usize = 0;

            azip!((&cell_type in &self.cell_type, &density in &self.densities) {
                if cell_type == CellType::Fluid {
                    sum += density;
                    num_fluid_cells += 1;
                }
            });

            if num_fluid_cells > 0 {
                self.rest_density = sum / num_fluid_cells as f32;
            }
        }
    }

    fn transfer_velocities_to_grid(&mut self) {
        let h = self.spacing;
        let h1 = h.recip();
        let h2 = 0.5 * h;

        self.prev_uvs.assign(&self.uvs);
        self.dudvs.fill(Vec2::ZERO);
        self.uvs.fill(Vec2::ZERO);

        azip!((cell_type in &mut self.cell_type, &s in &self.solid) {
            *cell_type = if s == 0.0 { CellType::Solid } else { CellType::Air };
        });

        for p in self.positions.iter() {
            let pi = (p * h1).floor().as_uvec2().clamp(UVec2::ZERO, self.size - 1);

            if self.cell_type[(pi.x as usize, pi.y as usize)] == CellType::Air {
                self.cell_type[(pi.x as usize, pi.y as usize)] = CellType::Fluid;
            }
        }

        for dim in 0..2 {
            let delta = Vec2::new(
                if dim == 0 { 0.0 } else { h2 },
                if dim == 1 { 0.0 } else { h2 },
            );

            for i in 0..self.n_particles {
                let p = self.positions[i];
                let pi = p.clamp(Vec2::splat(h), (self.size - 1).as_vec2() * h);

                let p0 = ((pi - delta) * h1).floor().as_uvec2().min(self.size - 2);
                let t = ((pi - delta) - p0.as_vec2() * h) * h1;
                let p1 = (p0 + 1).min(self.size - 2);
                let s = 1.0 - t;

                let i0 = (p0.x as usize, p0.y as usize);
                let i1 = (p1.x as usize, p0.y as usize);
                let i2 = (p1.x as usize, p1.y as usize);
                let i3 = (p0.x as usize, p1.y as usize);

                let d0 = s.x * s.y;
                let d1 = t.x * s.y;
                let d2 = t.x * t.y;
                let d3 = s.x * t.y;

                let v = self.velocities[i][dim];
                self.uvs[i0][dim] += v * d0;
                self.uvs[i1][dim] += v * d1;
                self.uvs[i2][dim] += v * d2;
                self.uvs[i3][dim] += v * d3;
                self.dudvs[i0][dim] += d0;
                self.dudvs[i1][dim] += d1;
                self.dudvs[i2][dim] += d2;
                self.dudvs[i3][dim] += d3;
            }

            azip!((uv in &mut self.uvs, &dudv in &self.dudvs) {
                if dudv[dim] > 0.0 {
                    uv[dim] /= dudv[dim];
                }
            });

            for i in 0..self.size.x as usize {
                for j in 0..self.size.y as usize {
                    let solid = self.cell_type[(i, j)] == CellType::Solid;

                    if solid || (i > 0 && self.cell_type[(i - 1, j)] == CellType::Solid) {
                        self.uvs[(i, j)].x = self.prev_uvs[(i, j)].x;
                    }
                    if solid || (j > 0 && self.cell_type[(i, j - 1)] == CellType::Solid) {
                        self.uvs[(i, j)].y = self.prev_uvs[(i, j)].y;
                    }
                }
            }
        }
    }

    fn transfer_velocities_to_particles(&mut self, flip_ratio: f32) {
        let h = self.spacing;
        let h1 = h.recip();
        let h2 = 0.5 * h;

        for dim in 0..2 {
            let delta = Vec2::new(
                if dim == 0 { 0.0 } else { h2 },
                if dim == 1 { 0.0 } else { h2 },
            );

            for i in 0..self.n_particles {
                let p = self.positions[i];
                let pi = p.clamp(Vec2::splat(h), (self.size - 1).as_vec2() * h);

                let p0 = ((pi - delta) * h1).floor().as_uvec2().min(self.size - 2);
                let t = ((pi - delta) - p0.as_vec2() * h) * h1;
                let p1 = (p0 + 1).min(self.size - 2);
                let s = 1.0 - t;

                let i0 = (p0.x as usize, p0.y as usize);
                let i1 = (p1.x as usize, p0.y as usize);
                let i2 = (p1.x as usize, p1.y as usize);
                let i3 = (p0.x as usize, p1.y as usize);

                let d0 = s.x * s.y;
                let d1 = t.x * s.y;
                let d2 = t.x * t.y;
                let d3 = s.x * t.y;

                let offset = if dim == 0 { (1, 0) } else { (0, 1) };
                let valid0 = self.cell_type[i0] != CellType::Air || self.cell_type[(i0.0 - offset.0, i0.1 - offset.1)] != CellType::Air;
                let valid1 = self.cell_type[i1] != CellType::Air || self.cell_type[(i1.0 - offset.0, i1.1 - offset.1)] != CellType::Air;
                let valid2 = self.cell_type[i2] != CellType::Air || self.cell_type[(i2.0 - offset.0, i2.1 - offset.1)] != CellType::Air;
                let valid3 = self.cell_type[i3] != CellType::Air || self.cell_type[(i3.0 - offset.0, i3.1 - offset.1)] != CellType::Air;
                let v0 = if valid0 { 1.0 } else { 0.0 };
                let v1 = if valid1 { 1.0 } else { 0.0 };
                let v2 = if valid2 { 1.0 } else { 0.0 };
                let v3 = if valid3 { 1.0 } else { 0.0 };

                let v = self.velocities[i][dim];
                let d = v0 * d0 + v1 * d1 + v2 * d2 + v3 * d3;

                if d > 0.0 {
                    let picv = (v0 * d0 * self.uvs[i0][dim] + v1 * d1 * self.uvs[i1][dim] 
                        + v2 * d2 * self.uvs[i2][dim] + v3 * d3 * self.uvs[i3][dim]) / d;
                    let corr = (v0 * d0 * (self.uvs[i0][dim] - self.prev_uvs[i0][dim]) 
                        + v1 * d1 * (self.uvs[i1][dim] - self.prev_uvs[i1][dim])
                        + v2 * d2 * (self.uvs[i2][dim] - self.prev_uvs[i2][dim])
                        + v3 * d3 * (self.uvs[i3][dim] - self.prev_uvs[i3][dim])) / d;
                    let flipv = v + corr;

                    self.velocities[i][dim] = (1.0 - flip_ratio) * picv + flip_ratio * flipv;
                }
            }
        }
    }

    fn solve_incompressibility(&mut self, num_iters: usize, dt: f32, over_relaxation: f32, compensate_drift: bool) {
        self.pressure.fill(0.0);
        self.prev_uvs.assign(&self.uvs);

        let cp = self.density * self.spacing / dt;

        for _iter in 0..num_iters {
            for i in 1..self.size.x as usize - 1 {
                for j in 1..self.size.y as usize - 1 {
                    if self.cell_type[(i, j)] != CellType::Fluid {
                        continue;
                    }

                    let center = (i, j);
                    let left = (i - 1, j);
                    let right = (i + 1, j);
                    let bottom = (i, j - 1);
                    let top = (i, j + 1);

                    let sx0 = self.solid[left];
                    let sx1 = self.solid[right];
                    let sy0 = self.solid[bottom];
                    let sy1 = self.solid[top];
                    let s = sx0 + sx1 + sy0 + sy1;

                    if s == 0.0 {
                        continue;
                    }

                    let mut div = self.uvs[right].x - self.uvs[center].x
                        + self.uvs[top].y - self.uvs[center].y;

                    if self.rest_density > 0.0 && compensate_drift {
                        let k = 1.0;
                        let compression = self.densities[(i, j)] - self.rest_density;
                        if compression > 0.0 {
                            div -= k * compression;
                        }
                    }

                    let mut p = -div / s;
                    p *= over_relaxation;
                    self.pressure[center] += cp * p;

                    self.uvs[center].x -= sx0 * p;
                    self.uvs[right].x += sx1 * p;
                    self.uvs[center].y -= sy0 * p;
                    self.uvs[top].y += sy1 * p;
                }
            }
        }
    }

    fn update_roughness(&mut self) {
        let h1 = self.spacing.recip();
        let d0 = self.rest_density;

        for i in 0..self.n_particles {
            let s = 0.01;
            let p = self.positions[i];
            let pi = (p * h1).floor().as_uvec2().clamp(UVec2::ONE, self.size - 1);

            self.roughness[i] = (self.roughness[i] - s).clamp(0.0, 1.0);

            if d0 > 0.0 {
                let rel_density = self.densities[(pi.x as usize, pi.y as usize)] / d0;
                if rel_density < 0.7 {
                    let s = 0.8;
                    self.roughness[i] = s;
                }
            }
        }
    }

    pub fn set_obstacles(&mut self, obstacles: &ObstacleSet<2>, dt: f32) {
        for i in 1..self.size.x as usize - 2 {
            for j in 1..self.size.y as usize - 2 {
                self.solid[(i, j)] = 1.0;
                let p = Vec2::new(i as f32 + 0.5, j as f32 + 0.5) * self.spacing;
                let sdf = obstacles.sdf(p.into());

                if sdf.distance < 0.0 {
                    // TODO: add velocity of obstacle to this.
                    let v = -sdf.distance * Vec2::from(sdf.gradient) / dt;
                    self.solid[(i, j)] = 0.0;
                    self.uvs[(i, j)] = v;
                    self.uvs[(i + 1, j)].x = v.x;
                    self.uvs[(i, j + 1)].y = v.y;
                }
            }
        }
    }
}

pub struct FlipFluid2DParams {
    pub num_substeps: usize,
    pub gravity: Vec2,
    pub flip_ratio: f32,
    pub num_pressure_iters: usize,
    pub num_particle_iters: usize,
    pub over_relaxation: f32,
    pub compensate_drift: bool,
    pub separate_particles: bool,
}

impl Default for FlipFluid2DParams {
    fn default() -> Self {
        Self {
            num_substeps: 2,
            gravity: Vec2::new(0.0, -9.81),
            flip_ratio: 0.9,
            num_pressure_iters: 100,
            num_particle_iters: 2,
            over_relaxation: 1.9,
            compensate_drift: true,
            separate_particles: true,
        }
    }
}

impl Fluid<2> for FlipFluid2D {
    type Params = FlipFluid2DParams;

    fn step(&mut self, dt: f32, params: &Self::Params, obstacles: &ObstacleSet<2>) {
        let sdt = dt / params.num_substeps as f32;

        self.set_obstacles(obstacles, dt);

        for _step in 0..params.num_substeps {
            self.integrate_particles(sdt, params.gravity);
            if params.separate_particles {
                self.push_particles_apart(params.num_particle_iters);
            }
            self.handle_particle_collisions(obstacles, sdt);
            self.transfer_velocities_to_grid();
            self.update_particle_density();
            self.solve_incompressibility(params.num_pressure_iters, sdt, params.over_relaxation, params.compensate_drift);
            self.transfer_velocities_to_particles(params.flip_ratio);
        }

        self.update_roughness();
    }
    
    fn encode_state<W: std::io::Write>(&self, encoder: &mut FluidDataEncoder<W>) -> Result<(), crate::io::encode::EncodingError> {
        encoder.encode_section(self.positions.len(), self.positions.iter().copied())?;

        Ok(())
    }
}
