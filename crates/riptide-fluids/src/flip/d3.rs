use glam::{UVec3, Vec3};
use ndarray::{azip, Array0, Array1, Array3, Axis};

use crate::{obstacle::{Obstacle, ObstacleSet}, Fluid};

use super::CellType;

#[derive(Debug, Clone)]
pub struct FlipFluid3D {
    /// The density of the fluid, in kg/m³.
    ///
    /// Air in `0` kg/m³ and water is `1000` kg/m³.
    density: f32,
    size: UVec3,
    /// Cell size.
    spacing: f32,

    rest_density: f32,
    particle_radius: f32,
    particle_spacing: f32,
    n_particles: usize,
    particle_resolution: UVec3,

    /// Grid velocities.
    uvws: Array3<Vec3>,
    /// Grid velocity deltas.
    dudvdws: Array3<Vec3>,
    /// Previous grid velocities.
    prev_uvws: Array3<Vec3>,
    /// Pressure of the grid.
    pressure: Array3<f32>,
    /// Solid grid cells. `0.0` for completely solid and `1.0` for not solid.
    solid: Array3<f32>,
    /// Grid cell types (`Fluid`, `Solid` or `Air`).
    cell_type: Array3<CellType>,
    /// Grid densities.
    densities: Array3<f32>,

    cell_particle_count: Array1<usize>,
    first_cell_particle: Array1<usize>,

    /// Particle positions.
    pub positions: Array1<Vec3>,
    /// Particle velocities.
    velocities: Array1<Vec3>,
    /// Fluid roughness approximation per particle.
    roughness: Array1<f32>,
    cell_particle_indices: Array1<usize>,
}

impl FlipFluid3D {
    pub fn new(
        density: f32,
        dimensions: UVec3,
        spacing: f32,
        particle_radius: f32,
    ) -> Self {
        let size = (dimensions.as_vec3() / spacing).floor().as_uvec3() + 1;
        let h = (dimensions.as_vec3() / size.as_vec3()).max_element();

        let shape = (size.x as usize, size.y as usize, size.z as usize);
        let uvs = Array3::from_elem(shape, Vec3::ZERO);
        let dudvs = Array3::from_elem(shape, Vec3::ZERO);
        let prev_uvs = Array3::from_elem(shape, Vec3::ZERO);
        let pressure = Array3::from_elem(shape, 0.0);
        let solid = Array3::from_elem(shape, 1.0);
        let cell_type = Array3::from_elem(shape, CellType::Fluid);
        let densities = Array3::from_elem(shape, 0.0);

        let positions = Array1::from_vec(vec![]);
        let velocities = Array1::from_vec(vec![]);
        let roughness = Array1::from_vec(vec![]);

        let particle_spacing = 2.2 * particle_radius;
        let particle_resolution = (dimensions.as_vec3() / particle_spacing).floor().as_uvec3() + 1;

        let cell_count = (particle_resolution.x * particle_resolution.y * particle_resolution.z) as usize;
        let cell_particle_count = Array1::from_elem(cell_count, 0);
        let first_cell_particle = Array1::from_elem(cell_count + 1, 0);

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
            uvws: uvs,
            dudvdws: dudvs,
            prev_uvws: prev_uvs,
            pressure,
            solid,
            cell_type,
            densities,
            cell_particle_count,
            first_cell_particle,
            positions,
            velocities,
            roughness,
            cell_particle_indices,
        }
    }

    pub fn resize(&mut self, dimensions: UVec3, spacing: f32) {
        let size = (dimensions.as_vec3() / spacing).floor().as_uvec3() + 1;
        self.spacing = (dimensions.as_vec3() / size.as_vec3()).max_element();

        for dim in 0..3 {
            match size[dim].cmp(&self.size[dim]) {
                std::cmp::Ordering::Greater => {
                    let mut s = ndarray::Ix3(0, 0, 0);
                    for i in 0..3 {
                        s[i] = if i == dim {
                            (size[dim] - self.size[dim]) as usize
                        } else {
                            self.size[dim] as usize
                        }
                    }

                    let _ = self.uvws.append(Axis(dim), Array3::from_elem(s, Vec3::ZERO).view());
                    let _ = self.dudvdws.append(Axis(dim), Array3::from_elem(s, Vec3::ZERO).view());
                    let _ = self.prev_uvws.append(Axis(dim), Array3::from_elem(s, Vec3::ZERO).view());
                    let _ = self.pressure.append(Axis(dim), Array3::from_elem(s, 0.0).view());
                    let _ = self.solid.append(Axis(dim), Array3::from_elem(s, 1.0).view());
                    let _ = self.cell_type.append(Axis(dim), Array3::from_elem(s, CellType::Fluid).view());
                    let _ = self.densities.append(Axis(dim), Array3::from_elem(s, 0.0).view());
                },
                std::cmp::Ordering::Less => {
                    for _ in 0..(self.size[dim] - size[dim]) {
                        let i = self.uvws.len_of(Axis(dim)) - 1;
                        self.uvws.remove_index(Axis(dim), i);
                        self.dudvdws.remove_index(Axis(dim), i);
                        self.prev_uvws.remove_index(Axis(dim), i);
                        self.pressure.remove_index(Axis(dim), i);
                        self.solid.remove_index(Axis(dim), i);
                        self.cell_type.remove_index(Axis(dim), i);
                        self.densities.remove_index(Axis(dim), i);
                    }
                },
                std::cmp::Ordering::Equal => (),
            }
        }

        self.size = size;
        self.particle_resolution = (dimensions.as_vec3() / self.particle_spacing).floor().as_uvec3() + 1;
    }

    pub fn insert_particle(&mut self, pos: Vec3) {
        let _ = self.positions.push(Axis(0), Array0::from_elem((), pos).view());
        let _ = self.velocities.push(Axis(0), Array0::from_elem((), Vec3::ZERO).view());
        let _ = self.roughness.push(Axis(0), Array0::from_elem((), 0.0).view());
        let _ = self.cell_particle_indices.push(Axis(0), Array0::from_elem((), 0).view());
        self.n_particles += 1;
    }

    pub fn set_solid(&mut self, i: usize, j: usize, k: usize, v: f32) {
        self.solid[(i, j, k)] = v;
    }

    pub fn iter_positions(&self) -> impl Iterator<Item = &Vec3> {
        self.positions.iter()
    }

    pub fn iter_particles(&self) -> impl Iterator<Item = (&Vec3, &Vec3, &f32)> {
        self.positions.iter().zip(self.velocities.iter()).zip(self.roughness.iter()).map(|((p, v), r)| (p, v, r))
    }

    pub fn size(&self) -> UVec3 {
        self.size
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        (
            Vec3::splat(self.spacing + self.particle_radius),
            (self.size - 1).as_vec3() * self.spacing - self.particle_radius,
        )
    }

    fn integrate_particles(&mut self, dt: f32, gravity: Vec3) {
        self.velocities.map_inplace(|v| *v += dt * gravity);

        azip!((p in &mut self.positions, vel in &self.velocities) {
            *p += vel * dt;
        });
    }

    fn push_particles_apart(&mut self, num_iters: usize) {
        const ROUGHNESS_DIFFUSION: f32 = 0.001;

        self.cell_particle_count.fill(0);
        self.first_cell_particle.fill(0);

        for p in self.positions.iter() {
            let pi = (p / self.particle_spacing).floor().as_uvec3()
                .clamp(UVec3::ZERO, self.particle_resolution - 1);
            let cell_nr = (self.particle_resolution.x * (self.particle_resolution.y * pi.z + pi.y) + pi.x) as usize;
            self.cell_particle_count[cell_nr] += 1;
        }

        let mut first = 0;

        for (count, first_cell) in self.cell_particle_count.iter().zip(self.first_cell_particle.iter_mut()) {
            first += count;
            *first_cell = first;
        }

        self.first_cell_particle[(self.particle_resolution.x * self.particle_resolution.y) as usize] = first;

        for (i, p) in self.positions.iter().enumerate() {
            let pi = (p / self.particle_spacing).floor().as_uvec3()
                .clamp(UVec3::ZERO, self.particle_resolution - 1);
            let cell_nr = (self.particle_resolution.x * (self.particle_resolution.y * pi.z + pi.y) + pi.x) as usize;
            self.first_cell_particle[cell_nr] -= 1;
            self.cell_particle_indices[self.first_cell_particle[cell_nr]] = i;
        }

        let min_dist = 2.0 * self.particle_radius;
        let min_dist2 = min_dist * min_dist;

        for _iter in 0..num_iters {
            for i in 0..self.n_particles {
                let p = self.positions[i];

                let pi = (p / self.particle_spacing).floor().as_uvec3();
                let p0 = pi.max(UVec3::ONE) - 1;
                let p1 = (pi + 1).min(self.particle_resolution - 1);

                for xi in p0.x..=p1.x {
                    for yi in p0.y..=p1.y {
                        for zi in p0.z..=p1.z {
                            let cell_nr = (self.particle_resolution.x * (self.particle_resolution.y * zi + yi) + xi) as usize;
                            let first = self.first_cell_particle[cell_nr];
                            let last = self.first_cell_particle[cell_nr + 1];

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
    }

    fn handle_particle_collisions(&mut self, obstacles: &ObstacleSet<3>, dt: f32) {
        let (min, max) = self.bounds();

        azip!((p in &mut self.positions, v in &mut self.velocities) {
            let sdf = obstacles.sdf((*p).into());
            if sdf.distance < 0.0 {
                // TODO: add velocity of obstacle to this.
                *v = -sdf.distance * Vec3::from(sdf.gradient) / dt;
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

            if p.z < min.z {
                p.z = min.z;
                v.z = 0.0;
            }

            if p.z > max.z {
                p.z = max.z;
                v.z = 0.0;
            }
        });
    }

    fn update_particle_density(&mut self) {
        let h = self.spacing;
        let h1 = h.recip();
        let h2 = 0.5 * h;

        self.densities.fill(0.0);

        for p in self.positions.iter() {
            let pi = p.clamp(Vec3::splat(h), (self.size - 1).as_vec3() * h);

            let p0 = ((pi - h2) * h1).floor().as_uvec3();
            let t = ((pi - h2) - p0.as_vec3() * h) * h1;
            let p1 = (p0 + 1).min(self.size - 2);
            let s = 1.0 - t;

            if p0.x < self.size.x && p0.y < self.size.y && p0.z < self.size.z {
                self.densities[(p0.x as usize, p0.y as usize, p0.z as usize)] += s.x * s.y * s.z;
            }

            if p1.x < self.size.x && p0.y < self.size.y && p0.z < self.size.z {
                self.densities[(p1.x as usize, p0.y as usize, p0.z as usize)] += t.x * s.y * s.z;
            }

            if p0.x < self.size.x && p1.y < self.size.y && p0.z < self.size.z {
                self.densities[(p0.x as usize, p1.y as usize, p0.z as usize)] += s.x * t.y * s.z;
            }

            if p1.x < self.size.x && p1.y < self.size.y && p0.z < self.size.z {
                self.densities[(p1.x as usize, p1.y as usize, p0.z as usize)] += t.x * t.y * s.z;
            }

            if p0.x < self.size.x && p0.y < self.size.y && p1.z < self.size.z {
                self.densities[(p0.x as usize, p0.y as usize, p1.z as usize)] += s.x * s.y * t.z;
            }

            if p1.x < self.size.x && p0.y < self.size.y && p1.z < self.size.z {
                self.densities[(p1.x as usize, p0.y as usize, p1.z as usize)] += t.x * s.y * t.z;
            }

            if p0.x < self.size.x && p1.y < self.size.y && p1.z < self.size.z {
                self.densities[(p0.x as usize, p1.y as usize, p1.z as usize)] += s.x * t.y * t.z;
            }

            if p1.x < self.size.x && p1.y < self.size.y && p1.z < self.size.z {
                self.densities[(p1.x as usize, p1.y as usize, p1.z as usize)] += t.x * t.y * t.z;
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

        self.prev_uvws.assign(&self.uvws);
        self.dudvdws.fill(Vec3::ZERO);
        self.uvws.fill(Vec3::ZERO);

        azip!((cell_type in &mut self.cell_type, &s in &self.solid) {
            *cell_type = if s == 0.0 { CellType::Solid } else { CellType::Air };
        });

        for p in self.positions.iter() {
            let pi = (p * h1).floor().as_uvec3().clamp(UVec3::ZERO, self.size - 1);

            if self.cell_type[(pi.x as usize, pi.y as usize, pi.z as usize)] == CellType::Air {
                self.cell_type[(pi.x as usize, pi.y as usize, pi.z as usize)] = CellType::Fluid;
            }
        }

        for dim in 0..3 {
            let delta = Vec3::new(
                if dim == 0 { 0.0 } else { h2 },
                if dim == 1 { 0.0 } else { h2 },
                if dim == 2 { 0.0 } else { h2 },
            );

            for i in 0..self.n_particles {
                let p = self.positions[i];
                let pi = p.clamp(Vec3::splat(h), (self.size - 1).as_vec3() * h);

                let p0 = ((pi - delta) * h1).floor().as_uvec3().min(self.size - 2);
                let t = ((pi - delta) - p0.as_vec3() * h) * h1;
                let p1 = (p0 + 1).min(self.size - 2);
                let s = 1.0 - t;

                let i0 = (p0.x as usize, p0.y as usize, p0.z as usize);
                let i1 = (p1.x as usize, p0.y as usize, p0.z as usize);
                let i2 = (p0.x as usize, p1.y as usize, p0.z as usize);
                let i3 = (p1.x as usize, p1.y as usize, p0.z as usize);
                let i4 = (p0.x as usize, p0.y as usize, p1.z as usize);
                let i5 = (p1.x as usize, p0.y as usize, p1.z as usize);
                let i6 = (p0.x as usize, p1.y as usize, p1.z as usize);
                let i7 = (p1.x as usize, p1.y as usize, p1.z as usize);

                let d0 = s.x * s.y * s.z;
                let d1 = t.x * s.y * s.z;
                let d2 = s.x * t.y * s.z;
                let d3 = t.x * t.y * s.z;
                let d4 = s.x * s.y * t.z;
                let d5 = t.x * s.y * t.z;
                let d6 = s.x * t.y * t.z;
                let d7 = t.x * t.y * t.z;

                let v = self.velocities[i][dim];
                self.uvws[i0][dim] += v * d0;
                self.uvws[i1][dim] += v * d1;
                self.uvws[i2][dim] += v * d2;
                self.uvws[i3][dim] += v * d3;
                self.uvws[i4][dim] += v * d4;
                self.uvws[i5][dim] += v * d5;
                self.uvws[i6][dim] += v * d6;
                self.uvws[i7][dim] += v * d7;
                self.dudvdws[i0][dim] += d0;
                self.dudvdws[i1][dim] += d1;
                self.dudvdws[i2][dim] += d2;
                self.dudvdws[i3][dim] += d3;
                self.dudvdws[i4][dim] += d4;
                self.dudvdws[i5][dim] += d5;
                self.dudvdws[i6][dim] += d6;
                self.dudvdws[i7][dim] += d7;
            }

            azip!((uv in &mut self.uvws, &dudv in &self.dudvdws) {
                if dudv[dim] > 0.0 {
                    uv[dim] /= dudv[dim];
                }
            });

            for i in 0..self.size.x as usize {
                for j in 0..self.size.y as usize {
                    for k in 0..self.size.z as usize {
                        let solid = self.cell_type[(i, j, k)] == CellType::Solid;

                        if solid || (i > 0 && self.cell_type[(i - 1, j, k)] == CellType::Solid) {
                            self.uvws[(i, j, k)].x = self.prev_uvws[(i, j, k)].x;
                        }

                        if solid || (j > 0 && self.cell_type[(i, j - 1, k)] == CellType::Solid) {
                            self.uvws[(i, j, k)].y = self.prev_uvws[(i, j, k)].y;
                        }

                        if solid || (k > 0 && self.cell_type[(i, j, k - 1)] == CellType::Solid) {
                            self.uvws[(i, j, k)].z = self.prev_uvws[(i, j, k)].z;
                        }
                    }
                }
            }
        }
    }

    fn transfer_velocities_to_particles(&mut self, flip_ratio: f32) {
        let h = self.spacing;
        let h1 = h.recip();
        let h2 = 0.5 * h;

        for dim in 0..3 {
            let delta = Vec3::new(
                if dim == 0 { 0.0 } else { h2 },
                if dim == 1 { 0.0 } else { h2 },
                if dim == 2 { 0.0 } else { h2 },
            );

            for i in 0..self.n_particles {
                let p = self.positions[i];
                let pi = p.clamp(Vec3::splat(h), (self.size - 1).as_vec3() * h);

                let p0 = ((pi - delta) * h1).floor().as_uvec3().min(self.size - 2);
                let t = ((pi - delta) - p0.as_vec3() * h) * h1;
                let p1 = (p0 + 1).min(self.size - 2);
                let s = 1.0 - t;

                let i0 = (p0.x as usize, p0.y as usize, p0.z as usize);
                let i1 = (p1.x as usize, p0.y as usize, p0.z as usize);
                let i2 = (p0.x as usize, p1.y as usize, p0.z as usize);
                let i3 = (p1.x as usize, p1.y as usize, p0.z as usize);
                let i4 = (p0.x as usize, p0.y as usize, p1.z as usize);
                let i5 = (p1.x as usize, p0.y as usize, p1.z as usize);
                let i6 = (p0.x as usize, p1.y as usize, p1.z as usize);
                let i7 = (p1.x as usize, p1.y as usize, p1.z as usize);

                let d0 = s.x * s.y * s.z;
                let d1 = t.x * s.y * s.z;
                let d2 = s.x * t.y * s.z;
                let d3 = t.x * t.y * s.z;
                let d4 = s.x * s.y * t.z;
                let d5 = t.x * s.y * t.z;
                let d6 = s.x * t.y * t.z;
                let d7 = t.x * t.y * t.z;

                let offset = if dim == 0 { (1, 0, 0) } else if dim == 1 { (0, 1, 0) } else { (0, 0, 1) };
                let valid0 = self.cell_type[i0] != CellType::Air || self.cell_type[(i0.0 - offset.0, i0.1 - offset.1, i0.2 - offset.2)] != CellType::Air;
                let valid1 = self.cell_type[i1] != CellType::Air || self.cell_type[(i1.0 - offset.0, i1.1 - offset.1, i1.2 - offset.2)] != CellType::Air;
                let valid2 = self.cell_type[i2] != CellType::Air || self.cell_type[(i2.0 - offset.0, i2.1 - offset.1, i2.2 - offset.2)] != CellType::Air;
                let valid3 = self.cell_type[i3] != CellType::Air || self.cell_type[(i3.0 - offset.0, i3.1 - offset.1, i3.2 - offset.2)] != CellType::Air;
                let valid4 = self.cell_type[i4] != CellType::Air || self.cell_type[(i4.0 - offset.0, i4.1 - offset.1, i4.2 - offset.2)] != CellType::Air;
                let valid5 = self.cell_type[i5] != CellType::Air || self.cell_type[(i5.0 - offset.0, i5.1 - offset.1, i5.2 - offset.2)] != CellType::Air;
                let valid6 = self.cell_type[i6] != CellType::Air || self.cell_type[(i6.0 - offset.0, i6.1 - offset.1, i6.2 - offset.2)] != CellType::Air;
                let valid7 = self.cell_type[i7] != CellType::Air || self.cell_type[(i7.0 - offset.0, i7.1 - offset.1, i7.2 - offset.2)] != CellType::Air;
                let v0 = if valid0 { 1.0 } else { 0.0 };
                let v1 = if valid1 { 1.0 } else { 0.0 };
                let v2 = if valid2 { 1.0 } else { 0.0 };
                let v3 = if valid3 { 1.0 } else { 0.0 };
                let v4 = if valid4 { 1.0 } else { 0.0 };
                let v5 = if valid5 { 1.0 } else { 0.0 };
                let v6 = if valid6 { 1.0 } else { 0.0 };
                let v7 = if valid7 { 1.0 } else { 0.0 };

                let v = self.velocities[i][dim];
                let d = v0 * d0 + v1 * d1 + v2 * d2 + v3 * d3 + v4 * d4 + v5 * d5 + v6 * d6 + v7 * d7;

                if d > 0.0 {
                    let picv = (v0 * d0 * self.uvws[i0][dim] + v1 * d1 * self.uvws[i1][dim] 
                        + v2 * d2 * self.uvws[i2][dim] + v3 * d3 * self.uvws[i3][dim]
                        + v4 * d4 * self.uvws[i4][dim] + v5 * d5 * self.uvws[i5][dim]
                        + v6 * d6 * self.uvws[i6][dim] + v7 * d7 * self.uvws[i7][dim]) / d;
                    let corr = (v0 * d0 * (self.uvws[i0][dim] - self.prev_uvws[i0][dim]) 
                        + v1 * d1 * (self.uvws[i1][dim] - self.prev_uvws[i1][dim])
                        + v2 * d2 * (self.uvws[i2][dim] - self.prev_uvws[i2][dim])
                        + v3 * d3 * (self.uvws[i3][dim] - self.prev_uvws[i3][dim])
                        + v4 * d4 * (self.uvws[i4][dim] - self.prev_uvws[i4][dim])
                        + v5 * d5 * (self.uvws[i5][dim] - self.prev_uvws[i5][dim])
                        + v6 * d6 * (self.uvws[i6][dim] - self.prev_uvws[i6][dim])
                        + v7 * d7 * (self.uvws[i7][dim] - self.prev_uvws[i7][dim])) / d;
                    let flipv = v + corr;

                    self.velocities[i][dim] = (1.0 - flip_ratio) * picv + flip_ratio * flipv;
                }
            }
        }
    }

    fn solve_incompressibility(&mut self, num_iters: usize, dt: f32, over_relaxation: f32, compensate_drift: bool) {
        self.pressure.fill(0.0);
        self.prev_uvws.assign(&self.uvws);

        let cp = self.density * self.spacing / dt;

        for _iter in 0..num_iters {
            for i in 1..self.size.x as usize - 1 {
                for j in 1..self.size.y as usize - 1 {
                    for k in 1..self.size.z as usize - 1 {
                        if self.cell_type[(i, j, k)] != CellType::Fluid {
                            continue;
                        }

                        let center = (i, j, k);
                        let left = (i - 1, j, k);
                        let right = (i + 1, j, k);
                        let bottom = (i, j - 1, k);
                        let top = (i, j + 1, k);
                        let back = (i, j, k - 1);
                        let front = (i, j, k + 1);

                        let sx0 = self.solid[left];
                        let sx1 = self.solid[right];
                        let sy0 = self.solid[bottom];
                        let sy1 = self.solid[top];
                        let sz0 = self.solid[back];
                        let sz1 = self.solid[front];
                        let s = sx0 + sx1 + sy0 + sy1 + sz0 + sz1;

                        if s == 0.0 {
                            continue;
                        }

                        let mut div = self.uvws[right].x - self.uvws[center].x
                            + self.uvws[top].y - self.uvws[center].y
                            + self.uvws[front].z - self.uvws[center].z;

                        if self.rest_density > 0.0 && compensate_drift {
                            let stiffness = 1.0;
                            let compression = self.densities[(i, j, k)] - self.rest_density;
                            if compression > 0.0 {
                                div -= stiffness * compression;
                            }
                        }

                        let mut p = -div / s;
                        p *= over_relaxation;
                        self.pressure[center] += cp * p;

                        self.uvws[center].x -= sx0 * p;
                        self.uvws[right].x += sx1 * p;
                        self.uvws[center].y -= sy0 * p;
                        self.uvws[top].y += sy1 * p;
                        self.uvws[center].z -= sz0 * p;
                        self.uvws[front].z += sz1 * p;
                    }
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
            let pi = (p * h1).floor().as_uvec3().clamp(UVec3::ONE, self.size - 1);

            self.roughness[i] = (self.roughness[i] - s).clamp(0.0, 1.0);

            if d0 > 0.0 {
                let rel_density = self.densities[(pi.x as usize, pi.y as usize, pi.z as usize)] / d0;
                if rel_density < 0.7 {
                    let s = 0.8;
                    self.roughness[i] = s;
                }
            }
        }
    }

    pub fn set_obstacles(&mut self, obstacles: &ObstacleSet<3>, dt: f32) {
        for i in 1..self.size.x as usize - 2 {
            for j in 1..self.size.y as usize - 2 {
                for k in 1..self.size.z as usize - 2 {
                    self.solid[(i, j, k)] = 1.0;
                    let p = Vec3::new(i as f32 + 0.5, j as f32 + 0.5, k as f32 + 0.5) * self.spacing;
                    let sdf = obstacles.sdf(p.into());

                    if sdf.distance < 0.0 {
                        // TODO: add velocity of obstacle to this.
                        let v = -sdf.distance * Vec3::from(sdf.gradient) / dt;
                        self.solid[(i, j, k)] = 0.0;
                        self.uvws[(i, j, k)] = v;
                        self.uvws[(i + 1, j, k)].x = v.x;
                        self.uvws[(i, j + 1, k)].y = v.y;
                        self.uvws[(i, j, k + 1)].z = v.z;
                    }
                }
            }
        }
    }
}


pub struct FlipFluid3DParams {
    pub num_substeps: usize,
    pub gravity: Vec3,
    pub flip_ratio: f32,
    pub num_pressure_iters: usize,
    pub num_particle_iters: usize,
    pub over_relaxation: f32,
    pub compensate_drift: bool,
    pub separate_particles: bool,
}

impl Default for FlipFluid3DParams {
    fn default() -> Self {
        Self {
            num_substeps: 2,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            flip_ratio: 0.9,
            num_pressure_iters: 100,
            num_particle_iters: 2,
            over_relaxation: 1.9,
            compensate_drift: true,
            separate_particles: true,
        }
    }
}

impl Fluid<3> for FlipFluid3D {
    type Params = FlipFluid3DParams;

    fn step(&mut self, dt: f32, params: &Self::Params, obstacles: &ObstacleSet<3>) {
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
}
