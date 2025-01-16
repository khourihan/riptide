use std::f32::consts::PI;

use glam::{UVec3, Vec3};

use crate::{obstacle::{Obstacle, ObstacleSet}, Fluid};

use super::{mac_3d::MacGrid3D, CellType};

#[derive(Debug, Clone)]
pub struct FlipFluid3D {
    mac: MacGrid3D,
    /// The density of the fluid, in kg/m³.
    ///
    /// Air in `1` kg/m³ (approximated as `0` kg/m³) and water is `1000` kg/m³.
    density: f32,

    rest_density: f32,
    /// Radius of particles.
    particle_radius: f32,
    /// Cell size of particle grid.
    particle_spacing: f32,
    /// Total number of particles.
    n_particles: usize,
    /// Resolution of particle grid.
    particle_resolution: UVec3,

    /// Number of particles per cell of particle grid.
    cell_particle_count: Vec<usize>,
    first_cell_particle: Vec<usize>,

    /// Particle positions.
    pub positions: Vec<Vec3>,
    /// Particle velocities.
    velocities: Vec<Vec3>,
    cell_particle_indices: Vec<usize>,
}

impl FlipFluid3D {
    pub fn new(
        density: f32,
        size: Vec3,
        spacing: f32,
        particle_radius: f32,
    ) -> Self {
        let positions = vec![];
        let velocities = vec![];

        let particle_spacing = 2.2 * particle_radius;
        let particle_resolution = (size / particle_spacing).floor().as_uvec3() + 1;

        let cell_count = (particle_resolution.x * particle_resolution.y * particle_resolution.z) as usize;
        let cell_particle_count = vec![0; cell_count];
        let first_cell_particle = vec![0; cell_count + 1];

        let cell_particle_indices = vec![];

        Self {
            mac: MacGrid3D::new(size, spacing),
            density,
            rest_density: 0.0,
            particle_radius,
            particle_spacing,
            n_particles: 0,
            particle_resolution,
            cell_particle_count,
            first_cell_particle,
            positions,
            velocities,
            cell_particle_indices,
        }
    }

    pub fn insert_particle(&mut self, pos: Vec3) {
        self.positions.push(pos);
        self.velocities.push(Vec3::ZERO);
        self.cell_particle_indices.push(0);
        self.n_particles += 1;
    }

    pub fn set_solid(&mut self, i: usize, j: usize, k: usize, v: bool) {
        let idx = self.mac.idx(i, j, k);
        self.mac.solid[idx] = v;
    }

    pub fn iter_positions(&self) -> impl Iterator<Item = &Vec3> {
        self.positions.iter()
    }

    pub fn iter_particles(&self) -> impl Iterator<Item = (&Vec3, &Vec3)> {
        self.positions.iter().zip(self.velocities.iter())
    }

    pub fn size(&self) -> UVec3 {
        self.mac.grid_size
    }

    pub fn spacing(&self) -> f32 {
        self.mac.spacing
    }

    fn integrate_particles(&mut self, dt: f32, gravity: Vec3, obstacles: &ObstacleSet<3>) {
        self.positions.iter().zip(self.velocities.iter_mut()).for_each(|(p, v)| {
            let sdf = obstacles.sdf((*p).into());
            if sdf.distance < 0.0 {
                *v = -sdf.distance * Vec3::from(sdf.gradient) / dt;
            }
        });
        
        self.velocities.iter_mut().for_each(|v| *v += dt * gravity);

        self.positions.iter_mut().zip(self.velocities.iter()).for_each(|(p, v)| {
            *p += v * dt;
        });
    }

    fn separate_particles(&mut self, num_iters: usize) {
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
                            }
                        }
                    }
                }
            }
        }
    }

    fn handle_particle_collisions(&mut self) {
        let min = Vec3::splat(self.mac.spacing + self.particle_radius);
        let max = self.mac.size - min;

        self.positions.iter_mut().zip(self.velocities.iter_mut()).for_each(|(p, v)| {
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
        let h = self.mac.spacing;
        let h1 = self.mac.inv_spacing;
        let h2 = 0.5 * h;

        self.mac.densities.fill(0.0);

        for p in self.positions.iter() {
            let pi = p.clamp(Vec3::splat(h), (self.mac.grid_size - 1).as_vec3() * h);

            let p0 = ((pi - h2) * h1).floor().as_uvec3();
            let t = ((pi - h2) - p0.as_vec3() * h) * h1;
            let p1 = (p0 + 1).min(self.mac.grid_size - 2);
            let s = 1.0 - t;

            if p0.x < self.mac.grid_size.x && p0.y < self.mac.grid_size.y && p0.z < self.mac.grid_size.z {
                let idx = self.mac.idx(p0.x as usize, p0.y as usize, p0.z as usize);
                self.mac.densities[idx] += s.x * s.y * s.z;
            }

            if p1.x < self.mac.grid_size.x && p0.y < self.mac.grid_size.y && p0.z < self.mac.grid_size.z {
                let idx = self.mac.idx(p1.x as usize, p0.y as usize, p0.z as usize);
                self.mac.densities[idx] += t.x * s.y * s.z;
            }

            if p0.x < self.mac.grid_size.x && p1.y < self.mac.grid_size.y && p0.z < self.mac.grid_size.z {
                let idx = self.mac.idx(p0.x as usize, p1.y as usize, p0.z as usize);
                self.mac.densities[idx] += s.x * t.y * s.z;
            }

            if p1.x < self.mac.grid_size.x && p1.y < self.mac.grid_size.y && p0.z < self.mac.grid_size.z {
                let idx = self.mac.idx(p1.x as usize, p1.y as usize, p0.z as usize);
                self.mac.densities[idx] += t.x * t.y * s.z;
            }

            if p0.x < self.mac.grid_size.x && p0.y < self.mac.grid_size.y && p1.z < self.mac.grid_size.z {
                let idx = self.mac.idx(p0.x as usize, p0.y as usize, p1.z as usize);
                self.mac.densities[idx] += s.x * s.y * t.z;
            }

            if p1.x < self.mac.grid_size.x && p0.y < self.mac.grid_size.y && p1.z < self.mac.grid_size.z {
                let idx = self.mac.idx(p1.x as usize, p0.y as usize, p1.z as usize);
                self.mac.densities[idx] += t.x * s.y * t.z;
            }

            if p0.x < self.mac.grid_size.x && p1.y < self.mac.grid_size.y && p1.z < self.mac.grid_size.z {
                let idx = self.mac.idx(p0.x as usize, p1.y as usize, p1.z as usize);
                self.mac.densities[idx] += s.x * t.y * t.z;
            }

            if p1.x < self.mac.grid_size.x && p1.y < self.mac.grid_size.y && p1.z < self.mac.grid_size.z {
                let idx = self.mac.idx(p1.x as usize, p1.y as usize, p1.z as usize);
                self.mac.densities[idx] += t.x * t.y * t.z;
            }
        }

        if self.rest_density == 0.0 {
            let mut sum: f32 = 0.0;
            let mut num_fluid_cells: usize = 0;

            for (&cell_type, &density) in self.mac.cell_type.iter().zip(self.mac.densities.iter()) {
                if cell_type == CellType::Fluid {
                    sum += density;
                    num_fluid_cells += 1;
                }
            }

            if num_fluid_cells > 0 {
                self.rest_density = sum / num_fluid_cells as f32;
            }
        }
    }

    fn particle_to_grid(&mut self) {
        self.mac.u.fill(0.0);
        self.mac.v.fill(0.0);
        self.mac.w.fill(0.0);
        self.mac.weight_u.fill(0.0);
        self.mac.weight_v.fill(0.0);
        self.mac.weight_w.fill(0.0);

        let nx = self.mac.nx;
        let ny = self.mac.ny;
        let nz = self.mac.nz;

        let h_half = 0.5 * self.mac.spacing;
        let h1 = self.mac.inv_spacing;
        let h = 2.0 * self.mac.spacing;
        let h2 = h * h;
        let h4 = h2 * h2;
        let coeff = 315.0 / (64.0 * PI * h4 * h4 * h);

        for (cell_type, &s) in self.mac.cell_type.iter_mut().zip(self.mac.solid.iter()) {
            *cell_type = if s { CellType::Solid } else { CellType::Air };
        }

        for i in 0..self.n_particles {
            let pos = self.positions[i];
            let vel = self.velocities[i];

            let pi = (pos * h1).floor().as_uvec3().clamp(UVec3::ZERO, self.mac.grid_size - 1);

            let idx = self.mac.idx(pi.x as usize, pi.y as usize, pi.z as usize);
            if self.mac.cell_type[idx] == CellType::Air {
                self.mac.cell_type[idx] = CellType::Fluid;
            }

            if pi.x >= 2 && pi.y >= 2 && pi.z >= 2 && pi.x < self.mac.grid_size.x - 3
                && pi.y < self.mac.grid_size.y - 3 && pi.z < self.mac.grid_size.z - 3
            {
                for k in pi.z as usize - 2..=pi.z as usize + 3 {
                    for j in pi.y as usize - 2..=pi.y as usize + 3 {
                        for i in pi.x as usize - 2..=pi.x as usize + 3 {
                            let rx = pos.x - i as f32 * self.mac.spacing;
                            let ry = pos.y - j as f32 * self.mac.spacing;
                            let rz = pos.z - k as f32 * self.mac.spacing;

                            let x_diff = h2 - ry * ry - rz * rz - (rx + h_half) * (rx + h_half);
                            let y_diff = h2 - rx * rx - rz * rz - (ry + h_half) * (ry + h_half);
                            let z_diff = h2 - rx * rx - ry * ry - (rz + h_half) * (rz + h_half);

                            if x_diff >= 0.0 {
                                let u_weight_1 = coeff * x_diff * x_diff * x_diff;
                                let u_idx = self.mac.u_idx(i, j, k);

                                self.mac.u[u_idx] += u_weight_1 * vel.x;
                                self.mac.weight_u[u_idx] += u_weight_1;
                            }

                            if y_diff >= 0.0 {
                                let v_weight_1 = coeff * y_diff * y_diff * y_diff;
                                let v_idx = self.mac.v_idx(i, j, k);

                                self.mac.v[v_idx] += v_weight_1 * vel.y;
                                self.mac.weight_v[v_idx] += v_weight_1;
                            }

                            if z_diff >= 0.0 {
                                let w_weight_1 = coeff * z_diff * z_diff * z_diff;
                                let w_idx = self.mac.w_idx(i, j, k);

                                self.mac.w[w_idx] += w_weight_1 * vel.z;
                                self.mac.weight_w[w_idx] += w_weight_1;
                            }
                        }
                    }
                }
            } else {
                for k in pi.z.max(2) as usize - 2..=pi.z as usize + 3 {
                    for j in pi.y.max(2) as usize - 2..=pi.y as usize + 3 {
                        for i in pi.x.max(2) as usize - 2..=pi.x as usize + 3 {
                            let rx = pos.x - i as f32 * self.mac.spacing;
                            let ry = pos.y - j as f32 * self.mac.spacing;
                            let rz = pos.z - k as f32 * self.mac.spacing;

                            if i <= nx && j < ny && k < nz {
                                let x_diff = h2 - ry * ry - rz * rz - (rx + h_half) * (rx + h_half);

                                if x_diff >= 0.0 {
                                    let u_weight_1 = coeff * x_diff * x_diff * x_diff;
                                    let u_idx = self.mac.u_idx(i, j, k);

                                    self.mac.u[u_idx] += u_weight_1 * vel.x;
                                    self.mac.weight_u[u_idx] += u_weight_1;
                                }
                            }

                            if i < nx && j <= ny && k < nz {
                                let y_diff = h2 - rx * rx - rz * rz - (ry + h_half) * (ry + h_half);

                                if y_diff >= 0.0 {
                                    let v_weight_1 = coeff * y_diff * y_diff * y_diff;
                                    let v_idx = self.mac.v_idx(i, j, k);

                                    self.mac.v[v_idx] += v_weight_1 * vel.y;
                                    self.mac.weight_v[v_idx] += v_weight_1;
                                }
                            }

                            if i < nx && j < ny && k <= nz {
                                let z_diff = h2 - rx * rx - ry * ry - (rz + h_half) * (rz + h_half);

                                if z_diff >= 0.0 {
                                    let w_weight_1 = coeff * z_diff * z_diff * z_diff;
                                    let w_idx = self.mac.w_idx(i, j, k);

                                    self.mac.w[w_idx] += w_weight_1 * vel.z;
                                    self.mac.weight_w[w_idx] += w_weight_1;
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut visited_u = vec![false; (nx + 1) * ny * nz];
        let mut visited_v = vec![false; nx * (ny + 1) * nz];
        let mut visited_w = vec![false; nx * ny * (nz + 1)];

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let u_idx = self.mac.u_idx(i, j, k);
                    let v_idx = self.mac.v_idx(i, j, k);
                    let w_idx = self.mac.w_idx(i, j, k);

                    let u_weight = self.mac.weight_u[u_idx];
                    let v_weight = self.mac.weight_v[v_idx];
                    let w_weight = self.mac.weight_w[w_idx];

                    if u_weight != 0.0 {
                        self.mac.u[u_idx] /= u_weight;
                        visited_u[u_idx] = true;
                    }

                    if v_weight != 0.0 {
                        self.mac.v[v_idx] /= v_weight;
                        visited_v[v_idx] = true;
                    }

                    if w_weight != 0.0 {
                        self.mac.w[w_idx] /= w_weight;
                        visited_w[w_idx] = true;
                    }
                }
            }
        }

        for k in 0..nz {
            for j in 0..ny {
                let u_idx = self.mac.u_idx(nx, j, k);
                let u_weight = self.mac.weight_u[u_idx];

                if u_weight != 0.0 {
                    self.mac.u[u_idx] /= u_weight;
                    visited_u[u_idx] = true;
                }
            }
        }

        for k in 0..nz {
            for i in 0..nx {
                let v_idx = self.mac.v_idx(i, ny, k);
                let v_weight = self.mac.weight_v[v_idx];

                if v_weight != 0.0 {
                    self.mac.v[v_idx] /= v_weight;
                    visited_v[v_idx] = true;
                }
            }
        }

        for j in 0..ny {
            for i in 0..nx {
                let w_idx = self.mac.w_idx(i, j, nz);
                let w_weight = self.mac.weight_w[w_idx];

                if w_weight != 0.0 {
                    self.mac.w[w_idx] /= w_weight;
                    visited_w[w_idx] = true;
                }
            }
        }

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let u_idx = self.mac.u_idx(i, j, k);
                    let v_idx = self.mac.v_idx(i, j, k);
                    let w_idx = self.mac.w_idx(i, j, k);

                    if !visited_u[u_idx] {
                        let mut u_counter: u8 = 0;

                        let u_left = if i > 0 && visited_u[u_idx - 1] {
                            u_counter += 1;
                            self.mac.u[u_idx - 1]
                        } else {
                            0.0
                        };

                        let u_right = if i < nx - 1 && visited_u[u_idx + 1] {
                            u_counter += 1;
                            self.mac.u[u_idx + 1]
                        } else {
                            0.0
                        };

                        let u_down = if j > 0 && visited_u[u_idx - (nx + 1)] {
                            u_counter += 1;
                            self.mac.u[u_idx - (nx + 1)]
                        } else {
                            0.0
                        };

                        let u_up = if j < ny - 1 && visited_u[u_idx + (nx + 1)] {
                            u_counter += 1;
                            self.mac.u[u_idx + (nx + 1)]
                        } else {
                            0.0
                        };

                        let u_back = if k > 0 && visited_u[u_idx - (nx + 1) * ny] {
                            u_counter += 1;
                            self.mac.u[u_idx - (nx + 1) * ny]
                        } else {
                            0.0
                        };

                        let u_front = if k < nz - 1 && visited_u[u_idx + (nx + 1) * ny] {
                            u_counter += 1;
                            self.mac.u[u_idx + (nx + 1) * ny]
                        } else {
                            0.0
                        };

                        if u_counter != 0 {
                            self.mac.u[u_idx] = (u_left + u_right + u_down + u_up + u_back + u_front) / u_counter as f32;
                        }
                    }

                    if !visited_v[v_idx] {
                        let mut v_counter: u8 = 0;

                        let v_left = if i > 0 && visited_v[v_idx - 1] {
                            v_counter += 1;
                            self.mac.v[v_idx - 1]
                        } else {
                            0.0
                        };

                        let v_right = if i < nx - 1 && visited_v[v_idx + 1] {
                            v_counter += 1;
                            self.mac.v[v_idx + 1]
                        } else {
                            0.0
                        };

                        let v_down = if j > 0 && visited_v[v_idx - nx] {
                            v_counter += 1;
                            self.mac.v[v_idx - nx]
                        } else {
                            0.0
                        };

                        let v_up = if j < ny - 1 && visited_v[v_idx + nx] {
                            v_counter += 1;
                            self.mac.v[v_idx + nx]
                        } else {
                            0.0
                        };

                        let v_back = if k > 0 && visited_v[v_idx - nx * (ny + 1)] {
                            v_counter += 1;
                            self.mac.v[v_idx - nx * (ny + 1)]
                        } else {
                            0.0
                        };

                        let v_front = if k < nz - 1 && visited_v[v_idx + nx * (ny + 1)] {
                            v_counter += 1;
                            self.mac.v[v_idx + nx * (ny + 1)]
                        } else {
                            0.0
                        };

                        if v_counter != 0 {
                            self.mac.v[v_idx] = (v_left + v_right + v_down + v_up + v_back + v_front) / v_counter as f32;
                        }
                    }

                    if !visited_w[w_idx] {
                        let mut w_counter: u8 = 0;

                        let w_left = if i > 0 && visited_w[w_idx - 1] {
                            w_counter += 1;
                            self.mac.w[w_idx - 1]
                        } else {
                            0.0
                        };

                        let w_right = if i < nx - 1 && visited_w[w_idx + 1] {
                            w_counter += 1;
                            self.mac.w[w_idx + 1]
                        } else {
                            0.0
                        };

                        let w_down = if j > 0 && visited_w[w_idx - nx] {
                            w_counter += 1;
                            self.mac.w[w_idx - nx]
                        } else {
                            0.0
                        };

                        let w_up = if j < ny - 1 && visited_w[w_idx + nx] {
                            w_counter += 1;
                            self.mac.w[w_idx + nx]
                        } else {
                            0.0
                        };

                        let w_back = if k > 0 && visited_w[w_idx - nx * ny] {
                            w_counter += 1;
                            self.mac.w[w_idx - nx * ny]
                        } else {
                            0.0
                        };

                        let w_front = if k < nz - 1 && visited_w[w_idx + nx * ny] {
                            w_counter += 1;
                            self.mac.w[w_idx + nx * ny]
                        } else {
                            0.0
                        };

                        if w_counter != 0 {
                            self.mac.w[w_idx] = (w_left + w_right + w_down + w_up + w_back + w_front) / w_counter as f32;
                        }
                    }
                }
            }
        }

        for k in 0..nz {
            for j in 0..ny {
                let u_idx = self.mac.u_idx(nx, j, k);

                if !visited_u[u_idx] {
                    let mut u_counter: u8 = 0;

                    let u_left = if visited_u[u_idx - 1] {
                        u_counter += 1;
                        self.mac.u[u_idx - 1]
                    } else {
                        0.0
                    };

                    let u_down = if j > 0 && visited_u[u_idx - (nx + 1)] {
                        u_counter += 1;
                        self.mac.u[u_idx - (nx + 1)]
                    } else {
                        0.0
                    };

                    let u_up = if j < ny - 1 && visited_u[u_idx + (nx + 1)] {
                        u_counter += 1;
                        self.mac.u[u_idx + (nx + 1)]
                    } else {
                        0.0
                    };

                    let u_back = if k > 0 && visited_u[u_idx - (nx + 1) * ny] {
                        u_counter += 1;
                        self.mac.u[u_idx - (nx + 1) * ny]
                    } else {
                        0.0
                    };

                    let u_front = if k < nz - 1 && visited_u[u_idx + (nx + 1) * ny] {
                        u_counter += 1;
                        self.mac.u[u_idx + (nx + 1) * ny]
                    } else {
                        0.0
                    };

                    if u_counter != 0 {
                        self.mac.u[u_idx] = (u_left + u_down + u_up + u_back + u_front) / u_counter as f32;
                    }
                }
            }
        }

        for k in 0..nz {
            for i in 0..nx {
                let v_idx = self.mac.v_idx(i, ny, k);

                if !visited_v[v_idx] {
                    let mut v_counter: u8 = 0;

                    let v_left = if i > 0 && visited_v[v_idx - 1] {
                        v_counter += 1;
                        self.mac.v[v_idx - 1]
                    } else {
                        0.0
                    };

                    let v_right = if i < nx - 1 && visited_v[v_idx + 1] {
                        v_counter += 1;
                        self.mac.v[v_idx + 1]
                    } else {
                        0.0
                    };

                    let v_down = if visited_v[v_idx - nx] {
                        v_counter += 1;
                        self.mac.v[v_idx - nx]
                    } else {
                        0.0
                    };

                    let v_back = if k > 0 && visited_v[v_idx - nx * (ny + 1)] {
                        v_counter += 1;
                        self.mac.v[v_idx - nx * (ny + 1)]
                    } else {
                        0.0
                    };

                    let v_front = if k < nz - 1 && visited_v[v_idx + nx * (ny + 1)] {
                        v_counter += 1;
                        self.mac.v[v_idx + nx * (ny + 1)]
                    } else {
                        0.0
                    };

                    if v_counter != 0 {
                        self.mac.v[v_idx] = (v_left + v_right + v_down + v_back + v_front) / v_counter as f32;
                    }
                }
            }
        }

        for j in 0..ny {
            for i in 0..nx {
                let w_idx = self.mac.w_idx(i, j, nz);

                if !visited_w[w_idx] {
                    let mut w_counter: u8 = 0;

                    let w_left = if i > 0 && visited_w[w_idx - 1] {
                        w_counter += 1;
                        self.mac.w[w_idx - 1]
                    } else {
                        0.0
                    };

                    let w_right = if i < nx - 1 && visited_w[w_idx + 1] {
                        w_counter += 1;
                        self.mac.w[w_idx + 1]
                    } else {
                        0.0
                    };

                    let w_down = if j > 0 && visited_w[w_idx - nx] {
                        w_counter += 1;
                        self.mac.w[w_idx - nx]
                    } else {
                        0.0
                    };

                    let w_up = if j < ny - 1 && visited_w[w_idx + nx] {
                        w_counter += 1;
                        self.mac.w[w_idx + nx]
                    } else {
                        0.0
                    };

                    let w_back = if visited_w[w_idx - nx * ny] {
                        w_counter += 1;
                        self.mac.w[w_idx - nx * ny]
                    } else {
                        0.0
                    };

                    if w_counter != 0 {
                        self.mac.w[w_idx] = (w_left + w_right + w_down + w_up + w_back) / w_counter as f32;
                    }
                }
            }
        }
    }

    fn grid_to_particle(&mut self, alpha: f32) {
        let nx = self.mac.grid_size.x;
        let ny = self.mac.grid_size.y;
        let nz = self.mac.grid_size.z;
        let h1 = self.mac.inv_spacing;

        for i in 0..self.n_particles {
            let p0 = self.positions[i];
            let v0 = self.velocities[i];

            let pi = (p0 * h1).floor().as_uvec3().clamp(UVec3::ZERO, self.mac.grid_size - 1);

            let (interp_u, interp_u_star) = self.mac.get_bilerp_u(p0);
            let (interp_v, interp_v_star) = self.mac.get_bilerp_v(p0);
            let (interp_w, interp_w_star) = self.mac.get_bilerp_w(p0);

            let interp_n1 = Vec3::new(interp_u, interp_v, interp_w);
            let interp_star = Vec3::new(interp_u_star, interp_v_star, interp_w_star);

            let u_update = if pi.x == 0 || pi.x == nx - 1 || pi.y == 0 || pi.y == ny - 1 || pi.z == 0 || pi.z == nz - 1 {
                interp_n1 + (v0 - interp_star) * (1.0 - f32::min(1.0, 2.0 * alpha))
            } else {
                interp_n1 + (v0 - interp_star) * (1.0 - alpha)
            };

            self.velocities[i] = u_update;
        }
    }

    fn solve_pressure(&mut self, num_iters: usize, dt: f32, over_relaxation: f32, compensate_drift: bool) {
        self.mac.pressure.fill(0.0);

        let nx = self.mac.nx;
        let ny = self.mac.ny;
        let nz = self.mac.nz;
        let cp = self.density * self.mac.spacing / dt;

        for _iter in 0..num_iters {
            for k in 1..nz - 1 {
                for j in 1..ny - 1 {
                    for i in 1..nx - 1 {
                        let idx = self.mac.idx(i, j, k);
                        let u_idx = self.mac.u_idx(i, j, k);
                        let v_idx = self.mac.v_idx(i, j, k);
                        let w_idx = self.mac.w_idx(i, j, k);
                        
                        if self.mac.cell_type[idx] != CellType::Fluid {
                            continue;
                        }

                        let sx0 = if self.mac.solid[idx - 1] { 0.0 } else { 1.0 };
                        let sx1 = if self.mac.solid[idx + 1] { 0.0 } else { 1.0 };
                        let sy0 = if self.mac.solid[idx - nx] { 0.0 } else { 1.0 };
                        let sy1 = if self.mac.solid[idx + nx] { 0.0 } else { 1.0 };
                        let sz0 = if self.mac.solid[idx - nx * ny] { 0.0 } else { 1.0 };
                        let sz1 = if self.mac.solid[idx + nx * ny] { 0.0 } else { 1.0 };
                        let s = sx0 + sx1 + sy0 + sy1 + sz0 + sz1;

                        if s == 0.0 {
                            continue;
                        }

                        let mut div = self.mac.u[u_idx + 1] - self.mac.u[u_idx]
                            + self.mac.v[v_idx + nx] - self.mac.v[v_idx]
                            + self.mac.w[w_idx + nx * ny] - self.mac.w[w_idx];

                        if self.rest_density > 0.0 && compensate_drift {
                            let stiffness = 1.0;
                            let compression = self.mac.densities[idx] - self.rest_density;
                            if compression > 0.0 {
                                div -= stiffness * compression;
                            }
                        }

                        let mut p = -div / s;
                        p *= over_relaxation;
                        self.mac.pressure[idx] += cp * p;

                        self.mac.u[u_idx] -= sx0 * p;
                        self.mac.u[u_idx + 1] += sx1 * p;
                        self.mac.v[v_idx] -= sy0 * p;
                        self.mac.v[v_idx + nx] += sy1 * p;
                        self.mac.w[w_idx] -= sz0 * p;
                        self.mac.w[w_idx + nx * ny] += sz1 * p;
                    }
                }
            }
        }
    }

    fn set_obstacles(&mut self, obstacles: &ObstacleSet<3>, dt: f32) {
        for i in 1..self.mac.nx - 1 {
            for j in 1..self.mac.ny - 1 {
                for k in 1..self.mac.nz - 1 {
                    let idx = self.mac.idx(i, j, k);
                    let u_idx = self.mac.u_idx(i, j, k);
                    let v_idx = self.mac.v_idx(i, j, k);
                    let w_idx = self.mac.w_idx(i, j, k);

                    self.mac.solid[idx] = false;
                    let p = Vec3::new(i as f32 + 0.5, j as f32 + 0.5, k as f32 + 0.5) * self.mac.spacing;
                    let sdf = obstacles.sdf(p.into());

                    if sdf.distance < 0.0 {
                        let v = -sdf.distance * Vec3::from(sdf.gradient) / dt;
                        self.mac.solid[idx] = true;
                        self.mac.u[u_idx] = v.x;
                        self.mac.v[v_idx] = v.y;
                        self.mac.w[w_idx] = v.z;
                        self.mac.u[u_idx + 1] = v.x;
                        self.mac.v[v_idx + self.mac.nx] = v.y;
                        self.mac.w[w_idx + self.mac.nx * self.mac.ny] = v.z;
                    }
                }
            }
        }
    }

    pub fn sample_density(&self, p: Vec3) -> f32 {
        let h1 = self.mac.inv_spacing;

        let x0 = (p.x * h1).floor() as usize;
        let x1 = if (p.x * h1).fract() > 0.5 { x0.min(self.mac.nx - 2) + 1 } else { x0.max(1) - 1 };
        let y0 = (p.y * h1).floor() as usize;
        let y1 = if (p.y * h1).fract() > 0.5 { y0.min(self.mac.ny - 2) + 1 } else { y0.max(1) - 1 };
        let z0 = (p.z * h1).floor() as usize;
        let z1 = if (p.z * h1).fract() > 0.5 { z0.min(self.mac.nz - 2) + 1 } else { z0.max(1) - 1 };

        let dx = (p.x * h1) - (x0 as f32 + 0.5);
        let dy = (p.y * h1) - (y0 as f32 + 0.5);
        let dz = (p.z * h1) - (z0 as f32 + 0.5);

        let nx = self.mac.nx;
        let ny = self.mac.ny;

        let v000 = self.mac.densities.get(x0 + nx * (y0 + ny * z0)).copied().unwrap_or(0.0);
        let v001 = self.mac.densities.get(x0 + nx * (y0 + ny * z1)).copied().unwrap_or(0.0);
        let v010 = self.mac.densities.get(x0 + nx * (y1 + ny * z0)).copied().unwrap_or(0.0);
        let v011 = self.mac.densities.get(x0 + nx * (y1 + ny * z1)).copied().unwrap_or(0.0);
        let v100 = self.mac.densities.get(x1 + nx * (y0 + ny * z0)).copied().unwrap_or(0.0);
        let v101 = self.mac.densities.get(x1 + nx * (y0 + ny * z1)).copied().unwrap_or(0.0);
        let v110 = self.mac.densities.get(x1 + nx * (y1 + ny * z0)).copied().unwrap_or(0.0);
        let v111 = self.mac.densities.get(x1 + nx * (y1 + ny * z1)).copied().unwrap_or(0.0);

        v000 * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
            + v001 * (1.0 - dx) * (1.0 - dy) * dz
            + v010 * (1.0 - dx) * dy * (1.0 - dz)
            + v011 * (1.0 - dx) * dy * dz
            + v100 * dx * (1.0 - dy) * (1.0 - dz)
            + v101 * dx * (1.0 - dy) * dz
            + v110 * dx * dy * (1.0 - dz)
            + v111 * dx * dy * dz
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
}

impl Default for FlipFluid3DParams {
    fn default() -> Self {
        Self {
            num_substeps: 2,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            flip_ratio: 0.1,
            num_pressure_iters: 100,
            num_particle_iters: 2,
            over_relaxation: 1.9,
            compensate_drift: true,
        }
    }
}

impl Fluid<3> for FlipFluid3D {
    type Params = FlipFluid3DParams;

    fn step(&mut self, dt: f32, params: &Self::Params, obstacles: &ObstacleSet<3>) {
        let sdt = dt / params.num_substeps as f32;

        self.set_obstacles(obstacles, dt);

        for _step in 0..params.num_substeps {
            self.integrate_particles(sdt, params.gravity, obstacles);

            if params.compensate_drift {
                self.separate_particles(params.num_particle_iters);
            }

            self.handle_particle_collisions();

            self.particle_to_grid();

            self.mac.assign_uvw_star();

            self.update_particle_density();

            self.solve_pressure(params.num_pressure_iters, sdt, params.over_relaxation, params.compensate_drift);

            self.grid_to_particle(params.flip_ratio);
        }
    }

    fn particle_radius(&self) -> f32 {
        self.particle_radius
    }
}
