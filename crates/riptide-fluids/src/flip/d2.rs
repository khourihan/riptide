use std::f32::consts::PI;

use glam::{UVec2, Vec2};
use ndarray::{azip, Array0, Array1, Array2, ArrayView2, Axis};

use crate::{obstacle::{Obstacle, ObstacleSet}, Fluid};

use super::CellType;

#[derive(Debug, Clone)]
pub struct FlipFluid2D {
    /// The density of the fluid, in kg/m³.
    ///
    /// Air in `0` kg/m³ and water is `1000` kg/m³.
    density: f32,
    size: Vec2,
    grid_size: UVec2,
    /// Cell size.
    pub spacing: f32,

    rest_density: f32,
    particle_radius: f32,
    particle_spacing: f32,
    n_particles: usize,
    particle_resolution: UVec2,

    /// Grid velocities.
    u: Array2<f32>,
    v: Array2<f32>,
    /// Grid velocity deltas.
    weight_u: Array2<f32>,
    weight_v: Array2<f32>,
    /// Previous grid velocities.
    u_star: Array2<f32>,
    v_star: Array2<f32>,
    /// Pressure of the grid.
    pressure: Array2<f32>,
    /// Solid grid cells. `true` for completely solid and `false` for not solid.
    solid: Array2<bool>,
    /// Grid cell types (`Fluid`, `Solid` or `Air`).
    cell_type: Array2<CellType>,
    /// Grid densities.
    pub densities: Array2<f32>,

    cell_particle_count: Array1<usize>,
    first_cell_particle: Array1<usize>,

    /// Particle positions.
    pub positions: Array1<Vec2>,
    /// Particle velocities.
    velocities: Array1<Vec2>,
    /// Fluid roughness approximation per particle.
    roughness: Array1<f32>,
    cell_particle_indices: Array1<usize>,
}

impl FlipFluid2D {
    pub fn new(
        density: f32,
        size: Vec2,
        spacing: f32,
        particle_radius: f32,
    ) -> Self {
        let grid_size = (size / spacing).floor().as_uvec2() + 1;
        let h = f32::max(size.x / grid_size.x as f32, size.y / grid_size.y as f32);

        let u = Array2::from_elem((grid_size.x as usize + 1, grid_size.y as usize), 0.0);
        let v = Array2::from_elem((grid_size.x as usize, grid_size.y as usize + 1), 0.0);
        let weight_u = Array2::from_elem((grid_size.x as usize + 1, grid_size.y as usize), 0.0);
        let weight_v = Array2::from_elem((grid_size.x as usize, grid_size.y as usize + 1), 0.0);
        let u_star = Array2::from_elem((grid_size.x as usize + 1, grid_size.y as usize), 0.0);
        let v_star = Array2::from_elem((grid_size.x as usize, grid_size.y as usize + 1), 0.0);
        let pressure = Array2::from_elem((grid_size.x as usize, grid_size.y as usize), 0.0);
        let solid = Array2::from_elem((grid_size.x as usize, grid_size.y as usize), false);
        let cell_type = Array2::from_elem((grid_size.x as usize, grid_size.y as usize), CellType::Fluid);
        let densities = Array2::from_elem((grid_size.x as usize, grid_size.y as usize), 0.0);

        let positions = Array1::from_vec(vec![]);
        let velocities = Array1::from_vec(vec![]);
        let roughness = Array1::from_vec(vec![]);

        let particle_spacing = 2.2 * particle_radius;
        let particle_resolution = UVec2::new(
            (size.x / particle_spacing).floor() as u32 + 1,
            (size.y / particle_spacing).floor() as u32 + 1,
        );

        let cell_count = (particle_resolution.x * particle_resolution.y) as usize;
        let cell_particle_count = Array1::from_elem(cell_count, 0);
        let first_cell_particle = Array1::from_elem(cell_count + 1, 0);

        let cell_particle_indices = Array1::from_vec(vec![]);

        Self {
            density,
            size,
            grid_size,
            spacing: h,
            rest_density: 0.0,
            particle_radius,
            particle_spacing,
            n_particles: 0,
            particle_resolution,
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
            cell_particle_count,
            first_cell_particle,
            positions,
            velocities,
            roughness,
            cell_particle_indices,
        }
    }

    pub fn insert_particle(&mut self, pos: Vec2) {
        let _ = self.positions.push(Axis(0), Array0::from_elem((), pos).view());
        let _ = self.velocities.push(Axis(0), Array0::from_elem((), Vec2::ZERO).view());
        let _ = self.roughness.push(Axis(0), Array0::from_elem((), 0.0).view());
        let _ = self.cell_particle_indices.push(Axis(0), Array0::from_elem((), 0).view());
        self.n_particles += 1;
    }

    pub fn set_solid(&mut self, i: usize, j: usize, v: bool) {
        self.solid[(i, j)] = v;
    }

    pub fn iter_positions(&self) -> impl Iterator<Item = &Vec2> {
        self.positions.iter()
    }

    pub fn iter_particles(&self) -> impl Iterator<Item = (&Vec2, &Vec2, &f32)> {
        self.positions.iter().zip(self.velocities.iter()).zip(self.roughness.iter()).map(|((p, v), r)| (p, v, r))
    }

    pub fn size(&self) -> UVec2 {
        self.grid_size
    }

    pub fn bounds(&self) -> (Vec2, Vec2) {
        (
            Vec2::splat(self.spacing + self.particle_radius),
            (self.grid_size - 1).as_vec2() * self.spacing - self.particle_radius,
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

        self.cell_particle_count.fill(0);
        self.first_cell_particle.fill(0);

        for p in self.positions.iter() {
            let pi = (p / self.particle_spacing).floor().as_uvec2()
                .clamp(UVec2::ZERO, self.particle_resolution - 1);
            let cell_nr = pi.x * self.particle_resolution.y + pi.y;
            self.cell_particle_count[cell_nr as usize] += 1;
        }

        let mut first = 0;

        for (count, first_cell) in self.cell_particle_count.iter().zip(self.first_cell_particle.iter_mut()) {
            first += count;
            *first_cell = first;
        }

        self.first_cell_particle[(self.particle_resolution.x * self.particle_resolution.y) as usize] = first;

        for (i, p) in self.positions.iter().enumerate() {
            let pi = (p / self.particle_spacing).floor().as_uvec2()
                .clamp(UVec2::ZERO, self.particle_resolution - 1);
            let cell_nr = (pi.x * self.particle_resolution.y + pi.y) as usize;
            self.first_cell_particle[cell_nr] -= 1;
            self.cell_particle_indices[self.first_cell_particle[cell_nr]] = i;
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
            let pi = p.clamp(Vec2::splat(h), (self.grid_size - 1).as_vec2() * h);

            let p0 = ((pi - h2) * h1).floor().as_uvec2();
            let t = ((pi - h2) - p0.as_vec2() * h) * h1;
            let p1 = (p0 + 1).min(self.grid_size - 2);
            let s = 1.0 - t;

            if p0.x < self.grid_size.x && p0.y < self.grid_size.y {
                self.densities[(p0.x as usize, p0.y as usize)] += s.x * s.y;
            }

            if p1.x < self.grid_size.x && p0.y < self.grid_size.y {
                self.densities[(p1.x as usize, p0.y as usize)] += t.x * s.y;
            }

            if p1.x < self.grid_size.x && p1.y < self.grid_size.y {
                self.densities[(p1.x as usize, p1.y as usize)] += t.x * t.y;
            }

            if p0.x < self.grid_size.x && p1.y < self.grid_size.y {
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
        self.u.fill(0.0);
        self.v.fill(0.0);
        self.weight_u.fill(0.0);
        self.weight_v.fill(0.0);

        let nx = self.grid_size.x as usize;
        let ny = self.grid_size.y as usize;

        let h1 = self.spacing.recip();
        let h = 2.0 * self.spacing;
        let h2 = h * h;
        let h4 = h2 * h2;
        let coeff = 315.0 / (64.0 * PI * h4 * h4 * h);

        azip!((cell_type in &mut self.cell_type, &s in &self.solid) {
            *cell_type = if s { CellType::Solid } else { CellType::Air };
        });

        for i in 0..self.n_particles {
            let pos = self.positions[i];
            let vel = self.velocities[i];

            let pi = (pos * h1).floor().as_uvec2().clamp(UVec2::ZERO, self.grid_size - 1);

            if self.cell_type[(pi.x as usize, pi.y as usize)] == CellType::Air {
                self.cell_type[(pi.x as usize, pi.y as usize)] = CellType::Fluid;
            }

            if pi.x >= 2 && pi.y >= 2 && pi.x < self.grid_size.x - 3 && pi.y < self.grid_size.y - 3 {
                for j in pi.y as usize - 2..=pi.y as usize + 3 {
                    for i in pi.x as usize - 2..=pi.x as usize + 3 {
                        let rx = pos.x - i as f32 * self.spacing;
                        let ry = pos.y - j as f32 * self.spacing;

                        let x_diff = h2 - ry * ry - (rx + 0.5 * self.spacing) * (rx + 0.5 * self.spacing);
                        let y_diff = h2 - rx * rx - (ry + 0.5 * self.spacing) * (ry + 0.5 * self.spacing);

                        if x_diff >= 0.0 {
                            let u_weight_1 = coeff * x_diff * x_diff * x_diff;
                            self.u[(i, j)] += u_weight_1 * vel.x;
                            self.weight_u[(i, j)] += u_weight_1;
                        }

                        if y_diff >= 0.0 {
                            let v_weight_1 = coeff * y_diff * y_diff * y_diff;
                            self.v[(i, j)] += v_weight_1 * vel.y;
                            self.weight_v[(i, j)] += v_weight_1;
                        }
                    }
                }
            } else {
                for j in pi.y.max(2) as usize - 2..=pi.y as usize + 3 {
                    for i in pi.x.max(2) as usize - 2..=pi.x as usize + 3 {
                        let rx = pos.x - i as f32 * self.spacing;
                        let ry = pos.y - j as f32 * self.spacing;

                        if i <= nx && j < ny {
                            let x_diff = h2 - ry * ry - (rx + 0.5 * self.spacing) * (rx + 0.5 * self.spacing);

                            if x_diff >= 0.0 {
                                let u_weight_1 = coeff * x_diff * x_diff * x_diff;
                                self.u[(i, j)] += u_weight_1 * vel.x;
                                self.weight_u[(i, j)] += u_weight_1;
                            }
                        }

                        if i < nx && j <= ny {
                            let y_diff = h2 - rx * rx - (ry + 0.5 * self.spacing) * (ry + 0.5 * self.spacing);

                            if y_diff >= 0.0 {
                                let v_weight_1 = coeff * y_diff * y_diff * y_diff;
                                self.v[(i, j)] += v_weight_1 * vel.y;
                                self.weight_v[(i, j)] += v_weight_1;
                            }
                        }
                    }
                }
            }
        }

        let mut visited_u = vec![false; (self.grid_size.y * (self.grid_size.x + 1)) as usize];
        let mut visited_v = vec![false; (self.grid_size.x * (self.grid_size.y + 1)) as usize];

        for j in 0..ny {
            for i in 0..nx {
                let u_idx = (nx + 1) * j + i;
                let v_idx = nx * j + i;

                let u_weight = self.weight_u[(i, j)];
                let v_weight = self.weight_v[(i, j)];

                if u_weight != 0.0 {
                    self.u[(i, j)] /= u_weight;
                    visited_u[u_idx] = true;
                }

                if v_weight != 0.0 {
                    self.v[(i, j)] /= v_weight;
                    visited_v[v_idx] = true;
                }
            }
        }

        for j in 0..ny {
            let u_idx = (nx + 1) * j + nx;
            let u_weight = self.weight_u[(nx, j)];

            if u_weight != 0.0 {
                self.u[(nx, j)] /= u_weight;
                visited_u[u_idx] = true;
            }
        }

        for i in 0..nx {
            let v_idx = nx * ny + i;
            let v_weight = self.weight_v[(i, ny)];

            if v_weight != 0.0 {
                self.v[(i, ny)] /= v_weight;
                visited_v[v_idx] = true;
            }
        }

        for j in 0..ny {
            for i in 0..nx {
                let u_idx = (nx + 1) * j + i;
                let v_idx = nx * j + i;

                if !visited_u[u_idx] {
                    let mut u_counter: u8 = 0;

                    let u_left = if i > 0 && visited_u[u_idx - 1] {
                        u_counter += 1;
                        self.u[(i - 1, j)]
                    } else {
                        0.0
                    };

                    let u_right = if i < nx - 1 && visited_u[u_idx + 1] {
                        u_counter += 1;
                        self.u[(i + 1, j)]
                    } else {
                        0.0
                    };

                    let u_down = if j > 0 && visited_u[u_idx - (nx + 1)] {
                        u_counter += 1;
                        self.u[(i, j - 1)]
                    } else {
                        0.0
                    };

                    let u_up = if j < ny - 1 && visited_u[u_idx + (nx + 1)] {
                        u_counter += 1;
                        self.u[(i, j + 1)]
                    } else {
                        0.0
                    };

                    if u_counter != 0 {
                        self.u[(i, j)] = (u_left + u_right + u_down + u_up) / u_counter as f32;
                    }
                }

                if !visited_v[v_idx] {
                    let mut v_counter: u8 = 0;

                    let v_left = if i > 0 && visited_v[v_idx - 1] {
                        v_counter += 1;
                        self.v[(i - 1, j)]
                    } else {
                        0.0
                    };

                    let v_right = if i < nx - 1 && visited_v[v_idx + 1] {
                        v_counter += 1;
                        self.v[(i + 1, j)]
                    } else {
                        0.0
                    };

                    let v_down = if j > 0 && visited_v[v_idx - nx] {
                        v_counter += 1;
                        self.v[(i, j - 1)]
                    } else {
                        0.0
                    };

                    let v_up = if j < ny - 1 && visited_v[v_idx + nx] {
                        v_counter += 1;
                        self.v[(i, j + 1)]
                    } else {
                        0.0
                    };

                    if v_counter != 0 {
                        self.v[(i, j)] = (v_left + v_right + v_down + v_up) / v_counter as f32;
                    }
                }
            }
        }

        for j in 0..ny {
            let u_idx = (nx + 1) * j + nx;

            if !visited_u[u_idx] {
                let mut u_counter: u8 = 0;

                let u_left = if visited_u[u_idx - 1] {
                    u_counter += 1;
                    self.u[(nx - 1, j)]
                } else {
                    0.0
                };

                let u_down = if j > 0 && visited_u[u_idx - (nx + 1)] {
                    u_counter += 1;
                    self.u[(nx, j - 1)]
                } else {
                    0.0
                };

                let u_up = if j < ny - 1 && visited_u[u_idx + (nx + 1)] {
                    u_counter += 1;
                    self.u[(nx, j + 1)]
                } else {
                    0.0
                };

                if u_counter != 0 {
                    self.u[(nx, j)] = (u_left + u_down + u_up) / u_counter as f32;
                }
            }
        }

        for i in 0..nx {
            let v_idx = nx * ny + i;

            if !visited_v[v_idx] {
                let mut v_counter: u8 = 0;

                let v_left = if i > 0 && visited_v[v_idx - 1] {
                    v_counter += 1;
                    self.v[(i - 1, ny)]
                } else {
                    0.0
                };

                let v_right = if i < nx - 1 && visited_v[v_idx + 1] {
                    v_counter += 1;
                    self.v[(i + 1, ny)]
                } else {
                    0.0
                };

                let v_down = if visited_v[v_idx - nx] {
                    v_counter += 1;
                    self.v[(i, ny - 1)]
                } else {
                    0.0
                };

                if v_counter != 0 {
                    self.v[(i, ny)] = (v_left + v_right + v_down) / v_counter as f32;
                }
            }
        }
    }

    fn transfer_velocities_to_particles(&mut self, alpha: f32) {
        let nx = self.grid_size.x;
        let ny = self.grid_size.y;
        let h1 = self.spacing.recip();

        for i in 0..self.n_particles {
            let p0 = self.positions[i];
            let v0 = self.velocities[i];

            let pi = (p0 * h1).floor().as_uvec2().clamp(UVec2::ZERO, self.grid_size - 1);

            let interp_u_star = Vec2::new(
                get_bilerp_x(self.u_star.view(), p0, self.spacing, self.grid_size, self.size),
                get_bilerp_y(self.v_star.view(), p0, self.spacing, self.grid_size, self.size),
            );

            let interp_u_n1 = Vec2::new(
                get_bilerp_x(self.u.view(), p0, self.spacing, self.grid_size, self.size),
                get_bilerp_y(self.v.view(), p0, self.spacing, self.grid_size, self.size),
            );

            let u_update = if pi.x == 0 || pi.x == nx - 1 || pi.y == 0 || pi.y == ny - 1 {
                interp_u_n1 + (v0 - interp_u_star) * (1.0 - f32::min(1.0, 2.0 * alpha))
            } else {
                interp_u_n1 + (v0 - interp_u_star) * (1.0 - alpha)
            };

            self.velocities[i] = u_update;
        }
    }

    fn solve_incompressibility(&mut self, num_iters: usize, dt: f32, over_relaxation: f32, compensate_drift: bool) {
        self.pressure.fill(0.0);
        self.u_star.assign(&self.u);
        self.v_star.assign(&self.v);

        let cp = self.density * self.spacing / dt;

        for _iter in 0..num_iters {
            for i in 1..self.grid_size.x as usize - 1 {
                for j in 1..self.grid_size.y as usize - 1 {
                    if self.cell_type[(i, j)] != CellType::Fluid {
                        continue;
                    }

                    let center = (i, j);
                    let left = (i - 1, j);
                    let right = (i + 1, j);
                    let bottom = (i, j - 1);
                    let top = (i, j + 1);

                    let sx0 = if self.solid[left] { 0.0 } else { 1.0 };
                    let sx1 = if self.solid[right] { 0.0 } else { 1.0 };
                    let sy0 = if self.solid[bottom] { 0.0 } else { 1.0 };
                    let sy1 = if self.solid[top] { 0.0 } else { 1.0 };
                    let s = sx0 + sx1 + sy0 + sy1;

                    if s == 0.0 {
                        continue;
                    }

                    let mut div = self.u[right] - self.u[center]
                        + self.v[top] - self.v[center];

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

                    self.u[center] -= sx0 * p;
                    self.u[right] += sx1 * p;
                    self.v[center] -= sy0 * p;
                    self.v[top] += sy1 * p;
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
            let pi = (p * h1).floor().as_uvec2().clamp(UVec2::ONE, self.grid_size - 1);

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
        for i in 1..self.grid_size.x as usize - 1 {
            for j in 1..self.grid_size.y as usize - 1 {
                self.solid[(i, j)] = false;
                let p = Vec2::new(i as f32 + 0.5, j as f32 + 0.5) * self.spacing;
                let sdf = obstacles.sdf(p.into());

                if sdf.distance < 0.0 {
                    // TODO: add velocity of obstacle to this.
                    let v = -sdf.distance * Vec2::from(sdf.gradient) / dt;
                    self.solid[(i, j)] = true;
                    self.u[(i, j)] = v.x;
                    self.v[(i, j)] = v.y;
                    self.u[(i + 1, j)] = v.x;
                    self.v[(i, j + 1)] = v.y;
                }
            }
        }
    }

    pub fn sample_density(&self, p: Vec2) -> f32 {
        let h1 = self.spacing.recip();

        let x0 = (p.x * h1).floor() as usize;
        let x1 = if (p.x * h1).fract() > 0.5 { x0 + 1 } else { x0 - 1 };
        let y0 = (p.y * h1).floor() as usize;
        let y1 = if (p.y * h1).fract() > 0.5 { y0 + 1 } else { y0 - 1 };

        let dx = (p.x * h1) - (x0 as f32 + 0.5);
        let dy = (p.y * h1) - (y0 as f32 + 0.5);

        let v00 = self.densities.get((x0, y0)).copied().unwrap_or(0.0);
        let v01 = self.densities.get((x0, y1)).copied().unwrap_or(0.0);
        let v10 = self.densities.get((x1, y0)).copied().unwrap_or(0.0);
        let v11 = self.densities.get((x1, y1)).copied().unwrap_or(0.0);

        v00 * (1.0 - dx) * (1.0 - dy)
            + v01 * (1.0 - dx) * dy
            + v10 * dx * (1.0 - dy)
            + v11 * dx * dy
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
            flip_ratio: 0.1,
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

    fn particle_radius(&self) -> f32 {
        self.particle_radius
    }
}

fn get_bilerp_x(
    vals: ArrayView2<f32>,
    p: Vec2,
    spacing: f32,
    grid_size: UVec2,
    size: Vec2,
) -> f32 {
    let pi = (p / spacing).floor().as_uvec2().clamp(UVec2::ZERO, grid_size - 1);

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
    grid_size: UVec2,
    size: Vec2,
) -> f32 {
    let pi = (p / spacing).floor().as_uvec2().clamp(UVec2::ZERO, grid_size - 1);

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
