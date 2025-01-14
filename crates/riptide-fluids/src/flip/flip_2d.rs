use std::f32::consts::PI;

use glam::{UVec2, Vec2};
use ndarray::azip;

use crate::{obstacle::{Obstacle, ObstacleSet}, Fluid};

use super::{mac_2d::MacGrid2D, CellType};

#[derive(Debug, Clone)]
pub struct FlipFluid2D {
    /// MAC grid.
    mac: MacGrid2D,
    /// The density of the fluid, in kg/m³.
    ///
    /// Air is `1` kg/m³ (approximated as `0` kg/m³) and water is `1000` kg/m³.
    density: f32,

    rest_density: f32,
    /// Radius of particles.
    particle_radius: f32,
    /// Cell size of particle grid.
    particle_spacing: f32,
    /// Total number of particles.
    n_particles: usize,
    /// Resolution of particle grid.
    particle_resolution: UVec2,

    /// Number of particles per cell of particle grid.
    cell_particle_count: Vec<usize>,
    first_cell_particle: Vec<usize>,

    /// Particle positions.
    pub positions: Vec<Vec2>,
    /// Particle velocities.
    velocities: Vec<Vec2>,
    cell_particle_indices: Vec<usize>,
}

impl FlipFluid2D {
    pub fn new(
        density: f32,
        size: Vec2,
        spacing: f32,
        particle_radius: f32,
    ) -> Self {
        let positions = vec![];
        let velocities = vec![];

        let particle_spacing = 2.2 * particle_radius;
        let particle_resolution = UVec2::new(
            (size.x / particle_spacing).floor() as u32 + 1,
            (size.y / particle_spacing).floor() as u32 + 1,
        );

        let cell_count = (particle_resolution.x * particle_resolution.y) as usize;
        let cell_particle_count = vec![0; cell_count];
        let first_cell_particle = vec![0; cell_count + 1];

        let cell_particle_indices = vec![];

        Self {
            mac: MacGrid2D::new(size, spacing),
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

    pub fn insert_particle(&mut self, pos: Vec2) {
        self.positions.push(pos);
        self.velocities.push(Vec2::ZERO);
        self.cell_particle_indices.push(0);
        self.n_particles += 1;
    }

    pub fn set_solid(&mut self, i: usize, j: usize, v: bool) {
        self.mac.solid[(i, j)] = v;
    }

    pub fn iter_positions(&self) -> impl Iterator<Item = &Vec2> {
        self.positions.iter()
    }

    pub fn iter_particles(&self) -> impl Iterator<Item = (&Vec2, &Vec2)> {
        self.positions.iter().zip(self.velocities.iter())
    }

    pub fn size(&self) -> UVec2 {
        self.mac.grid_size
    }

    pub fn spacing(&self) -> f32 {
        self.mac.spacing
    }

    fn integrate_particles(&mut self, dt: f32, gravity: Vec2, obstacles: &ObstacleSet<2>) {
        azip!((p in &self.positions, v in &mut self.velocities) {
            let sdf = obstacles.sdf((*p).into());
            if sdf.distance < 0.0 {
                *v = -sdf.distance * Vec2::from(sdf.gradient) / dt;
            }
        });

        self.velocities.iter_mut().for_each(|v| *v += dt * gravity);

        azip!((p in &mut self.positions, vel in &self.velocities) {
            *p += vel * dt;
        });
    }

    fn separate_particles(&mut self, num_iters: usize) {
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
                        }
                    }
                }
            }
        }
    }

    fn handle_particle_collisions(&mut self) {
        let min = Vec2::splat(self.mac.spacing + self.particle_radius);
        let max = self.mac.size - min;

        azip!((p in &mut self.positions, v in &mut self.velocities) {
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
        let h = self.mac.spacing;
        let h1 = self.mac.inv_spacing;
        let h2 = 0.5 * h;

        self.mac.densities.fill(0.0);

        for p in self.positions.iter() {
            let pi = p.clamp(Vec2::splat(h), (self.mac.grid_size - 1).as_vec2() * h);

            let p0 = ((pi - h2) * h1).floor().as_uvec2();
            let t = ((pi - h2) - p0.as_vec2() * h) * h1;
            let p1 = (p0 + 1).min(self.mac.grid_size - 2);
            let s = 1.0 - t;

            if p0.x < self.mac.grid_size.x && p0.y < self.mac.grid_size.y {
                self.mac.densities[(p0.x as usize, p0.y as usize)] += s.x * s.y;
            }

            if p1.x < self.mac.grid_size.x && p0.y < self.mac.grid_size.y {
                self.mac.densities[(p1.x as usize, p0.y as usize)] += t.x * s.y;
            }

            if p1.x < self.mac.grid_size.x && p1.y < self.mac.grid_size.y {
                self.mac.densities[(p1.x as usize, p1.y as usize)] += t.x * t.y;
            }

            if p0.x < self.mac.grid_size.x && p1.y < self.mac.grid_size.y {
                self.mac.densities[(p0.x as usize, p1.y as usize)] += s.x * t.y;
            }
        }

        if self.rest_density == 0.0 {
            let mut sum: f32 = 0.0;
            let mut num_fluid_cells: usize = 0;

            azip!((&cell_type in &self.mac.cell_type, &density in &self.mac.densities) {
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

    fn particle_to_grid(&mut self) {
        self.mac.u.fill(0.0);
        self.mac.v.fill(0.0);
        self.mac.weight_u.fill(0.0);
        self.mac.weight_v.fill(0.0);

        let nx = self.mac.nx;
        let ny = self.mac.ny;

        let h_half = 0.5 * self.mac.spacing;
        let h1 = self.mac.inv_spacing;
        let h = 2.0 * self.mac.spacing;
        let h2 = h * h;
        let h4 = h2 * h2;
        let coeff = 315.0 / (64.0 * PI * h4 * h4 * h);

        azip!((cell_type in &mut self.mac.cell_type, &s in &self.mac.solid) {
            *cell_type = if s { CellType::Solid } else { CellType::Air };
        });

        for i in 0..self.n_particles {
            let pos = self.positions[i];
            let vel = self.velocities[i];

            let pi = (pos * h1).floor().as_uvec2().clamp(UVec2::ZERO, self.mac.grid_size - 1);

            if self.mac.cell_type[(pi.x as usize, pi.y as usize)] == CellType::Air {
                self.mac.cell_type[(pi.x as usize, pi.y as usize)] = CellType::Fluid;
            }

            if pi.x >= 2 && pi.y >= 2 && pi.x < self.mac.grid_size.x - 3 && pi.y < self.mac.grid_size.y - 3 {
                for j in pi.y as usize - 2..=pi.y as usize + 3 {
                    for i in pi.x as usize - 2..=pi.x as usize + 3 {
                        let rx = pos.x - i as f32 * self.mac.spacing;
                        let ry = pos.y - j as f32 * self.mac.spacing;

                        let x_diff = h2 - ry * ry - (rx + h_half) * (rx + h_half);
                        let y_diff = h2 - rx * rx - (ry + h_half) * (ry + h_half);

                        if x_diff >= 0.0 {
                            let u_weight_1 = coeff * x_diff * x_diff * x_diff;
                            self.mac.u[(i, j)] += u_weight_1 * vel.x;
                            self.mac.weight_u[(i, j)] += u_weight_1;
                        }

                        if y_diff >= 0.0 {
                            let v_weight_1 = coeff * y_diff * y_diff * y_diff;
                            self.mac.v[(i, j)] += v_weight_1 * vel.y;
                            self.mac.weight_v[(i, j)] += v_weight_1;
                        }
                    }
                }
            } else {
                for j in pi.y.max(2) as usize - 2..=pi.y as usize + 3 {
                    for i in pi.x.max(2) as usize - 2..=pi.x as usize + 3 {
                        let rx = pos.x - i as f32 * self.mac.spacing;
                        let ry = pos.y - j as f32 * self.mac.spacing;

                        if i <= nx && j < ny {
                            let x_diff = h2 - ry * ry - (rx + 0.5 * self.mac.spacing) * (rx + 0.5 * self.mac.spacing);

                            if x_diff >= 0.0 {
                                let u_weight_1 = coeff * x_diff * x_diff * x_diff;
                                self.mac.u[(i, j)] += u_weight_1 * vel.x;
                                self.mac.weight_u[(i, j)] += u_weight_1;
                            }
                        }

                        if i < nx && j <= ny {
                            let y_diff = h2 - rx * rx - (ry + 0.5 * self.mac.spacing) * (ry + 0.5 * self.mac.spacing);

                            if y_diff >= 0.0 {
                                let v_weight_1 = coeff * y_diff * y_diff * y_diff;
                                self.mac.v[(i, j)] += v_weight_1 * vel.y;
                                self.mac.weight_v[(i, j)] += v_weight_1;
                            }
                        }
                    }
                }
            }
        }

        let mut visited_u = vec![false; self.mac.ny * (self.mac.nx + 1)];
        let mut visited_v = vec![false; self.mac.nx * (self.mac.ny + 1)];

        for j in 0..ny {
            for i in 0..nx {
                let u_idx = (nx + 1) * j + i;
                let v_idx = nx * j + i;

                let u_weight = self.mac.weight_u[(i, j)];
                let v_weight = self.mac.weight_v[(i, j)];

                if u_weight != 0.0 {
                    self.mac.u[(i, j)] /= u_weight;
                    visited_u[u_idx] = true;
                }

                if v_weight != 0.0 {
                    self.mac.v[(i, j)] /= v_weight;
                    visited_v[v_idx] = true;
                }
            }
        }

        for j in 0..ny {
            let u_idx = (nx + 1) * j + nx;
            let u_weight = self.mac.weight_u[(nx, j)];

            if u_weight != 0.0 {
                self.mac.u[(nx, j)] /= u_weight;
                visited_u[u_idx] = true;
            }
        }

        for i in 0..nx {
            let v_idx = nx * ny + i;
            let v_weight = self.mac.weight_v[(i, ny)];

            if v_weight != 0.0 {
                self.mac.v[(i, ny)] /= v_weight;
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
                        self.mac.u[(i - 1, j)]
                    } else {
                        0.0
                    };

                    let u_right = if i < nx - 1 && visited_u[u_idx + 1] {
                        u_counter += 1;
                        self.mac.u[(i + 1, j)]
                    } else {
                        0.0
                    };

                    let u_down = if j > 0 && visited_u[u_idx - (nx + 1)] {
                        u_counter += 1;
                        self.mac.u[(i, j - 1)]
                    } else {
                        0.0
                    };

                    let u_up = if j < ny - 1 && visited_u[u_idx + (nx + 1)] {
                        u_counter += 1;
                        self.mac.u[(i, j + 1)]
                    } else {
                        0.0
                    };

                    if u_counter != 0 {
                        self.mac.u[(i, j)] = (u_left + u_right + u_down + u_up) / u_counter as f32;
                    }
                }

                if !visited_v[v_idx] {
                    let mut v_counter: u8 = 0;

                    let v_left = if i > 0 && visited_v[v_idx - 1] {
                        v_counter += 1;
                        self.mac.v[(i - 1, j)]
                    } else {
                        0.0
                    };

                    let v_right = if i < nx - 1 && visited_v[v_idx + 1] {
                        v_counter += 1;
                        self.mac.v[(i + 1, j)]
                    } else {
                        0.0
                    };

                    let v_down = if j > 0 && visited_v[v_idx - nx] {
                        v_counter += 1;
                        self.mac.v[(i, j - 1)]
                    } else {
                        0.0
                    };

                    let v_up = if j < ny - 1 && visited_v[v_idx + nx] {
                        v_counter += 1;
                        self.mac.v[(i, j + 1)]
                    } else {
                        0.0
                    };

                    if v_counter != 0 {
                        self.mac.v[(i, j)] = (v_left + v_right + v_down + v_up) / v_counter as f32;
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
                    self.mac.u[(nx - 1, j)]
                } else {
                    0.0
                };

                let u_down = if j > 0 && visited_u[u_idx - (nx + 1)] {
                    u_counter += 1;
                    self.mac.u[(nx, j - 1)]
                } else {
                    0.0
                };

                let u_up = if j < ny - 1 && visited_u[u_idx + (nx + 1)] {
                    u_counter += 1;
                    self.mac.u[(nx, j + 1)]
                } else {
                    0.0
                };

                if u_counter != 0 {
                    self.mac.u[(nx, j)] = (u_left + u_down + u_up) / u_counter as f32;
                }
            }
        }

        for i in 0..nx {
            let v_idx = nx * ny + i;

            if !visited_v[v_idx] {
                let mut v_counter: u8 = 0;

                let v_left = if i > 0 && visited_v[v_idx - 1] {
                    v_counter += 1;
                    self.mac.v[(i - 1, ny)]
                } else {
                    0.0
                };

                let v_right = if i < nx - 1 && visited_v[v_idx + 1] {
                    v_counter += 1;
                    self.mac.v[(i + 1, ny)]
                } else {
                    0.0
                };

                let v_down = if visited_v[v_idx - nx] {
                    v_counter += 1;
                    self.mac.v[(i, ny - 1)]
                } else {
                    0.0
                };

                if v_counter != 0 {
                    self.mac.v[(i, ny)] = (v_left + v_right + v_down) / v_counter as f32;
                }
            }
        }
    }

    fn grid_to_particle(&mut self, alpha: f32) {
        let nx = self.mac.grid_size.x;
        let ny = self.mac.grid_size.y;
        let h1 = self.mac.inv_spacing;

        for i in 0..self.n_particles {
            let p0 = self.positions[i];
            let v0 = self.velocities[i];

            let pi = (p0 * h1).floor().as_uvec2().clamp(UVec2::ZERO, self.mac.grid_size - 1);

            let interp_u_star = Vec2::new(
                self.mac.get_bilerp_u_star(p0),
                self.mac.get_bilerp_v_star(p0),
            );

            let interp_u_n1 = Vec2::new(
                self.mac.get_bilerp_u(p0),
                self.mac.get_bilerp_v(p0),
            );

            let u_update = if pi.x == 0 || pi.x == nx - 1 || pi.y == 0 || pi.y == ny - 1 {
                interp_u_n1 + (v0 - interp_u_star) * (1.0 - f32::min(1.0, 2.0 * alpha))
            } else {
                interp_u_n1 + (v0 - interp_u_star) * (1.0 - alpha)
            };

            self.velocities[i] = u_update;
        }
    }

    fn solve_pressure(&mut self, num_iters: usize, dt: f32, over_relaxation: f32, compensate_drift: bool) {
        self.mac.pressure.fill(0.0);

        let nx = self.mac.nx;
        let ny = self.mac.ny;
        let cp = self.density * self.mac.spacing / dt;

        for _iter in 0..num_iters {
            for i in 1..nx - 1 {
                for j in 1..ny - 1 {
                    if self.mac.cell_type[(i, j)] != CellType::Fluid {
                        continue;
                    }

                    let sx0 = if self.mac.solid[(i - 1, j)] { 0.0 } else { 1.0 };
                    let sx1 = if self.mac.solid[(i + 1, j)] { 0.0 } else { 1.0 };
                    let sy0 = if self.mac.solid[(i, j - 1)] { 0.0 } else { 1.0 };
                    let sy1 = if self.mac.solid[(i, j + 1)] { 0.0 } else { 1.0 };
                    let s = sx0 + sx1 + sy0 + sy1;

                    if s == 0.0 {
                        continue;
                    }

                    let mut div = self.mac.u[(i + 1, j)] - self.mac.u[(i, j)]
                        + self.mac.v[(i, j + 1)] - self.mac.v[(i, j)];

                    if self.rest_density > 0.0 && compensate_drift {
                        let stiffness = 1.0;
                        let compression = self.mac.densities[(i, j)] - self.rest_density;
                        if compression > 0.0 {
                            div -= stiffness * compression;
                        }
                    }

                    let mut p = -div / s;
                    p *= over_relaxation;
                    self.mac.pressure[(i, j)] += cp * p;

                    self.mac.u[(i, j)] -= sx0 * p;
                    self.mac.u[(i + 1, j)] += sx1 * p;
                    self.mac.v[(i, j)] -= sy0 * p;
                    self.mac.v[(i, j + 1)] += sy1 * p;
                }
            }
        }
    }

    fn set_obstacles(&mut self, obstacles: &ObstacleSet<2>, dt: f32) {
        for i in 1..self.mac.nx - 1 {
            for j in 1..self.mac.ny - 1 {
                self.mac.solid[(i, j)] = false;
                let p = Vec2::new(i as f32 + 0.5, j as f32 + 0.5) * self.mac.spacing;
                let sdf = obstacles.sdf(p.into());

                if sdf.distance < 0.0 {
                    let v = -sdf.distance * Vec2::from(sdf.gradient) / dt;
                    self.mac.solid[(i, j)] = true;
                    self.mac.u[(i, j)] = v.x;
                    self.mac.v[(i, j)] = v.y;
                    self.mac.u[(i + 1, j)] = v.x;
                    self.mac.v[(i, j + 1)] = v.y;
                }
            }
        }
    }

    pub fn sample_density(&self, p: Vec2) -> f32 {
        let h1 = self.mac.inv_spacing;

        let x0 = (p.x * h1).floor() as usize;
        let x1 = if (p.x * h1).fract() > 0.5 { x0.min(self.mac.nx - 2) + 1 } else { x0.max(1) - 1 };
        let y0 = (p.y * h1).floor() as usize;
        let y1 = if (p.y * h1).fract() > 0.5 { y0.min(self.mac.ny - 2) + 1 } else { y0.max(1) - 1 };

        let dx = (p.x * h1) - (x0 as f32 + 0.5);
        let dy = (p.y * h1) - (y0 as f32 + 0.5);

        let v00 = self.mac.densities.get((x0, y0)).copied().unwrap_or(0.0);
        let v01 = self.mac.densities.get((x0, y1)).copied().unwrap_or(0.0);
        let v10 = self.mac.densities.get((x1, y0)).copied().unwrap_or(0.0);
        let v11 = self.mac.densities.get((x1, y1)).copied().unwrap_or(0.0);

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
        }
    }
}

impl Fluid<2> for FlipFluid2D {
    type Params = FlipFluid2DParams;

    fn step(&mut self, dt: f32, params: &Self::Params, obstacles: &ObstacleSet<2>) {
        let sdt = dt / params.num_substeps as f32;

        self.set_obstacles(obstacles, dt);

        for _step in 0..params.num_substeps {
            self.integrate_particles(sdt, params.gravity, obstacles);

            if params.compensate_drift {
                self.separate_particles(params.num_particle_iters);
            }

            self.handle_particle_collisions();

            self.particle_to_grid();

            self.mac.assign_uv_star();

            self.update_particle_density();

            self.solve_pressure(params.num_pressure_iters, sdt, params.over_relaxation, params.compensate_drift);

            self.grid_to_particle(params.flip_ratio);
        }
    }

    fn particle_radius(&self) -> f32 {
        self.particle_radius
    }
}
