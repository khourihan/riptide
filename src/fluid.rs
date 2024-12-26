use glam::{UVec2, Vec2};
use ndarray::{azip, Array0, Array1, Array2, Axis};

pub struct Fluid {
    density: f32,
    pub size: UVec2,
    spacing: f32,

    rest_density: f32,
    particle_radius: f32,
    particle_spacing: f32,
    n_particles: usize,
    particle_resolution: UVec2,

    uvs: Array2<Vec2>,
    dudvs: Array2<Vec2>,
    prev_uvs: Array2<Vec2>,
    pressure: Array2<f32>,
    pub solid: Array2<f32>,
    cell_type: Array2<CellType>,

    pub positions: Array1<Vec2>,
    velocities: Array1<Vec2>,
    densities: Array2<f32>,
    cell_particle_count: Array1<usize>,
    first_cell_particle: Array1<usize>,
    cell_particle_indices: Array1<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CellType {
    Fluid,
    Solid,
    Air,
}

impl Fluid {
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
        let solid = Array2::from_elem((size.x as usize, size.y as usize), 0.0);
        let cell_type = Array2::from_elem((size.x as usize, size.y as usize), CellType::Fluid);

        let positions = Array1::from_vec(vec![]);
        let velocities = Array1::from_vec(vec![]);
        let densities = Array2::from_elem((size.x as usize, size.y as usize), 0.0);

        let particle_spacing = 2.2 * particle_radius;
        let particle_resolution = UVec2::new(
            (width as f32 / particle_spacing).floor() as u32 + 1,
            (height as f32 / particle_spacing).floor() as u32 + 1,
        );
        let cell_count = (particle_resolution.x * particle_resolution.y) as usize;

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
            uvs,
            dudvs,
            prev_uvs,
            pressure,
            solid,
            cell_type,
            positions,
            velocities,
            densities,
            cell_particle_count,
            first_cell_particle,
            cell_particle_indices,
        }
    }

    pub fn insert_particle(&mut self, pos: Vec2) {
        // let (min, max) = self.bounds();
        // pos = (1.0 - pos) * min + pos * max;

        let _ = self.positions.push(Axis(0), Array0::from_elem((), pos).view());
        let _ = self.velocities.push(Axis(0), Array0::from_elem((), Vec2::ZERO).view());
        let _ = self.cell_particle_indices.push(Axis(0), Array0::from_elem((), 0).view());
        self.n_particles += 1;
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
        self.cell_particle_count.fill(0);

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
        let (min, max) = self.bounds();

        azip!((p in &mut self.positions, v in &mut self.velocities) {
            // TODO: Obstacle collision

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

            for (i, cell_type) in self.cell_type.indexed_iter() {
                if *cell_type == CellType::Fluid {
                    sum += self.densities[i];
                    num_fluid_cells += 1;
                }
            }

            if num_fluid_cells > 0 {
                self.rest_density = sum / num_fluid_cells as f32;
            }
        }
    }

    fn transfer_velocities<const TO_GRID: bool>(&mut self, flip_ratio: f32) {
        let h = self.spacing;
        let h1 = h.recip();
        let h2 = 0.5 * h;

        if TO_GRID {
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
        }

        for dim in 0..2 {
            let delta = Vec2::new(
                if dim == 0 { 0.0 } else { h2 },
                if dim == 0 { h2 } else { 0.0 },
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

                if TO_GRID {
                    let v = self.velocities[i][dim];
                    self.uvs[i0][dim] += v * d0;
                    self.uvs[i1][dim] += v * d1;
                    self.uvs[i2][dim] += v * d2;
                    self.uvs[i3][dim] += v * d3;
                    self.dudvs[i0] += d0;
                    self.dudvs[i1] += d1;
                    self.dudvs[i2] += d2;
                    self.dudvs[i3] += d3;
                } else {
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

            if TO_GRID {
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
    }

    fn solve_incompressibility(&mut self, num_iters: usize, dt: f32, overrelaxation: f32, compensate_drift: bool) {
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
                    p *= overrelaxation;
                    self.pressure[center] += cp * p;

                    self.uvs[center].x -= sx0 * p;
                    self.uvs[right].x += sx1 * p;
                    self.uvs[center].y -= sy0 * p;
                    self.uvs[top].y += sy1 * p;
                }
            }
        }
    }

    pub fn step(
        &mut self,
        dt: f32,
        gravity: Vec2,
        flip_ratio: f32,
        num_pressure_iters: usize,
        num_particle_iters: usize,
        overrelaxation: f32,
        compensate_drift: bool,
        separate_particles: bool,
    ) {
        let num_substeps: usize = 2;
        let sdt = dt / num_substeps as f32;

        for _step in 0..num_substeps {
            self.integrate_particles(sdt, gravity);
            if separate_particles {
                self.push_particles_apart(num_particle_iters);
            }
            self.handle_particle_collisions();
            self.transfer_velocities::<true>(0.0);
            self.update_particle_density();
            self.solve_incompressibility(num_pressure_iters, sdt, overrelaxation, compensate_drift);
            self.transfer_velocities::<false>(flip_ratio);
        }
    }
}
