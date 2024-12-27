use glam::{UVec2, Vec2};

use super::{fluid::Fluid, obstacle::{Obstacle, ObstacleId, ObstacleSet}};

pub struct Scene {
    height: f32,
    width: f32,
    particle_radius: f32,
    spacing: f32,
    gravity: Vec2,
    flip_ratio: f32,
    num_pressure_iters: usize,
    num_particle_iters: usize,
    over_relaxation: f32,
    compensate_drift: bool,
    separate_particles: bool,

    fluid: Fluid,
    obstacles: ObstacleSet,
    n_obstacles: usize,
}

impl Scene {
    #[allow(clippy::new_ret_no_self)]
    #[inline(always)]
    pub fn new() -> SceneBuilder {
        SceneBuilder::default()
    }

    /// The domain width and height of the simulation.
    #[inline(always)]
    pub fn size(&self) -> Vec2 {
        Vec2::new(self.width, self.height)
    }

    /// The size of the grid representing the fluid.
    #[inline(always)]
    pub fn grid_size(&self) -> UVec2 {
        self.fluid.size()
    }

    /// The size of a grid cell in the fluid relative to the domain width and height.
    #[inline(always)]
    pub fn spacing(&self) -> f32 {
        self.spacing
    }

    /// The radius of a fluid particle relative to the domain width and height.
    #[inline(always)]
    pub fn particle_radius(&self) -> f32 {
        self.particle_radius
    }

    /// Insert a fluid particle at the given position.
    #[inline(always)]
    pub fn insert_particle(&mut self, pos: Vec2) {
        self.fluid.insert_particle(pos)
    }

    /// Set a grid cell's solid state.
    #[inline(always)]
    pub fn set_solid(&mut self, i: usize, j: usize, s: f32) {
        self.fluid.set_solid(i, j, s)
    }

    #[inline(always)]
    pub fn iter_particles(&self) -> impl Iterator<Item = (&Vec2, &Vec2, &f32)> {
        self.fluid.iter_particles()
    }

    /// Adds an obstacle to the set, returning its ID.
    pub fn add_obstacle<T: Obstacle + 'static>(&mut self, obstacle: T) -> ObstacleId {
        let i = self.n_obstacles;
        self.n_obstacles += 1;

        self.obstacles.obstacles.insert(i, Box::new(obstacle));
        ObstacleId(i)
    }

    /// Removes an obstacle from the set, given its ID.
    pub fn remove_obstacle(&mut self, id: ObstacleId) -> Option<Box<dyn Obstacle>> {
        self.obstacles.obstacles.remove(&id.0)
    }

    /// Insert an obstacle into the set at the given ID, overriding and returning the old value if
    /// it was previously in the set.
    pub fn insert_obstacle<T: Obstacle + 'static>(&mut self, id: ObstacleId, obstacle: T) -> Option<Box<dyn Obstacle>> {
        self.obstacles.obstacles.insert(id.0, Box::new(obstacle))
    }

    pub fn step(&mut self, dt: f32) {
        self.fluid.step(
            dt,
            self.gravity,
            self.flip_ratio,
            self.num_pressure_iters,
            self.num_particle_iters,
            self.over_relaxation,
            self.compensate_drift,
            self.separate_particles,
            &self.obstacles,
        );
    }
}

impl Default for Scene {
    fn default() -> Self {
        SceneBuilder::default().build()
    }
}

pub struct SceneBuilder {
    height: f32,
    aspect: f32,
    density: f32,
    resolution: u32,
    particle_radius: f32,
    gravity: Vec2,
    flip_ratio: f32,
    num_pressure_iters: usize,
    num_particle_iters: usize,
    over_relaxation: f32,
    compensate_drift: bool,
    separate_particles: bool,
}

impl SceneBuilder {
    /// The height of the simulation domain.
    ///
    /// Defaults to `3.0`.
    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// The aspect ratio of the simulation domain.
    ///
    /// Defaults to `1.0`.
    pub fn aspect(mut self, aspect: f32) -> Self {
        self.aspect = aspect;
        self
    }

    /// The density of the fluid.
    ///
    /// Defaults to `1000.0`.
    pub fn density(mut self, density: f32) -> Self {
        self.density = density;
        self
    }

    /// The resolution of the grid used to represent the fluid.
    ///
    /// Defaults to `100`.
    pub fn resolution(mut self, resolution: u32) -> Self {
        self.resolution = resolution;
        self
    }

    /// The relative radius of a particle compared to the size of a grid cell.
    ///
    /// Defaults to `0.3`.
    pub fn particle_radius(mut self, particle_radius: f32) -> Self {
        self.particle_radius = particle_radius;
        self
    }

    /// The constant gravitational acceleration applied to the fluid.
    ///
    /// Defaults to `(0.0, -9.81)`.
    pub fn gravity(mut self, gravity: Vec2) -> Self {
        self.gravity = gravity;
        self
    }

    /// The ratio of FLIP to PIC when applying motion to particles, where `1.0` corresponds to FLIP
    /// only and `0.0` corresponds to PIC only.
    ///
    /// Defaults to `0.9`.
    pub fn flip_ratio(mut self, flip_ratio: f32) -> Self {
        self.flip_ratio = flip_ratio;
        self
    }

    /// The number of iterations spent solving fluid incompressibility.
    ///
    /// Defaults to `100`.
    pub fn num_pressure_iters(mut self, num_pressure_iters: usize) -> Self {
        self.num_pressure_iters = num_pressure_iters;
        self
    }

    /// The number of iterations spent pushing particles apart.
    ///
    /// Defaults to `2`.
    pub fn num_particle_iters(mut self, num_particle_iters: usize) -> Self {
        self.num_particle_iters = num_particle_iters;
        self
    }

    /// The over-relaxation of the fluid.
    ///
    /// Defaults to `1.9`.
    pub fn over_relaxation(mut self, over_relaxation: f32) -> Self {
        self.over_relaxation = over_relaxation;
        self
    }

    /// Whether or not to compensate for drifting of particles.
    ///
    /// Defaults to `true`.
    pub fn compensate_drift(mut self, compensate_drift: bool) -> Self {
        self.compensate_drift = compensate_drift;
        self
    }

    /// Whether or not to separate overlapping particles.
    ///
    /// Defaults to `true`.
    pub fn separate_particles(mut self, separate_particles: bool) -> Self {
        self.separate_particles = separate_particles;
        self
    }

    pub fn build(self) -> Scene {
        let height = self.height;
        let width = self.height * self.aspect;
        let spacing = self.height / self.resolution as f32;
        let particle_radius = self.particle_radius * spacing;

        Scene {
            height,
            width,
            spacing,
            particle_radius,
            gravity: self.gravity,
            flip_ratio: self.flip_ratio,
            num_pressure_iters: self.num_pressure_iters,
            num_particle_iters: self.num_particle_iters,
            over_relaxation: self.over_relaxation,
            compensate_drift: self.compensate_drift,
            separate_particles: self.separate_particles,

            fluid: Fluid::new(self.density, width as u32, height as u32, spacing, particle_radius),
            obstacles: ObstacleSet::default(),
            n_obstacles: 0,
        }
    }
}

impl Default for SceneBuilder {
    fn default() -> Self {
        Self {
            height: 3.0,
            aspect: 1.0,
            density: 1000.0,
            resolution: 100,
            particle_radius: 0.3,
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
