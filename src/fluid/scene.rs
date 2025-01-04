use super::{obstacle::{Obstacle, ObstacleId, ObstacleSet}, Fluid};

pub struct Scene<const D: usize, F, P> {
    /// The fluid for this scene.
    pub fluid: F,
    /// The parameters for this scene's fluid.
    params: P,
    /// Domain size.
    size: [f32; D],
    /// The obstacles in this scene.
    obstacles: ObstacleSet<D>,
    /// The number of obstacles (used for IDs).
    n_obstacles: usize,
}

impl<const D: usize, F: Fluid<D, Params = P>, P> Scene<D, F, P> {
    #[inline(always)]
    pub fn new(fluid: F, params: P, size: [f32; D]) -> Self {
        Self {
            params,
            fluid,
            size,
            obstacles: ObstacleSet::default(),
            n_obstacles: 0,
        }
    }

    #[inline(always)]
    pub fn size(&self) -> [f32; D] {
        self.size
    }

    /// Adds an obstacle to the set, returning its ID.
    pub fn add_obstacle<T: Obstacle<D> + 'static>(&mut self, obstacle: T) -> ObstacleId {
        let i = self.n_obstacles;
        self.n_obstacles += 1;

        self.obstacles.obstacles.insert(i, Box::new(obstacle));
        ObstacleId(i)
    }

    /// Removes an obstacle from the set, given its ID.
    pub fn remove_obstacle(&mut self, id: ObstacleId) -> Option<Box<dyn Obstacle<D>>> {
        self.obstacles.obstacles.remove(&id.0)
    }

    /// Insert an obstacle into the set at the given ID, overriding and returning the old value if
    /// it was previously in the set.
    pub fn insert_obstacle<T: Obstacle<D> + 'static>(&mut self, id: ObstacleId, obstacle: T) -> Option<Box<dyn Obstacle<D>>> {
        self.obstacles.obstacles.insert(id.0, Box::new(obstacle))
    }

    pub fn step(&mut self, dt: f32) {
        self.fluid.step(
            dt,
            &self.params,
            &self.obstacles,
        );
    }
}
