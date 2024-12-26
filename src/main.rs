use draw::DrawState;
use backend::{fluid::Fluid, obstacle::{circle::Circle, ObstacleSet}};
use glam::{Vec2, Vec4};

mod backend;
mod draw;

fn main() {
    let fps = 60;
    let screen_width = 1920;
    let screen_height = 1080;

    let mut state = DrawState::new("output/fluid.mp4", screen_width, screen_height, fps);

    let height = 3.0;
    let width = height * (screen_width as f32 / screen_height as f32);
    let density = 1000.0;
    let res = 100;
    let spacing = height / res as f32;
    let particle_radius = 0.3 * spacing;

    let gravity = Vec2::new(0.0, -9.81);
    let flip_ratio = 0.9;
    let num_pressure_iters = 50;
    let num_particle_iters = 2;
    let overrelaxation = 1.9;
    let compensate_drift = true;
    let separate_particles = true;

    let mut fluid = Fluid::new(density, width as u32, height as u32, spacing, particle_radius);

    let water_height = 0.8;
    let water_width = 0.6;
    let dx = 2.0 * particle_radius;
    let dy = 3f32.sqrt() / 2.0 * dx;
    let nx = ((water_width * width - 2.0 * spacing - 2.0 * particle_radius) / dx).floor() as usize;
    let ny = ((water_height * height - 2.0 * spacing - 2.0 * particle_radius) / dy).floor() as usize;

    for i in 0..nx {
        for j in 0..ny {
            let x = spacing + particle_radius + dx * i as f32 + (if j % 2 == 0 { 0.0 } else { particle_radius });
            let y = spacing + particle_radius + dy * j as f32;
            fluid.insert_particle(Vec2::new(x, y));
        }
    }

    for i in 0..fluid.size().x {
        for j in 0..fluid.size().y {
            let mut s = 1.0;
            if i == 0 || i == fluid.size().x - 1 || j == 0 {
                s = 0.0;
            }
            fluid.set_solid(i as usize, j as usize, s);
        }
    }

    let obstacle_r = 2.0;
    let mut circle = Circle::new(Vec2::new(width / 2.0 + obstacle_r, height / 2.0), 0.25);

    let duration_s = 10.0;
    let frames = (duration_s * fps as f32) as usize;
    let dt = 1.0 / fps as f32;

    for frame in 0..frames {
        let t = frame as f32 / frames as f32;

        let obstacles = if frame > 60 && frame < 180 {
            let theta = t * 30.0 * std::f32::consts::TAU;
            let center = Vec2::new(width + obstacle_r * theta.cos(), height + obstacle_r * theta.sin()) / 2.0;
            circle.set_position(center);

            let mut center_screen = center / Vec2::new(width, height);
            center_screen.y = 1.0 - center_screen.y;
            state.circle(center_screen, 0.25 / height * screen_height as f32, Vec4::new(1.0, 0.0, 0.0, 1.0));

            ObstacleSet::new(vec![Box::new(circle)])
        } else {
            ObstacleSet::new(vec![])
        };

        fluid.step(
            dt,
            gravity,
            flip_ratio,
            num_pressure_iters,
            num_particle_iters,
            overrelaxation,
            compensate_drift,
            separate_particles,
            &obstacles,
        );
        
        for (&pos, _vel, &rough) in fluid.iter_particles() {
            let mut p = pos / Vec2::new(width, height);
            p.y = 1.0 - p.y;

            let col = Vec4::new(0.0, 0.0, 1.0, 1.0).lerp(Vec4::new(1.0, 1.0, 1.0, 1.0), rough);
            state.circle(p, particle_radius / height * screen_height as f32, col);
        }


        state.next();
    }

    state.finish();
}
