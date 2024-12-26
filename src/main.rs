use draw::DrawState;
use backend::fluid::Fluid;
use glam::{Vec2, Vec4};

mod backend;
mod draw;

fn main() {
    let fps = 60;
    let screen_width = 1920;
    let screen_height = 1080;

    let mut state = DrawState::new("output/fluid.mp4", screen_width, screen_height, fps);

    let height = 3.0;
    let width = 0.5 * height * (screen_width as f32 / screen_height as f32);
    let density = 1000.0;
    let res = 100;
    let spacing = height / res as f32;
    let particle_radius = 0.3 * spacing;

    let gravity = Vec2::new(0.0, -9.81);
    let flip_ratio = 0.9;
    let num_pressure_iters = 100;
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

    let duration_s = 20.0;
    let frames = (duration_s * fps as f32) as usize;
    let dt = 1.0 / fps as f32;

    for frame in 0..frames {
        if frame == frames / 4 {
            fluid.resize((width * 2.0) as u32, height as u32, spacing);

            for i in 0..fluid.size().x {
                for j in 0..fluid.size().y {
                    let mut s = 1.0;
                    if i == 0 || i == fluid.size().x - 1 || j == 0 {
                        s = 0.0;
                    }
                    fluid.set_solid(i as usize, j as usize, s);
                }
            }
        }

        fluid.step(dt, gravity, flip_ratio, num_pressure_iters, num_particle_iters, overrelaxation, compensate_drift, separate_particles);
        
        for (&pos, _vel, &rough) in fluid.iter_particles() {
            let mut p = pos / Vec2::new(width * 2.0, height);
            p.y = 1.0 - p.y;

            let col = Vec4::new(0.0, 0.0, 1.0, 1.0).lerp(Vec4::new(1.0, 1.0, 1.0, 1.0), rough);
            state.circle(p, particle_radius / height * screen_height as f32, col);
        }

        state.next();
    }

    state.finish();
}
