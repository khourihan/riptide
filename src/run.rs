use glam::{Vec2, Vec3};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};

use riptide_fluids::{flip::{d2::{FlipFluid2D, FlipFluid2DParams}, d3::{FlipFluid3D, FlipFluid3DParams}}, obstacle::circle::Circle, scene::Scene};
use riptide_io::encode::FluidDataEncoder;

pub fn run_d2(
    mut encoder: FluidDataEncoder,
    fps: u32,
    frames: u64,
    size: Vec2,
    resolution: u32,
    particle_radius: f32,
) {
    let spacing = size.y / resolution as f32;
    let particle_radius = particle_radius * spacing;

    let fluid = FlipFluid2D::new(1000.0, size, spacing, particle_radius);
    let params = FlipFluid2DParams::default();

    let mut scene = Scene::new(fluid, params, [size.x, size.y]);

    let water_height = 0.8;
    let water_width = 0.6;
    let dx = 2.0 * particle_radius;
    let dy = 3f32.sqrt() / 2.0 * dx;
    let nx = ((water_width * size.x - 2.0 * spacing - 2.0 * particle_radius) / dx).floor() as usize;
    let ny = ((water_height * size.y - 2.0 * spacing - 2.0 * particle_radius) / dy).floor() as usize;

    for i in 0..nx {
        for j in 0..ny {
            let x = spacing + particle_radius + dx * i as f32 + (if j % 2 == 0 { 0.0 } else { particle_radius });
            let y = spacing + particle_radius + dy * j as f32;
            scene.fluid.insert_particle(Vec2::new(x, y));
        }
    }

    let grid_size = scene.fluid.size();
    for i in 0..grid_size.x {
        for j in 0..grid_size.y {
            let mut s = false;
            if i == 0 || i == grid_size.x - 1 || j == 0 || j == grid_size.x - 1 {
                s = true;
            }
            scene.fluid.set_solid(i as usize, j as usize, s);
        }
    }

    let obstacle_r = 2.0;
    let mut circle = Circle::new(Vec2::new(size.x / 2.0 + obstacle_r, size.y / 2.0), 0.25);
    let circle_id = scene.add_obstacle(circle);

    let dt = 1.0 / fps as f32;

    let bar_template = "Running Simulation {spinner:.green} [{elapsed}] [{bar:50.white/white}] {pos}/{len} ({eta})";
    let style = ProgressStyle::with_template(bar_template).unwrap()
        .progress_chars("=> ").tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");
    let progress = ProgressBar::new(frames).with_style(style);

    let duration_s = frames as f32 / fps as f32;

    encoder.encode_metadata(&scene).unwrap();

    for frame in (0..frames).progress_with(progress) {
        let t = frame as f32 / frames as f32;

        // Water drop
        /*
        if frame == 120 || frame == 360 {
        let dx = 2.0 * particle_radius;
        let dy = 3f32.sqrt() / 2.0 * dx;
        let nx = ((size.x - 2.0 * spacing - 2.0 * particle_radius) / dx).floor() as usize;
        let ny = ((size.x - 2.0 * spacing - 2.0 * particle_radius) / dy).floor() as usize;
        let center = size * Vec2::new(0.25, 0.75);
        let radius = 0.5;

           for i in 0..nx {
               for j in 0..ny {
                   let x = spacing + particle_radius + dx * i as f32 + (if j % 2 == 0 { 0.0 } else { particle_radius });
                   let y = spacing + particle_radius + dy * j as f32;

                   let p = Vec2::new(x, y);
                   let d = p - center;

                   if d.length_squared() > radius * radius {
                       continue;
                   }

                   scene.insert_particle(Vec2::new(x, y));
               }
           }
        }
        */

        // Spinning circle
        if frame > 120 && t < 0.5 {
            let theta = t * duration_s * std::f32::consts::TAU + std::f32::consts::FRAC_PI_2;
            let center = Vec2::new(size.x + obstacle_r * theta.cos(), size.y + obstacle_r * theta.sin()) / 2.0;
            circle.set_position(center);

            let mut center_screen = center / Vec2::new(size.x, size.y);
            center_screen.y = 1.0 - center_screen.y;

            scene.insert_obstacle(circle_id, circle);
        } else {
            scene.remove_obstacle(circle_id);
        }

        scene.step(dt);
        encoder.encode_frame(&scene).unwrap();
    }
}

pub fn run_d3(
    mut encoder: FluidDataEncoder,
    fps: u32,
    frames: u64,
    size: Vec3,
    resolution: u32,
    particle_radius: f32,
) {
    let spacing = size.y / resolution as f32;
    let particle_radius = particle_radius * spacing;

    let fluid = FlipFluid3D::new(1000.0, size.as_uvec3(), spacing, particle_radius);
    let params = FlipFluid3DParams::default();

    let mut scene = Scene::new(fluid, params, [size.x, size.y, size.z]);

    let water_height = 0.8;
    let water_width = 0.6;
    let water_depth = 0.6;
    let dx = 2.0 * particle_radius;
    let dy = 3f32.sqrt() / 2.0 * dx;
    let dz = dx;
    let nx = ((water_width * size.x - 2.0 * spacing - 2.0 * particle_radius) / dx).floor() as usize;
    let ny = ((water_height * size.y - 2.0 * spacing - 2.0 * particle_radius) / dy).floor() as usize;
    let nz = ((water_depth * size.z - 2.0 * spacing - 2.0 * particle_radius) / dz).floor() as usize;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let x = spacing + particle_radius + dx * i as f32 + (if j % 2 == 0 { 0.0 } else { particle_radius });
                let y = spacing + particle_radius + dy * j as f32;
                let z = spacing + particle_radius + dz * k as f32 + (if j % 2 == 0 { 0.0 } else { particle_radius });
                scene.fluid.insert_particle(Vec3::new(x, y, z));
            }
        }
    }

    let grid_size = scene.fluid.size();
    for i in 0..grid_size.x {
        for j in 0..grid_size.y {
            for k in 0..grid_size.z {
                let mut s = 1.0;
                if i == 0 || i == grid_size.x - 1 || j == 0 || j == grid_size.y - 1 || k == 0 || k == grid_size.z - 1 {
                    s = 0.0;
                }
                scene.fluid.set_solid(i as usize, j as usize, k as usize, s);
            }
        }
    }

    let dt = 1.0 / fps as f32;

    let bar_template = "Running Simulation {spinner:.green} [{elapsed}] [{bar:50.white/white}] {pos}/{len} ({eta})";
    let style = ProgressStyle::with_template(bar_template).unwrap()
        .progress_chars("=> ").tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");
    let progress = ProgressBar::new(frames).with_style(style);

    encoder.encode_metadata(&scene).unwrap();

    for _frame in (0..frames).progress_with(progress) {
        scene.step(dt);
        encoder.encode_frame(&scene).unwrap();
    }
}
