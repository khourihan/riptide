use draw::DrawState;
use fluid::{flip::d2::{FlipFluid2D, FlipFluid2DParams}, obstacle::circle::Circle, scene::Scene};
use glam::{Vec2, Vec4};

mod fluid;
mod io;
mod draw;

fn main() {
    let fps = 60;
    let screen_width = 1920;
    let screen_height = 1080;
    let aspect = screen_width as f32 / screen_height as f32;

    let mut state = DrawState::new("output/fluid.mp4", screen_width, screen_height, fps);

    let height = 3.0;
    let resolution = 100;
    let width = height * aspect;
    let spacing = height / resolution as f32;
    let particle_radius = 0.2;

    let fluid = FlipFluid2D::new(1000.0, width as u32, height as u32, spacing, particle_radius * spacing);
    let params = FlipFluid2DParams::default();

    let mut scene = Scene::new(fluid, params, [width, height]);

    let size = Vec2::new(width, height);
    let particle_radius = particle_radius * spacing;

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
            let mut s = 1.0;
            if i == 0 || i == grid_size.x - 1 || j == 0 {
                s = 0.0;
            }
            scene.fluid.set_solid(i as usize, j as usize, s);
        }
    }

    let obstacle_r = 2.0;
    let mut circle = Circle::new(Vec2::new(size.x / 2.0 + obstacle_r, size.y / 2.0), 0.25);
    let circle_id = scene.add_obstacle(circle);

    let duration_s = 10.0;
    let frames = (duration_s * fps as f32) as usize;
    let dt = 1.0 / fps as f32;

    for frame in 0..frames {
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
            state.circle(center_screen, 0.25 / size.y * screen_height as f32, Vec4::new(1.0, 0.0, 0.0, 1.0));

            scene.insert_obstacle(circle_id, circle);
        } else {
            scene.remove_obstacle(circle_id);
        }

        scene.step(dt);
        
        for (&pos, _vel, &rough) in scene.fluid.iter_particles() {
            let mut p = pos / size;
            p.y = 1.0 - p.y;

            let col = Vec4::new(0.0, 0.0, 1.0, 1.0).lerp(Vec4::new(1.0, 1.0, 1.0, 1.0), rough);
            state.circle(p, particle_radius / size.y * screen_height as f32, col);
        }


        state.next();
    }

    state.finish();
}
