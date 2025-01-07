use bevy::{prelude::*, render::{camera::ScalingMode, mesh::{CircularMeshUvMode, CircularSectorMeshBuilder}}};

use super::{FluidData, FluidDataFrame, Particles, PlaybackPlugin, PlaybackState, SetupState};

pub struct Playback2DPlugin;

impl Plugin for Playback2DPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins(PlaybackPlugin)
            .add_systems(Startup, setup)
            .add_systems(Update, (
                spawn_particles.run_if(in_state(SetupState::NotReady)),
                progress_playback.run_if(in_state(PlaybackState::Playing)),
            ));
    }
}

#[derive(Component, Clone, Copy)]
struct Particle;

fn setup(
    mut commands: Commands,
) {
    commands.spawn((
        Camera2d,
        OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical { viewport_height: 3.0 },
            ..OrthographicProjection::default_2d()
        }
    ));
}

fn spawn_particles(
    mut commands: Commands,
    fluid: Res<FluidData>,
    mut particles: ResMut<Particles>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut next_state: ResMut<NextState<SetupState>>,
    particles_query: Query<Entity, With<Particle>>,
    mut cameras: Query<(&mut Transform, &mut OrthographicProjection)>,
) {
    particles.0.clear();

    for entity in &particles_query {
        commands.entity(entity).despawn();
    }

    let size: Vec2 = fluid.0.size::<2>().into();
    for (mut transform, mut projection) in &mut cameras {
        projection.scaling_mode = ScalingMode::FixedVertical { viewport_height: size.y };
        transform.translation.x = size.x / 2.0;
        transform.translation.y = size.y / 2.0;
    }

    let material = materials.add(Color::srgb(0.0, 0.0, 1.0));

    for pos in fluid.0.frames[0].positions.iter::<2>() {
        let pos: Vec2 = pos.into();

        let sector = CircularSector::from_turns(0.2 / 100.0 * size.y, 1.0);
        let sector_angle = -sector.half_angle();
        let sector_mesh = CircularSectorMeshBuilder::new(sector).uv_mode(CircularMeshUvMode::Mask {
            angle: sector_angle,
        });

        particles.0.push(commands.spawn((
            Mesh2d(meshes.add(sector_mesh)),
            MeshMaterial2d(material.clone()),
            Transform::from_xyz(pos.x, pos.y, 0.0),
            Particle,
        )).id());
    }

    info!("spawned {} fluid particles", particles.0.len());

    next_state.set(SetupState::Ready);
}

fn progress_playback(
    fluid: Res<FluidData>,
    mut frame: ResMut<FluidDataFrame>,
    particles: Res<Particles>,
    mut particles_query: Query<&mut Transform, With<Particle>>,
    mut next_state: ResMut<NextState<PlaybackState>>,
) {
    if frame.0 >= fluid.0.frames.len() {
        next_state.set(PlaybackState::Paused);
        frame.0 = 0;
        return;
    }

    for (pos, &entity) in fluid.0.frames[frame.0].positions.iter::<2>().zip(particles.0.iter()) {
        let pos: Vec2 = pos.into();

        let mut transform = particles_query.get_mut(entity).unwrap();

        transform.translation = Vec3::new(pos.x, pos.y, 0.0);
    }

    frame.0 += 1;
}
