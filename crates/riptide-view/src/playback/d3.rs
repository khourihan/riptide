use std::time::Instant;

use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use crate::particles_3d::{plugin::Particle3dPlugin, Particle3d, Particle3dColor, Particle3dMesh};

use super::{FluidDataDecoder, Particles, PlaybackPlugin, PlaybackState, SetupState};

pub struct Playback3DPlugin;

impl Plugin for Playback3DPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins(PlaybackPlugin)
            .add_plugins(PanOrbitCameraPlugin)
            .add_plugins(Particle3dPlugin)
            .add_systems(Startup, setup)
            .add_systems(Update, (
                spawn_particles.run_if(in_state(SetupState::NotReady)),
                progress_playback.run_if(in_state(PlaybackState::Playing)).run_if(in_state(SetupState::Ready)),
            ));
    }
}

fn setup(
    mut commands: Commands,
    mut ambient_light: ResMut<AmbientLight>,
) {
    ambient_light.brightness = 1000.0;

    commands.spawn((
        Transform::from_xyz(0.0, 0.0, 0.0),
        PanOrbitCamera {
            button_orbit: MouseButton::Left,
            button_pan: MouseButton::Left,
            modifier_pan: Some(KeyCode::ShiftLeft),
            ..default()
        },
    ));
}

fn spawn_particles(
    mut commands: Commands,
    mut fluid: ResMut<FluidDataDecoder>,
    mut particles: ResMut<Particles>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut next_state: ResMut<NextState<SetupState>>,
    particles_query: Query<Entity, With<Particle3d>>,
) {
    particles.0.clear();

    for entity in &particles_query {
        commands.entity(entity).despawn();
    }

    let before = Instant::now();
    let frame = fluid.0.decode_frame().unwrap();
    info!("decoding took {} seconds", Instant::now().duration_since(before).as_secs_f32());

    let Some(frame) = frame else {
        return;
    };

    for pos in frame.positions.iter::<3>() {
        let pos: Vec3 = pos.into();

        particles.0.push(commands.spawn((
            Particle3dColor(Color::srgb(0.0, 0.0, 1.0)),
            Particle3dMesh(meshes.add(Rectangle::from_size(Vec2::splat(0.3 / 50.0)))),
            Transform::from_translation(pos),
        )).id());
    }

    fluid.0.reset();

    info!("spawned {} fluid particles", particles.0.len());

    next_state.set(SetupState::Ready);
}

fn progress_playback(
    mut fluid: ResMut<FluidDataDecoder>,
    particles: Res<Particles>,
    mut particles_query: Query<(&mut Transform, &mut Particle3dColor), With<Particle3d>>,
    mut next_state: ResMut<NextState<PlaybackState>>,
) {
    let before = Instant::now();
    let Some(frame) = fluid.0.decode_frame().unwrap() else {
        next_state.set(PlaybackState::Paused);
        fluid.0.reset();
        return;
    };
    info!("decoding took {} seconds", Instant::now().duration_since(before).as_secs_f32());

    for (pos, &entity) in frame.positions.iter::<3>().zip(particles.0.iter()) {
        let pos: Vec3 = pos.into();

        let (mut transform, _color) = particles_query.get_mut(entity).unwrap();

        transform.translation = pos;
    }
}
