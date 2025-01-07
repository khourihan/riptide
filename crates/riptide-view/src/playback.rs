use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use riptide_io::decode::FluidData as IoFluidData;

use crate::particles::{plugin::ParticlePlugin, Particle, ParticleColor, ParticleMesh};

pub struct PlaybackPlugin;

impl Plugin for PlaybackPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins(PanOrbitCameraPlugin)
            .add_plugins(ParticlePlugin)
            .init_state::<PlaybackState>()
            .init_state::<SetupState>()
            .init_resource::<Particles>()
            .init_resource::<FluidDataFrame>()
            .add_systems(Startup, setup)
            .add_systems(Update, (
                spawn_particles.run_if(in_state(SetupState::NotReady)),
                progress_playback.run_if(in_state(PlaybackState::Playing)).run_if(in_state(SetupState::Ready)),

                change_state_playing.run_if(in_state(PlaybackState::Paused)),
                change_state_paused.run_if(in_state(PlaybackState::Playing)),
            ));
    }
}

#[derive(Resource)]
pub struct FluidData(pub IoFluidData);

#[derive(Resource, Default)]
struct FluidDataFrame(usize);

#[derive(States, Clone, PartialEq, Eq, Hash, Debug, Default)]
enum PlaybackState {
    Playing,
    #[default]
    Paused
}

#[derive(States, Clone, PartialEq, Eq, Hash, Debug, Default)]
enum SetupState {
    Ready,
    #[default]
    NotReady,
}

#[derive(Resource, Default)]
struct Particles(Vec<Entity>);

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
    fluid: Res<FluidData>,
    mut particles: ResMut<Particles>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut next_state: ResMut<NextState<SetupState>>,
    particles_query: Query<Entity, With<Particle>>,
) {
    particles.0.clear();

    for entity in &particles_query {
        commands.entity(entity).despawn();
    }

    for pos in fluid.0.frames[0].positions.iter::<3>() {
        let pos: Vec3 = pos.into();

        particles.0.push(commands.spawn((
            ParticleColor(Color::srgb(0.0, 0.0, 1.0)),
            ParticleMesh(meshes.add(Rectangle::from_size(Vec2::splat(0.3 / 50.0)))),
            Transform::from_translation(pos),
        )).id());
    }

    info!("spawned {} fluid particles", particles.0.len());

    next_state.set(SetupState::Ready);
}

fn progress_playback(
    fluid: Res<FluidData>,
    mut frame: ResMut<FluidDataFrame>,
    particles: Res<Particles>,
    mut particles_query: Query<(&mut Transform, &mut ParticleColor), With<Particle>>,
    mut next_state: ResMut<NextState<PlaybackState>>,
) {
    if frame.0 >= fluid.0.frames.len() {
        next_state.set(PlaybackState::Paused);
        frame.0 = 0;
        return;
    }

    for (pos, &entity) in fluid.0.frames[frame.0].positions.iter::<3>().zip(particles.0.iter()) {
        let pos: Vec3 = pos.into();

        let (mut transform, _color) = particles_query.get_mut(entity).unwrap();

        transform.translation = pos;
    }

    frame.0 += 1
}

fn change_state_playing(
    keys: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<PlaybackState>>,
) {
    if keys.just_pressed(KeyCode::Space) {
        next_state.set(PlaybackState::Playing);
    }
}

fn change_state_paused(
    keys: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<PlaybackState>>,
) {
    if keys.just_pressed(KeyCode::Space) {
        next_state.set(PlaybackState::Paused);
    }
}
