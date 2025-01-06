use bevy::prelude::*;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use riptide_io::decode::FluidData;

use crate::particles::{plugin::ParticlePlugin, ParticleColor, ParticleMesh};

pub struct PlaybackPlugin {
    // pub data: FluidData,
}

impl Plugin for PlaybackPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins(PanOrbitCameraPlugin)
            .add_plugins(ParticlePlugin)
            .add_systems(Startup, setup);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
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

    commands.spawn((
        ParticleColor(Color::srgb(0.0, 0.0, 1.0)),
        ParticleMesh(meshes.add(Rectangle::from_size(Vec2::splat(2.0)))),
        Transform::from_xyz(0.0, 0.0, 5.0),
    ));
}
