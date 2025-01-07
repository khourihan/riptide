use bevy::{color::palettes::css::GREEN, prelude::*, render::view::NoFrustumCulling};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use crate::particles_3d::{plugin::Particle3dPlugin, InstanceData, InstanceParticleData, Particle3d};

use super::{FluidDataDecoder, FluidMetadata, PlaybackPlugin, PlaybackState, SetupState};

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
                draw_bounds.run_if(in_state(SetupState::Ready)),
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
    mut meshes: ResMut<Assets<Mesh>>,
    mut next_state: ResMut<NextState<SetupState>>,
    particles_query: Query<Entity, With<Particle3d>>,
) {
    for entity in &particles_query {
        commands.entity(entity).despawn();
    }

    let frame = fluid.0.decode_frame().unwrap();

    let Some(frame) = frame else {
        return;
    };
    
    let mesh = meshes.add(Rectangle::from_size(Vec2::splat(0.3 / 50.0)));

    commands.spawn((
        Mesh3d(mesh),
        Particle3d,
        InstanceParticleData(
            frame.positions.iter::<3>()
                .map(|pos| InstanceData {
                    position: Vec3::new(pos[0], pos[1], pos[2]),
                    scale: 1.0,
                    color: LinearRgba::from(Color::srgb(0.0, 0.0, 1.0)).to_f32_array(),
                })
                .collect()
        ),
        NoFrustumCulling,
    ));

    fluid.0.reset();

    next_state.set(SetupState::Ready);
}

fn progress_playback(
    mut fluid: ResMut<FluidDataDecoder>,
    mut particles: Query<&mut InstanceParticleData>,
    mut next_state: ResMut<NextState<PlaybackState>>,
) {
    let Some(frame) = fluid.0.decode_frame().unwrap() else {
        next_state.set(PlaybackState::Paused);
        fluid.0.reset();
        return;
    };

    let mut particles = particles.single_mut();

    for (pos, instance_data) in frame.positions.iter::<3>().zip(particles.0.iter_mut()) {
        let pos: Vec3 = pos.into();

        instance_data.position = pos;
    }
}

fn draw_bounds(
    mut gizmos: Gizmos,
    meta: Res<FluidMetadata>,
) {
    let size: Vec3 = meta.0.size::<3>().into();
    gizmos.cuboid(Transform::from_scale(size).with_translation(size / 2.0), GREEN)
}
