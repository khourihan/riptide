use bevy::{color::palettes::css::{GREEN, PURPLE}, prelude::*, render::view::NoFrustumCulling};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use crate::particles_3d::{plugin::Particle3dPlugin, InstanceData, InstanceParticleData, Particle3d, ParticleLight};

use super::{FluidDataDecoder, FluidMetadata, PlaybackPlugin, PlaybackState, SetupState};

pub struct Playback3DPlugin;

impl Plugin for Playback3DPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins(PlaybackPlugin)
            .add_plugins(PanOrbitCameraPlugin)
            .add_plugins(Particle3dPlugin)
            .add_systems(Startup, setup)
            .insert_resource(Light {
                direction: Vec3::new(1.0, -0.5, 0.0).normalize(),
                brightness: 1.0,
                ambient: Vec4::new(1.0, 1.0, 1.0, 0.08),
            })
            .add_systems(Update, (
                spawn_particles.run_if(in_state(SetupState::NotReady)),
                progress_playback.run_if(in_state(PlaybackState::Playing)).run_if(in_state(SetupState::Ready)),
                draw_bounds.run_if(in_state(SetupState::Ready)),
                draw_light_direction.run_if(in_state(SetupState::Ready)),
            ));
    }
}

#[derive(Resource)]
struct Light {
    direction: Vec3,
    brightness: f32,
    ambient: Vec4,
}

fn setup(
    mut commands: Commands,
) {
    commands.insert_resource(ClearColor(Color::BLACK));

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
    meta: Res<FluidMetadata>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut next_state: ResMut<NextState<SetupState>>,
    particles_query: Query<Entity, With<Particle3d>>,
    light: Res<Light>,
) {
    for entity in &particles_query {
        commands.entity(entity).despawn();
    }

    let frame = fluid.0.decode_frame().unwrap();

    let Some(frame) = frame else {
        return;
    };
    
    let mesh = meshes.add(Rectangle::from_size(Vec2::splat(meta.0.particle_radius)));

    info!("spawned {} fluid particles", frame.positions.len::<3>());

    commands.spawn((
        Mesh3d(mesh),
        Particle3d,
        ParticleLight {
            direction: light.direction,
            brightness: light.brightness,
            ambient: light.ambient,
        },
        InstanceParticleData(
            frame.positions.iter::<3>()
                .zip(frame.gradients.iter::<3>())
                .map(|(pos, grad)| InstanceData::new(
                    Vec3::new(pos[0], pos[1], pos[2]),
                    Vec3::new(grad[0], grad[1], grad[2]),
                    1.0,
                    LinearRgba::from(Color::srgb(0.0, 0.0, 1.0)),
                ))
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

    for ((pos, grad), instance_data) in frame.positions.iter::<3>().zip(frame.gradients.iter::<3>()).zip(particles.0.iter_mut()) {
        let pos: Vec3 = pos.into();
        let grad: Vec3 = grad.into();

        instance_data.position = pos;
        instance_data.normal = Vec4::new(grad.x, grad.y, grad.z, 0.0);
    }
}

fn draw_bounds(
    mut gizmos: Gizmos,
    meta: Res<FluidMetadata>,
) {
    let size: Vec3 = meta.0.size::<3>().into();
    gizmos.cuboid(Transform::from_scale(size).with_translation(size / 2.0), GREEN)
}

fn draw_light_direction(
    mut gizmos: Gizmos,
    light: Res<Light>,
    meta: Res<FluidMetadata>,
) {
    let size: Vec3 = meta.0.size::<3>().into();
    gizmos.arrow(size, light.direction + size, PURPLE);
}
