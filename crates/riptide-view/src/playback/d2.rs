use bevy::{prelude::*, render::{camera::ScalingMode, view::NoFrustumCulling}};

use crate::particles_3d::{plugin::Particle3dPlugin, InstanceData, InstanceParticleData, Particle3d};

use super::{FluidDataDecoder, FluidMetadata, PlaybackPlugin, PlaybackState, SetupState};

pub struct Playback2DPlugin;

impl Plugin for Playback2DPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_plugins(PlaybackPlugin)
            .add_plugins(Particle3dPlugin)
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
    commands.insert_resource(ClearColor(Color::BLACK));

    commands.spawn((
        Camera3d::default(),
        Projection::Orthographic (
            OrthographicProjection {
                scaling_mode: ScalingMode::FixedVertical { viewport_height: 3.0 },
                ..OrthographicProjection::default_3d()
            },
        )
    ));
}

fn spawn_particles(
    mut commands: Commands,
    mut fluid: ResMut<FluidDataDecoder>,
    meta: Res<FluidMetadata>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut next_state: ResMut<NextState<SetupState>>,
    particles_query: Query<Entity, With<Particle>>,
    mut cameras: Query<(&mut Transform, &mut Projection)>,
) {
    for entity in &particles_query {
        commands.entity(entity).despawn();
    }

    let size: Vec2 = meta.0.size::<2>().into();
    for (mut transform, mut projection) in &mut cameras {
        if let Projection::Orthographic(proj) = &mut *projection {
            proj.scaling_mode = ScalingMode::FixedVertical { viewport_height: size.y };
        }
        transform.translation.x = size.x / 2.0;
        transform.translation.y = size.y / 2.0;
    }

    let mesh = meshes.add(Rectangle::from_size(Vec2::splat(meta.0.particle_radius)));

    let frame = fluid.0.decode_frame().unwrap();

    let Some(frame) = frame else {
        return;
    };

    commands.spawn((
        Mesh3d(mesh),
        Particle3d,
        InstanceParticleData(
            frame.positions.iter::<2>()
                .map(|pos| InstanceData::new(
                    Vec3::new(pos[0], pos[1], 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
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

    for (pos, instance_data) in frame.positions.iter::<2>().zip(particles.0.iter_mut()) {
        let pos: Vec2 = pos.into();

        instance_data.position = Vec3::new(pos.x, pos.y, 0.0);
    }
}
