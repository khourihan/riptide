use bevy::{prelude::*, render::extract_component::ExtractComponent};

pub mod pipeline;
pub mod plugin;
pub mod extract;

pub(self) const PARTICLE_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(0xc3f53d447b7a8000);

#[derive(Clone, Component, Default, Reflect)]
#[reflect(Component)]
#[require(Particle, ParticleMesh, Transform, Visibility)]
pub struct ParticleColor(pub Color);

#[derive(Clone, Component, Reflect, Default)]
#[reflect(Component)]
pub struct ParticleMesh(pub Handle<Mesh>);

#[derive(Clone, Component, Reflect, Debug, Copy)]
#[reflect(Component)]
pub struct ParticleDepth(pub bool);

impl Default for ParticleDepth {
    fn default() -> Self {
        Self(true)
    }
}

#[derive(Default, Clone, Copy, Component, ExtractComponent, Debug, Reflect)]
#[require(ParticleDepth)]
pub struct Particle;

#[derive(Default, Clone, Copy, Component, Debug, Reflect)]
pub struct ParticleLockAxis {
    pub y_axis: bool,
    pub rotation: bool,
}
