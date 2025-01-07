use bevy::{prelude::*, render::extract_component::ExtractComponent};

pub mod pipeline;
pub mod plugin;
pub mod extract;

const PARTICLE_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(0xc3f53d447b7a8000);

#[derive(Clone, Component, Default, Reflect)]
#[reflect(Component)]
#[require(Particle3d, Particle3dMesh, Transform, Visibility)]
pub struct Particle3dColor(pub Color);

#[derive(Clone, Component, Reflect, Default)]
#[reflect(Component)]
pub struct Particle3dMesh(pub Handle<Mesh>);

#[derive(Clone, Component, Reflect, Debug, Copy)]
#[reflect(Component)]
pub struct Particle3dDepth(pub bool);

impl Default for Particle3dDepth {
    fn default() -> Self {
        Self(true)
    }
}

#[derive(Default, Clone, Copy, Component, ExtractComponent, Debug, Reflect)]
#[require(Particle3dDepth)]
pub struct Particle3d;

#[derive(Default, Clone, Copy, Component, Debug, Reflect)]
pub struct Particle3dLockAxis {
    pub y_axis: bool,
    pub rotation: bool,
}
