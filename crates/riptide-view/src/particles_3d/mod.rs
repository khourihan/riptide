use bevy::{ecs::query::QueryItem, prelude::*, render::{extract_component::ExtractComponent, render_resource::Buffer}};
use bytemuck::{Pod, Zeroable};

pub mod pipeline;
pub mod plugin;
pub mod extract;

const PARTICLE_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(0xc3f53d447b7a8000);

#[derive(Default, Clone, Copy, Component, ExtractComponent, Debug, Reflect)]
#[require(Particle3dDepth)]
pub struct Particle3d;

#[derive(Clone, Component, Reflect, Debug, Copy)]
#[reflect(Component)]
pub struct Particle3dDepth(pub bool);

impl Default for Particle3dDepth {
    fn default() -> Self {
        Self(true)
    }
}

#[derive(Default, Clone, Copy, Component, Debug, Reflect)]
pub struct Particle3dLockAxis {
    pub y_axis: bool,
    pub rotation: bool,
}

#[derive(Component, Deref)]
pub struct InstanceParticleData(pub Vec<InstanceData>);

impl ExtractComponent for InstanceParticleData {
    type QueryData = &'static InstanceParticleData;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(item: QueryItem<'_, Self::QueryData>) -> Option<Self::Out> {
        Some(InstanceParticleData(item.0.clone()))
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct InstanceData {
    pub position: Vec3,
    pub scale: f32,
    pub color: [f32; 4],
}

#[derive(Component)]
pub struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
}
