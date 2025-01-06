use bevy::{asset::load_internal_asset, core_pipeline::core_3d::Transparent3d, prelude::*, render::{extract_component::{ExtractComponentPlugin, UniformComponentPlugin}, render_phase::AddRenderCommand, render_resource::SpecializedMeshPipelines, view::{check_visibility, VisibilitySystems::CheckVisibility}, Render, RenderApp, RenderSet}};

use crate::particles::{pipeline::ParticleUniform, Particle, ParticleColor, ParticleMesh, PARTICLE_SHADER_HANDLE};

use super::{extract::extract_particles, pipeline::{prepare_particle_bind_group, prepare_particle_view_bind_groups, queue_particles, DrawParticle, ParticlePipeline}};

pub struct ParticlePlugin;

impl Plugin for ParticlePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            PARTICLE_SHADER_HANDLE,
            "particle.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins(UniformComponentPlugin::<ParticleUniform>::default())
            .add_plugins(ExtractComponentPlugin::<Particle>::default())
            .register_type::<ParticleMesh>()
            .register_type::<ParticleColor>()
            .add_systems(
                PostUpdate,
                check_visibility::<With<Particle>>.in_set(CheckVisibility),
            );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .add_render_command::<Transparent3d, DrawParticle>()
            .init_resource::<ParticlePipeline>()
            .init_resource::<SpecializedMeshPipelines<ParticlePipeline>>()
            .add_systems(
                ExtractSchedule,
                extract_particles,
            )
            .add_systems(Render, queue_particles.in_set(RenderSet::Queue))
            .add_systems(
                Render,
                prepare_particle_bind_group.in_set(RenderSet::PrepareBindGroups),
            )
            .add_systems(
                Render,
                prepare_particle_view_bind_groups.in_set(RenderSet::PrepareBindGroups),
            );
    }
}
