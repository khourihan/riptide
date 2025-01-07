use std::mem;

use bevy::{
    core_pipeline::core_3d::{Transparent3d, CORE_3D_DEPTH_FORMAT}, ecs::{query::ROQueryItem, system::{lifetimeless::{Read, SRes}, SystemParamItem, SystemState}}, pbr::RenderMeshInstances, prelude::*, render::{
        mesh::{allocator::MeshAllocator, MeshVertexBufferLayoutRef, PrimitiveTopology, RenderMesh, RenderMeshBufferInfo, VertexBufferLayout}, render_asset::RenderAssets, render_phase::{
            DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases
        }, render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingType, BufferBindingType, BufferInitDescriptor, BufferUsages, ColorTargetState, ColorWrites, CompareFunction, DepthStencilState, Face, FragmentState, FrontFace, MultisampleState, PipelineCache, PolygonMode, PrimitiveState, RenderPipelineDescriptor, ShaderStages, ShaderType, SpecializedMeshPipeline, SpecializedMeshPipelineError, SpecializedMeshPipelines, TextureFormat, VertexAttribute, VertexFormat, VertexState, VertexStepMode
        }, renderer::RenderDevice, sync_world::MainEntity, view::{ExtractedView, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms}
    }
};

use crate::particles_3d::{InstanceData, PARTICLE_SHADER_HANDLE};

use super::{InstanceBuffer, InstanceParticleData, Particle3dDepth, Particle3dLockAxis};

#[derive(Clone, Copy, Component, Debug)]
pub struct RenderParticle {
    pub depth: Particle3dDepth,
    pub lock_axis: Option<Particle3dLockAxis>,
}

#[derive(Component)]
pub struct ParticleViewBindGroup(BindGroup);

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    #[repr(transparent)]
    pub struct ParticlePipelineKey: u32 {
        const TEXT = 0;
        const TEXTURE = 1 << 0;
        const DEPTH = 1 << 1;
        const LOCK_Y = 1 << 2;
        const LOCK_ROTATION = 1 << 3;
        const HDR = 1 << 4;
        const MSAA_RESERVED_BITS = Self::MSAA_MASK_BITS << Self::MSAA_SHIFT_BITS;
    }
}

impl ParticlePipelineKey {
    const MSAA_MASK_BITS: u32 = 0b111;
    const MSAA_SHIFT_BITS: u32 = 32 - Self::MSAA_MASK_BITS.count_ones();

    pub fn from_msaa_samples(msaa_samples: u32) -> Self {
        let msaa_bits = (msaa_samples.trailing_zeros() & Self::MSAA_MASK_BITS) << Self::MSAA_SHIFT_BITS;

        Self::from_bits_retain(msaa_bits)
    }

    pub fn msaa_samples(&self) -> u32 {
        1 << ((self.bits() >> Self::MSAA_SHIFT_BITS) & Self::MSAA_MASK_BITS)
    }
}

pub fn prepare_particle_view_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    particle_pipeline: Res<ParticlePipeline>,
    view_uniforms: Res<ViewUniforms>,
    views: Query<Entity, With<ExtractedView>>,
) {
    let Some(binding) = view_uniforms.uniforms.binding() else {
        return;
    };

    for entity in views.iter() {
        commands.entity(entity).insert(ParticleViewBindGroup(
            render_device.create_bind_group(
                Some("particle_view_bind_group"),
                &particle_pipeline.view_layout,
                &[BindGroupEntry {
                    binding: 0,
                    resource: binding.clone(),
                }],
            ),
        ));
    }
}

pub fn prepare_instance_buffers(
    mut commands: Commands,
    query: Query<(Entity, &InstanceParticleData)>,
    render_device: Res<RenderDevice>,
) {
    for (entity, instance_data) in &query {
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("instance_data_buffer"),
            contents: bytemuck::cast_slice(instance_data.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        commands.entity(entity).insert(InstanceBuffer {
            buffer,
            length: instance_data.len(),
        });
    }
}

pub fn queue_particles(
    mut views: Query<(Entity, &ExtractedView, &Msaa)>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    pipeline_cache: ResMut<PipelineCache>,
    mut particle_pipelines: ResMut<SpecializedMeshPipelines<ParticlePipeline>>,
    transparent_draw_functions: Res<DrawFunctions<Transparent3d>>,
    particle_pipeline: Res<ParticlePipeline>,
    render_mesh_instances: Res<RenderMeshInstances>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    particle_meshes: Query<(Entity, &MainEntity), With<InstanceParticleData>>,
    particles: Query<&RenderParticle>
) {
    let draw_transparent_particle = transparent_draw_functions
        .read()
        .id::<DrawParticle>();

    for (view_entity, view, msaa) in &mut views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view_entity) else {
            continue;
        };

        let rangefinder = view.rangefinder3d();

        for (entity, main_entity) in &particle_meshes {
            let Ok(particle) = particles.get(entity) else {
                continue;
            };

            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*main_entity) else {
                continue;
            };

            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            let mut key = ParticlePipelineKey::from_msaa_samples(msaa.samples());

            if particle.depth.0 {
                key |= ParticlePipelineKey::DEPTH;
            }

            if particle.lock_axis.map_or(false, |lock| lock.y_axis) {
                key |= ParticlePipelineKey::LOCK_Y;
            }

            if particle.lock_axis.map_or(false, |lock| lock.rotation) {
                key |= ParticlePipelineKey::LOCK_ROTATION;
            }

            if view.hdr {
                key |= ParticlePipelineKey::HDR;
            }

            let pipeline_id = particle_pipelines.specialize(
                &pipeline_cache,
                &particle_pipeline,
                key,
                &mesh.layout,
            );

            let pipeline_id = match pipeline_id {
                Ok(id) => id,
                Err(err) => {
                    error!("{err:?}");
                    continue;
                },
            };

            transparent_phase.add(Transparent3d {
                entity: (entity, *main_entity),
                pipeline: pipeline_id,
                draw_function: draw_transparent_particle,
                distance: rangefinder.distance_translation(&mesh_instance.translation),
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::NONE,
            });
        }
    }
}

#[derive(Resource, Clone)]
pub struct ParticlePipeline {
    view_layout: BindGroupLayout,
}

impl FromWorld for ParticlePipeline {
    fn from_world(world: &mut World) -> Self {
        let mut system_state: SystemState<(Res<RenderDevice>,)> = SystemState::new(world);

        let (render_device,) = system_state.get(world);

        let view_layout = render_device.create_bind_group_layout(
            "particle_view_layout",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(ViewUniform::min_size())
                },
                count: None,
            }],
        );

        Self {
            view_layout,
        }
    }
}

impl SpecializedMeshPipeline for ParticlePipeline {
    type Key = ParticlePipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        const DEF_LOCK_Y: &str = "LOCK_Y";
        const DEF_LOCK_ROTATION: &str = "LOCK_ROTATION";

        let mut shader_defs = Vec::with_capacity(4);
        let mut attributes = Vec::with_capacity(4);

        attributes.push(Mesh::ATTRIBUTE_POSITION.at_shader_location(0));
        attributes.push(Mesh::ATTRIBUTE_UV_0.at_shader_location(1));

        let layout = layout.0.as_ref();

        let vertex_buffer_layout = layout.get_layout(&attributes)?;

        let instance_buffer_layout = VertexBufferLayout {
            array_stride: mem::size_of::<InstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 2,
                },
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32x4.size(),
                    shader_location: 3,
                },
            ],
        };

        let depth_compare = if key.contains(ParticlePipelineKey::DEPTH) {
            CompareFunction::GreaterEqual
        } else {
            CompareFunction::Always
        };

        if key.contains(ParticlePipelineKey::LOCK_Y) {
            shader_defs.push(DEF_LOCK_Y.into());
        }

        if key.contains(ParticlePipelineKey::LOCK_ROTATION) {
            shader_defs.push(DEF_LOCK_ROTATION.into());
        }

        Ok(RenderPipelineDescriptor {
            label: Some("particle_pipeline".into()),
            layout: vec![
                self.view_layout.clone(),
            ],
            vertex: VertexState {
                shader: PARTICLE_SHADER_HANDLE,
                entry_point: "vertex".into(),
                buffers: vec![vertex_buffer_layout, instance_buffer_layout],
                shader_defs: shader_defs.clone(),
            },
            fragment: Some(FragmentState {
                shader: PARTICLE_SHADER_HANDLE,
                entry_point: "fragment".into(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: if key.contains(ParticlePipelineKey::HDR) {
                        ViewTarget::TEXTURE_FORMAT_HDR
                    } else {
                        TextureFormat::bevy_default()
                    },
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare,
                stencil: default(),
                bias: default(),
            }),
            multisample: MultisampleState {
                count: key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        })
    }
}

pub struct SetParticleViewBindGroup<const I: usize>;
impl<const I: usize> RenderCommand<Transparent3d> for SetParticleViewBindGroup<I> {
    type Param = ();
    type ViewQuery = (Read<ViewUniformOffset>, Read<ParticleViewBindGroup>);
    type ItemQuery = ();

    fn render<'w>(
        _item: &Transparent3d,
        (view_uniform, particle_mesh_bind_group): ROQueryItem<'w, Self::ViewQuery>,
        _entity: Option<ROQueryItem<'w, Self::ItemQuery>>,
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &particle_mesh_bind_group.0, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}

pub struct DrawParticleMesh;
impl<P: PhaseItem> RenderCommand<P> for DrawParticleMesh {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
    );
    type ViewQuery = ();
    type ItemQuery = Read<InstanceBuffer>;

    fn render<'w>(
        item: &P,
        _view: ROQueryItem<'w, Self::ViewQuery>,
        instance_buffer: Option<ROQueryItem<'w, Self::ItemQuery>>,
        (meshes, mesh_instances, mesh_allocator): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mesh_instances = mesh_instances.into_inner();
        let Some(mesh_instance) = mesh_instances.render_mesh_queue_data(item.main_entity()) else {
            return RenderCommandResult::Skip;
        };

        let meshes = meshes.into_inner();
        let Some(render_mesh) = meshes.get(mesh_instance.mesh_asset_id) else {
            return RenderCommandResult::Skip;
        };

        let Some(instance_buffer) = instance_buffer else {
            return RenderCommandResult::Skip;
        };

        let mesh_allocator = mesh_allocator.into_inner();
        let Some(vertex_buffer_slice) = mesh_allocator.mesh_vertex_slice(&mesh_instance.mesh_asset_id) else {
            return RenderCommandResult::Skip;
        };

        pass.set_vertex_buffer(0, vertex_buffer_slice.buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));

        match &render_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(index_buffer_slice) = mesh_allocator.mesh_index_slice(&mesh_instance.mesh_asset_id) else {
                    return RenderCommandResult::Skip;
                };

                pass.set_index_buffer(index_buffer_slice.buffer.slice(..), 0, *index_format);
                pass.draw_indexed(
                    index_buffer_slice.range.start..(index_buffer_slice.range.start + count),
                    vertex_buffer_slice.range.start as i32,
                    0..instance_buffer.length as u32,
                );
            },
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(
                    vertex_buffer_slice.range,
                    0..instance_buffer.length as u32,
                );
            },
        }

        RenderCommandResult::Success
    }
}

pub type DrawParticle = (
    SetItemPipeline,
    SetParticleViewBindGroup<0>,
    DrawParticleMesh,
);
