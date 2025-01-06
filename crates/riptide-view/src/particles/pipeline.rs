use bevy::{core_pipeline::core_3d::Transparent3d, ecs::{query::ROQueryItem, system::{lifetimeless::{Read, SRes}, SystemParamItem, SystemState}}, prelude::*, render::{extract_component::{ComponentUniforms, DynamicUniformIndex}, mesh::{allocator::MeshAllocator, MeshVertexBufferLayoutRef, PrimitiveTopology, RenderMesh, RenderMeshBufferInfo}, render_asset::RenderAssets, render_phase::{DrawFunctions, PhaseItemExtraIndex, RenderCommand, RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases}, render_resource::{BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType, BlendComponent, BlendFactor, BlendOperation, BlendState, BufferBindingType, ColorTargetState, ColorWrites, CompareFunction, DepthStencilState, FragmentState, FrontFace, MultisampleState, PipelineCache, PolygonMode, PrimitiveState, RenderPipelineDescriptor, ShaderStages, ShaderType, SpecializedMeshPipeline, SpecializedMeshPipelineError, SpecializedMeshPipelines, TextureFormat, VertexState}, renderer::RenderDevice, view::{ExtractedView, RenderVisibleEntities, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms}}, utils::HashMap};

use crate::particles::PARTICLE_SHADER_HANDLE;

use super::{Particle, ParticleDepth, ParticleLockAxis};

#[derive(Clone, Copy, ShaderType, Component)]
pub struct ParticleUniform {
    pub(crate) transform: Mat4,
    pub(crate) color: LinearRgba,
}

#[derive(Clone, Copy, Component, Debug)]
pub struct RenderParticleMesh {
    pub id: AssetId<Mesh>,
}

#[derive(Clone, Copy, Component, Debug)]
pub struct RenderParticle {
    pub depth: ParticleDepth,
    pub lock_axis: Option<ParticleLockAxis>,
}

#[derive(Resource)]
pub struct ParticleBindGroup(BindGroup);

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

pub fn prepare_particle_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    particle_pipeline: Res<ParticlePipeline>,
    particle_uniforms_buffer: Res<ComponentUniforms<ParticleUniform>>,
) {
    let Some(binding) = particle_uniforms_buffer.uniforms().binding() else {
        return;
    };

    commands.insert_resource(ParticleBindGroup(
        render_device.create_bind_group(
            Some("particle_bind_group"),
            &particle_pipeline.particle_layout,
            &[BindGroupEntry {
                binding: 0,
                resource: binding,
            }],
        ),
    ));
}

#[allow(clippy::too_many_arguments)]
pub fn queue_particles(
    mut views: Query<(Entity, &ExtractedView, &RenderVisibleEntities, &Msaa)>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    mut pipeline_cache: ResMut<PipelineCache>,
    mut particle_pipelines: ResMut<SpecializedMeshPipelines<ParticlePipeline>>,
    transparent_draw_functions: Res<DrawFunctions<Transparent3d>>,
    particle_pipeline: Res<ParticlePipeline>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    particles: Query<(
        &ParticleUniform,
        &RenderParticleMesh,
        &RenderParticle,
    )>
) {
    for (view_entity, view, visible_entities, msaa) in &mut views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view_entity) else {
            continue;
        };

        let draw_transparent_particle = transparent_draw_functions
            .read()
            .get_id::<DrawParticle>()
            .unwrap();

        let rangefinder = view.rangefinder3d();

        for visible_entity in visible_entities.iter::<With<Particle>>() {
            let Ok((uniform, mesh, particle)) = particles.get(visible_entity.0) else {
                continue;
            };

            let Some(render_mesh) = render_meshes.get(mesh.id) else {
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
                &mut pipeline_cache,
                &particle_pipeline,
                key,
                &render_mesh.layout,
            );

            let pipeline_id = match pipeline_id {
                Ok(id) => id,
                Err(err) => {
                    error!("{err:?}");
                    continue;
                },
            };

            let distance = rangefinder.distance(&uniform.transform);

            transparent_phase.add(Transparent3d {
                pipeline: pipeline_id,
                entity: *visible_entity,
                draw_function: draw_transparent_particle,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::NONE,
                distance,
            });
        }
    }
}

#[derive(Resource, Clone)]
pub struct ParticlePipeline {
    view_layout: BindGroupLayout,
    particle_layout: BindGroupLayout,
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

        let particle_layout = render_device.create_bind_group_layout(
            "particle_layout",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(ParticleUniform::min_size()),
                },
                count: None,
            }],
        );

        Self {
            view_layout,
            particle_layout,
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
        const DEF_VERTEX_COLOR: &str = "VERTEX_COLOR";
        const DEF_LOCK_Y: &str = "LOCK_Y";
        const DEF_LOCK_ROTATION: &str = "LOCK_ROTATION";

        let mut shader_defs = Vec::with_capacity(4);
        let mut attributes = Vec::with_capacity(4);

        attributes.push(Mesh::ATTRIBUTE_POSITION.at_shader_location(0));
        attributes.push(Mesh::ATTRIBUTE_UV_0.at_shader_location(1));

        let layout = layout.0.as_ref();

        if layout.contains(Mesh::ATTRIBUTE_COLOR) {
            shader_defs.push(DEF_VERTEX_COLOR.into());
            attributes.push(Mesh::ATTRIBUTE_COLOR.at_shader_location(2));
        }

        let vertex_buffer_layout = layout.get_layout(&attributes)?;

        let depth_compare = if key.contains(ParticlePipelineKey::DEPTH) {
            CompareFunction::Greater
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
                self.particle_layout.clone(),
            ],
            vertex: VertexState {
                shader: PARTICLE_SHADER_HANDLE,
                entry_point: "vertex".into(),
                buffers: vec![vertex_buffer_layout],
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
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::SrcAlpha,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
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

pub struct SetParticleBindGroup<const I: usize>;
impl<const I: usize> RenderCommand<Transparent3d> for SetParticleBindGroup<I> {
    type Param = SRes<ParticleBindGroup>;
    type ViewQuery = ();
    type ItemQuery = Read<DynamicUniformIndex<ParticleUniform>>;

    fn render<'w>(
        _item: &Transparent3d,
        _view: ROQueryItem<'w, Self::ViewQuery>,
        particle_index: Option<ROQueryItem<'w, Self::ItemQuery>>,
        particle_bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let particle_bind_group = particle_bind_group.into_inner();

        let Some(particle_index) = particle_index else {
            return RenderCommandResult::Skip;
        };

        pass.set_bind_group(I, &particle_bind_group.0, &[particle_index.index()]);

        RenderCommandResult::Success
    }
}

pub struct DrawParticleMesh;
impl RenderCommand<Transparent3d> for DrawParticleMesh {
    type Param = (SRes<RenderAssets<RenderMesh>>, SRes<MeshAllocator>);
    type ViewQuery = ();
    type ItemQuery = Read<RenderParticleMesh>;

    fn render<'w>(
        _item: &Transparent3d,
        _view: ROQueryItem<'w, Self::ViewQuery>,
        mesh: Option<ROQueryItem<'w, Self::ItemQuery>>,
        (meshes, mesh_allocator): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(mesh) = mesh else {
            return RenderCommandResult::Skip;
        };

        let Some(render_mesh) = meshes.into_inner().get(mesh.id) else {
            return RenderCommandResult::Skip;
        };

        let mesh_allocator = mesh_allocator.into_inner();
        let Some(vertex_buffer_slice) = mesh_allocator.mesh_vertex_slice(&mesh.id) else {
            return RenderCommandResult::Skip;
        };

        pass.set_vertex_buffer(0, vertex_buffer_slice.buffer.slice(..));

        match &render_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                count,
                index_format,
            } => {
                let Some(index_buffer_slice) = mesh_allocator.mesh_index_slice(&mesh.id) else {
                    return RenderCommandResult::Skip;
                };

                pass.set_index_buffer(index_buffer_slice.buffer.slice(..), 0, *index_format);
                pass.draw_indexed(
                    index_buffer_slice.range.start..(index_buffer_slice.range.start + count),
                    vertex_buffer_slice.range.start as i32,
                    0..1,
                );
            },
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(
                    vertex_buffer_slice.range.start
                        ..(vertex_buffer_slice.range.start + render_mesh.vertex_count),
                    0..1,
                );
            },
        }

        RenderCommandResult::Success
    }
}

pub type DrawParticle = (
    SetItemPipeline,
    SetParticleViewBindGroup<0>,
    SetParticleBindGroup<1>,
    DrawParticleMesh,
);
