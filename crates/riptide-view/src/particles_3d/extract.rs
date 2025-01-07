use bevy::{prelude::*, render::{sync_world::RenderEntity, Extract}};

use super::{
    pipeline::{ParticleUniform, RenderParticle, RenderParticleMesh},
    Particle3dColor, Particle3dDepth, Particle3dLockAxis, Particle3dMesh,
};

pub fn extract_particles(
    mut commands: Commands,
    mut previous_len: Local<usize>,
    particle_query: Extract<
        Query<(
            &RenderEntity,
            &ViewVisibility,
            &GlobalTransform,
            &Transform,
            &Particle3dMesh,
            &Particle3dColor,
            &Particle3dDepth,
            Option<&Particle3dLockAxis>,
        )>
    >,
) {
    let mut batch = Vec::with_capacity(*previous_len);

    for (
        render_entity,
        visibility,
        global_transform,
        transform,
        particle_mesh,
        particle_color,
        &depth,
        lock_axis
    ) in &particle_query {
        if !visibility.get() {
            continue;
        }
        
        let uniform = calculate_particle_uniform(global_transform, transform, lock_axis, particle_color);

        batch.push((
            render_entity.id(),
            (
                uniform,
                RenderParticleMesh {
                    id: particle_mesh.0.id(),
                },
                RenderParticle {
                    depth,
                    lock_axis: lock_axis.copied(),
                },
            ),
        ));
    }

    *previous_len = batch.len();
    commands.insert_batch(batch);
}

fn calculate_particle_uniform(
    global_transform: &GlobalTransform,
    transform: &Transform,
    lock_axis: Option<&Particle3dLockAxis>,
    color: &Particle3dColor,
) -> ParticleUniform {
    let transform = if lock_axis.is_some() {
        global_transform.compute_matrix()
    } else {
        let global_matrix = global_transform.compute_matrix();
        Mat4::from_cols(
            Mat4::IDENTITY.x_axis * transform.scale.x,
            Mat4::IDENTITY.y_axis * transform.scale.y,
            Mat4::IDENTITY.z_axis * transform.scale.z,
            global_matrix.w_axis,
        )
    };

    ParticleUniform {
        transform,
        color: color.0.to_linear(),
    }
}
