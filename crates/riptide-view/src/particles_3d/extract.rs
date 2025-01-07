use bevy::{prelude::*, render::{sync_world::RenderEntity, Extract}};

use super::{
    pipeline::RenderParticle,
    Particle3dDepth, Particle3dLockAxis,
};

pub fn extract_particles(
    mut commands: Commands,
    mut previous_len: Local<usize>,
    particle_query: Extract<
        Query<(
            &RenderEntity,
            &ViewVisibility,
            &Particle3dDepth,
            Option<&Particle3dLockAxis>,
        )>
    >,
) {
    let mut batch = Vec::with_capacity(*previous_len);

    for (
        render_entity,
        visibility,
        &depth,
        lock_axis
    ) in &particle_query {
        if !visibility.get() {
            continue;
        }
        
        batch.push((
            render_entity.id(),
            (
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
