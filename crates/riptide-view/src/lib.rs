#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use bevy::prelude::*;
use fps::FpsOverlayPlugin;
use riptide_io::decode::{FluidDataDecoder, FluidMetadata};

#[cfg(feature = "d2")]
use playback::d2::Playback2DPlugin;
#[cfg(feature = "d3")]
use playback::d3::Playback3DPlugin;

mod particles;
mod playback;
mod fps;

#[cfg(feature = "d2")]
pub fn view_2d(decoder: FluidDataDecoder, meta: FluidMetadata) {
    App::new()
        .add_plugins((
            DefaultPlugins,
            FpsOverlayPlugin::default(),
        ))
        .add_plugins(Playback2DPlugin)
        .insert_resource(playback::FluidDataDecoder(decoder))
        .insert_resource(playback::FluidMetadata(meta))
        .run();
}

#[cfg(feature = "d3")]
pub fn view_3d(decoder: FluidDataDecoder, meta: FluidMetadata) {
    App::new()
        .add_plugins((
            DefaultPlugins,
            FpsOverlayPlugin::default(),
        ))
        .add_plugins(Playback3DPlugin)
        .insert_resource(playback::FluidDataDecoder(decoder))
        .insert_resource(playback::FluidMetadata(meta))
        .run();
}
