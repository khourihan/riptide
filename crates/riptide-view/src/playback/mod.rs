use bevy::prelude::*;
use riptide_io::decode::{FluidDataDecoder as IoFluidDecoder, FluidMetadata as IoFluidMetadata};

pub mod d2;
pub mod d3;

struct PlaybackPlugin;

impl Plugin for PlaybackPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_state::<PlaybackState>()
            .init_state::<SetupState>()
            .add_systems(Update, (
                change_state_playing.run_if(in_state(PlaybackState::Paused)),
                change_state_paused.run_if(in_state(PlaybackState::Playing)),
            ));
    }
}


#[derive(Resource)]
pub struct FluidDataDecoder(pub IoFluidDecoder);

#[derive(Resource)]
pub struct FluidMetadata(pub IoFluidMetadata);

#[derive(States, Clone, PartialEq, Eq, Hash, Debug, Default)]
enum PlaybackState {
    Playing,
    #[default]
    Paused
}

#[derive(States, Clone, PartialEq, Eq, Hash, Debug, Default)]
enum SetupState {
    Ready,
    #[default]
    NotReady,
}

fn change_state_playing(
    keys: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<PlaybackState>>,
) {
    if keys.just_pressed(KeyCode::Space) {
        next_state.set(PlaybackState::Playing);
    }
}

fn change_state_paused(
    keys: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<PlaybackState>>,
) {
    if keys.just_pressed(KeyCode::Space) {
        next_state.set(PlaybackState::Paused);
    }
}
