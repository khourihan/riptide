use bevy::prelude::*;
use riptide_io::decode::FluidData as IoFluidData;

pub mod d2;
pub mod d3;

struct PlaybackPlugin;

impl Plugin for PlaybackPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_state::<PlaybackState>()
            .init_state::<SetupState>()
            .init_resource::<Particles>()
            .init_resource::<FluidDataFrame>()
            .add_systems(Update, (
                change_state_playing.run_if(in_state(PlaybackState::Paused)),
                change_state_paused.run_if(in_state(PlaybackState::Playing)),
            ));
    }
}


#[derive(Resource)]
pub struct FluidData(pub IoFluidData);

#[derive(Resource, Default)]
struct FluidDataFrame(usize);

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

#[derive(Resource, Default)]
struct Particles(Vec<Entity>);

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
