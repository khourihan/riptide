use bevy::{diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin}, prelude::*};

const FPS_OVERLAY_ZINDEX: i32 = i32::MAX - 32;

#[derive(Default)]
pub struct FpsOverlayPlugin {
    pub config: FpsOverlayConfig,
}

impl Plugin for FpsOverlayPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<FrameTimeDiagnosticsPlugin>() {
            app.add_plugins(FrameTimeDiagnosticsPlugin);
        }

        app.insert_resource(self.config.clone())
            .add_systems(Startup, setup)
            .add_systems(
                Update, 
                (
                    (customize_text, toggle_display).run_if(resource_changed::<FpsOverlayConfig>),
                    update_text,
                )
            );
    }
}

#[derive(Resource, Clone)]
pub struct FpsOverlayConfig {
    pub text_config: TextFont,
    pub text_color: Color,
    pub enabled: bool,
}

impl Default for FpsOverlayConfig {
    fn default() -> Self {
        FpsOverlayConfig {
            text_config: TextFont {
                font: Handle::<Font>::default(),
                font_size: 32.0,
                ..Default::default()
            },
            text_color: Color::WHITE,
            enabled: true,
        }
    }
}

#[derive(Component)]
struct FpsText;

fn setup(mut commands: Commands, overlay_config: Res<FpsOverlayConfig>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                ..Default::default()
            },
            GlobalZIndex(FPS_OVERLAY_ZINDEX),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("FPS: "),
                overlay_config.text_config.clone(),
                TextColor(overlay_config.text_color),
                FpsText,
            ))
            .with_child((TextSpan::default(), overlay_config.text_config.clone()));
        });
}

fn update_text(
    diagnostic: Res<DiagnosticsStore>,
    query: Query<Entity, With<FpsText>>,
    mut writer: TextUiWriter,
) {
    for entity in &query {
        if let Some(fps) = diagnostic.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                *writer.text(entity, 1) = format!("{value:.2}");
            }
        }
    }
}

fn customize_text(
    overlay_config: Res<FpsOverlayConfig>,
    query: Query<Entity, With<FpsText>>,
    mut writer: TextUiWriter,
) {
    for entity in &query {
        writer.for_each_font(entity, |mut font| {
            *font = overlay_config.text_config.clone();
        });
        writer.for_each_color(entity, |mut color| color.0 = overlay_config.text_color);
    }
}

fn toggle_display(
    overlay_config: Res<FpsOverlayConfig>,
    mut query: Query<&mut Visibility, With<FpsText>>,
) {
    for mut visibility in &mut query {
        visibility.set_if_neq(match overlay_config.enabled {
            true => Visibility::Visible,
            false => Visibility::Hidden,
        });
    }
}
