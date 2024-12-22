use draw::DrawState;
use glam::{Vec2, Vec4};

mod draw;

fn main() {
    let mut state = DrawState::new("output/test.mp4", 1280, 720, 24);

    let mut i = 0.0;
    while i < 1.0 {
        i += 0.01;

        state.circle(Vec2::new(i, 0.5), 5.0, Vec4::new(0.1, 0.4, 0.8, 0.75));
        state.next();
    }

    state.finish();
}
