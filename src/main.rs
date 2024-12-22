use draw::DrawState;

mod draw;

fn main() {
    let mut state = DrawState::new("output/test.mp4", 1280, 720, 24);

    for _ in 0..256 {
        state.draw_next();
    }

    state.finish();
}
