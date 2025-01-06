use std::{fs::File, path::PathBuf};

use draw::DrawState;
use glam::{Vec2, Vec4};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use riptide_io::decode::FluidDataDecoder;

pub mod draw;

pub fn render_2d(
    datfile: PathBuf,
    outfile: PathBuf,
    resolution: [usize; 2],
) {
    let fps = 60;
    let [_, screen_height] = resolution;

    let mut decoder = FluidDataDecoder::new(File::open(datfile).unwrap());
    let data = decoder.decode().unwrap();
    let size: Vec2 = data.size().into();

    let screen_width = (screen_height as f32 * size.x / size.y) as usize;

    let mut state = DrawState::new(outfile, screen_width, screen_height, fps);

    let bar_template = "Rendering {spinner:.green} [{elapsed}] [{bar:50.white/white}] {pos}/{len} ({eta})";
    let style = ProgressStyle::with_template(bar_template).unwrap()
        .progress_chars("=> ").tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");
    let progress = ProgressBar::new(data.frames.len() as u64).with_style(style);

    for frame in data.frames.into_iter().progress_with(progress) {
        for pos in frame.positions.iter::<2>() {
            let mut p = Vec2::from(pos) / size;
            p.y = 1.0 - p.y;

            let col = Vec4::new(0.0, 0.0, 1.0, 1.0).lerp(Vec4::new(1.0, 1.0, 1.0, 1.0), 0.0);
            state.circle(p, 0.2 / 100.0 * screen_height as f32, col);
        }

        state.next();
    }

    state.finish();
}
