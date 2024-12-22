use ndarray::Array3;
use video_rs::{encode::Settings, Encoder, Time};

pub struct DrawState {
    encoder: Encoder,
    width: usize,
    height: usize,
    frame: usize,
    position: Time,
    fps: Time,
}

impl DrawState {
    pub fn new<P: AsRef<std::path::Path>>(
        path: P,
        width: usize,
        height: usize,
        fps: usize,
    ) -> DrawState {
        let settings = Settings::preset_h264_yuv420p(width, height, false);
        let encoder = Encoder::new(path.as_ref(), settings).unwrap();

        DrawState {
            encoder,
            width,
            height,
            frame: 0,
            position: Time::zero(),
            fps: Time::from_nth_of_a_second(fps),
        }
    }

    pub fn draw_next(&mut self) {
        let col = self.frame as f32 / 256.0;
        let frame = Array3::from_shape_fn((self.height, self.width, 3), |(_y, _x, _c)| (col.min(1.0) * 255.0) as u8);

        self.encoder.encode(&frame, self.position).unwrap();
        self.position = self.position.aligned_with(self.fps).add();
        self.frame += 1;
    }

    pub fn finish(mut self) {
        self.encoder.finish().unwrap()
    } 
}
