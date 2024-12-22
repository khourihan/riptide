use glam::{UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};
use ndarray::Array3;
use video_rs::{encode::Settings, Encoder, Time};

pub struct DrawState {
    encoder: Encoder,
    size: UVec2,
    frame: Array3<u8>,
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
            size: UVec2::new(width as u32, height as u32),
            frame: Array3::from_elem((height, width, 3), 0),
            position: Time::zero(),
            fps: Time::from_nth_of_a_second(fps),
        }
    }

    /// Advance the encoder to the next frame and clear the screen.
    pub fn next(&mut self) {
        self.encoder.encode(&self.frame, self.position).unwrap();
        self.position = self.position.aligned_with(self.fps).add();
        self.frame.fill(0);
    }

    pub fn finish(mut self) {
        self.encoder.finish().unwrap()
    }

    /// Draw a point to the current frame at the given position `p` with a given `color`.
    pub fn point(&mut self, p: UVec2, color: Vec4) {
        for c in 0..3 {
            if let Some(col) = self.frame.get_mut((p.y as usize, p.x as usize, c)) {
                let v = *col as f32 / 255.0;
                *col = ((v * (1.0 - color.w) + (color[c] - v) * color.w).clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }

    /// Draw a circle to the current frame at the given `uv` position, with the given `radius` and
    /// `color`.
    pub fn circle(&mut self, uv: Vec2, radius: f32, color: Vec4) {
        let center = uv * self.size.as_vec2();

        for x in (center.x - radius).floor() as i32 - 1..(center.x + radius).ceil() as i32 + 1 {
            for y in (center.y - radius).floor() as i32 - 1..(center.y + radius).ceil() as i32 + 1 {
                if x < 0 || y < 0 || x >= self.size.x as i32 || y >= self.size.y as i32 {
                    continue;
                }

                let pos = Vec2::new(x as f32, y as f32) - center;

                const PIXEL_SIZE: f32 = std::f32::consts::SQRT_2;
                const SMOOTHING: f32 = 1.0;
                let d = pos.length() - radius;

                let alpha = (1.0 - d) + (1.0 - PIXEL_SIZE * SMOOTHING) * d;

                self.point(UVec2::new(x as u32, y as u32), Vec4::new(color.x, color.y, color.z, alpha * color.w));
            }
        }
    }
}
