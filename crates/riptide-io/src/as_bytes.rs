use glam::{Vec2, Vec3, Vec4};

pub trait AsBytes<const N: usize> {
    fn from_bytes(b: [u8; N]) -> Self;

    fn to_bytes(self) -> [u8; N];
}

impl AsBytes<4> for f32 {
    fn from_bytes(b: [u8; 4]) -> Self {
        f32::from_ne_bytes(b)
    }

    fn to_bytes(self) -> [u8; 4] {
        self.to_ne_bytes()
    }
}

impl AsBytes<8> for Vec2 {
    fn from_bytes(b: [u8; 8]) -> Self {
        Vec2::new(
            f32::from_bytes(b[0..4].try_into().unwrap()),
            f32::from_bytes(b[4..8].try_into().unwrap()),
        )
    }

    fn to_bytes(self) -> [u8; 8] {
        [self.x.to_bytes(), self.y.to_bytes()].concat().try_into().unwrap()
    }
}

impl AsBytes<12> for Vec3 {
    fn from_bytes(b: [u8; 12]) -> Self {
        Vec3::new(
            f32::from_bytes(b[0..4].try_into().unwrap()),
            f32::from_bytes(b[4..8].try_into().unwrap()),
            f32::from_bytes(b[8..12].try_into().unwrap()),
        )
    }

    fn to_bytes(self) -> [u8; 12] {
        [self.x.to_bytes(), self.y.to_bytes(), self.z.to_bytes()].concat().try_into().unwrap()
    }
}

impl AsBytes<16> for Vec4 {
    fn from_bytes(b: [u8; 16]) -> Self {
        Vec4::new(
            f32::from_bytes(b[0..4].try_into().unwrap()),
            f32::from_bytes(b[4..8].try_into().unwrap()),
            f32::from_bytes(b[8..12].try_into().unwrap()),
            f32::from_bytes(b[12..16].try_into().unwrap()),
        )
    }

    fn to_bytes(self) -> [u8; 16] {
        [self.x.to_bytes(), self.y.to_bytes(), self.z.to_bytes(), self.w.to_bytes()].concat().try_into().unwrap()
    }
}
