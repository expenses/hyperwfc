pub trait Direction: Sized + 'static {
    const ALL: &'static [Self];
    fn as_index(&self) -> usize;
    fn apply(
        &self,
        x: u32,
        y: u32,
        z: u32,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Option<(u32, u32, u32)>;
    fn opposite(&self) -> Self;
}

#[derive(Clone, Copy, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
    NegX,
    NegY,
    NegZ,
}

impl Direction for Axis {
    const ALL: &[Self] = &[
        Self::X,
        Self::Y,
        Self::Z,
        Self::NegX,
        Self::NegY,
        Self::NegZ,
    ];

    #[inline]
    fn as_index(&self) -> usize {
        *self as usize
    }

    fn opposite(&self) -> Self {
        match self {
            Self::X => Self::NegX,
            Self::Y => Self::NegY,
            Self::Z => Self::NegZ,
            Self::NegX => Self::X,
            Self::NegY => Self::Y,
            Self::NegZ => Self::Z,
        }
    }

    #[inline]
    fn apply(
        &self,
        mut x: u32,
        mut y: u32,
        mut z: u32,
        width: u32,
        height: u32,
        depth: u32,
    ) -> Option<(u32, u32, u32)> {
        match self {
            Self::X if x < width - 1 => x += 1,
            Self::Y if y < height - 1 => y += 1,
            Self::Z if z < depth - 1 => z += 1,
            Self::NegX if x > 0 => x -= 1,
            Self::NegY if y > 0 => y -= 1,
            Self::NegZ if z > 0 => z -= 1,
            _ => return None,
        }

        Some((x, y, z))
    }
}

#[derive(Clone, Copy)]
pub enum Axis2D {
    X,
    Y,
    NegX,
    NegY,
}

impl Direction for Axis2D {
    const ALL: &[Self] = &[Self::X, Self::Y, Self::NegX, Self::NegY];

    fn as_index(&self) -> usize {
        *self as usize
    }

    fn opposite(&self) -> Self {
        match self {
            Self::X => Self::NegX,
            Self::Y => Self::NegY,
            Self::NegX => Self::X,
            Self::NegY => Self::Y,
        }
    }

    #[inline]
    fn apply(
        &self,
        mut x: u32,
        mut y: u32,
        z: u32,
        width: u32,
        height: u32,
        _depth: u32,
    ) -> Option<(u32, u32, u32)> {
        match self {
            Self::X if x < width - 1 => x += 1,
            Self::Y if y < height - 1 => y += 1,
            Self::NegX if x > 0 => x -= 1,
            Self::NegY if y > 0 => y -= 1,
            _ => return None,
        }

        Some((x, y, z))
    }
}
