#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Array2D<T = Vec<u8>> {
    pub inner: T,
    width: usize,
    height: usize,
    depth: usize,
}

impl<T> Array2D<T> {
    pub fn new_from(array: T, width: usize, height: usize, depth: usize) -> Self {
        Self {
            inner: array,
            width: width,
            height: height,
            depth,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn shape_is_inbounds(
        &self,
        index: usize,
        width: usize,
        height: usize,
        depth: usize,
    ) -> bool {
        let (x, y, z) = decompose(index, self.width(), self.height());
        (x + width) <= self.width() && (y + height) <= self.height() && (z + depth) <= self.depth()
    }
}

impl Array2D<Vec<u8>> {
    pub fn new(slice: &[u8], width: usize, height: usize) -> Self {
        Self {
            inner: slice.to_owned(),
            width,
            height,
            depth: slice.len() / width / height,
        }
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> u8 {
        let index = compose(x, y, z, self.width, self.height);
        self.inner[index]
    }

    pub fn layers(&self) -> impl Iterator<Item = &[u8]> {
        self.inner.chunks_exact(self.width * self.height)
    }

    pub fn rows(&self) -> impl Iterator<Item = (usize, &[u8])> {
        self.layers()
            .enumerate()
            .flat_map(|(z, layer)| layer.chunks_exact(self.width).map(move |row| (z, row)))
    }

    #[inline]
    pub fn values(&self) -> impl Iterator<Item = (usize, usize, usize, u8)> + '_ {
        self.layers()
            .enumerate()
            .flat_map(|(z, layer)| {
                layer
                    .chunks_exact(self.width)
                    .enumerate()
                    .map(move |(y, row)| (y, z, row))
            })
            .flat_map(|(y, z, row)| row.iter().enumerate().map(move |(x, v)| (x, y, z, *v)))
    }

    #[inline]
    pub fn non_wildcard_values(&self) -> impl Iterator<Item = (usize, usize, usize, u8)> + '_ {
        self.values()
            .filter(|&(_, _, _, value)| value != crate::WILDCARD)
    }

    #[inline]
    pub fn non_wildcard_values_in_state(
        &self,
        state_width: usize,
        state_height: usize,
    ) -> impl Iterator<Item = (usize, u8)> + '_ {
        self.non_wildcard_values()
            .map(move |(x, y, z, value)| (compose(x, y, z, state_width, state_height), value))
    }

    pub fn permute<F: Fn(usize, usize, usize) -> (usize, usize, usize)>(
        &self,
        width: usize,
        height: usize,
        depth: usize,
        remap: F,
    ) -> Array2D {
        let mut array = Array2D {
            inner: vec![0; width * height * depth],
            width,
            height,
            depth,
        };

        for (x, y, z, value) in self.values() {
            let (x, y, z) = remap(x, y, z);
            array.inner[compose(x, y, z, width, height)] = value;
        }

        array
    }
}

pub fn compose(x: usize, y: usize, z: usize, width: usize, height: usize) -> usize {
    x + y * width + z * width * height
}

pub fn decompose(index: usize, width: usize, height: usize) -> (usize, usize, usize) {
    (
        index % width,
        (index / width) % height,
        index / width / height,
    )
}

impl Array2D<&mut [u8]> {
    pub fn put(&mut self, index: usize, value: u8) {
        self.inner[index] = value;
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ArrayPair {
    pub to: Array2D,
    pub from: Array2D,
}

impl ArrayPair {
    pub fn permute<F: Fn(usize, usize, usize) -> (usize, usize, usize)>(
        &self,
        width: usize,
        height: usize,
        depth: usize,
        remap: F,
    ) -> Self {
        Self {
            to: self.to.permute(width, height, depth, &remap),
            from: self.from.permute(width, height, depth, &remap),
        }
    }
}
