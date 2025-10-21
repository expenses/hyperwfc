use fnv::FnvBuildHasher;
use ordered_float::OrderedFloat;
use rand::Rng;
use rand::prelude::IndexedRandom;
#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::cmp::Ord;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, binary_heap, hash_map};
use std::fmt::Debug;
use std::hash::Hash;
#[cfg(feature = "rayon")]
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

#[inline]
fn count_ones(wave: &[u64]) -> u32 {
    wave.iter().map(|section| section.count_ones()).sum()
}

#[inline]
fn trailing_zeros(wave: &[u64]) -> u32 {
    let mut sum = 0;
    for section in wave.iter() {
        let trailing_zeros = section.trailing_zeros();
        sum += trailing_zeros;
        if trailing_zeros != 64 {
            break;
        }
    }
    sum
}

#[inline]
fn wave_contains(wave: &[u64], bit: u8) -> bool {
    wave[(bit / 64) as usize] >> ((bit % 64) as usize) & 1 != 0
}

// Custom IndexSet implementation. Mirrors `indexmap::IndexSet`
// but is a bit faster.
#[derive(Clone)]
struct IndexSet<T> {
    values: Vec<T>,
    indices: hashbrown::HashTable<u32>,
}

impl<T> Default for IndexSet<T> {
    fn default() -> Self {
        Self {
            values: Default::default(),
            indices: Default::default(),
        }
    }
}

impl<T: Hash + Eq + Clone + Debug> IndexSet<T> {
    fn new_from_vec(values: Vec<T>) -> Self {
        let mut indices = hashbrown::HashTable::with_capacity(values.len());

        for (index, value) in values.iter().cloned().enumerate() {
            use std::hash::BuildHasher;
            let hasher = FnvBuildHasher::default();
            // Only used when resizing
            let hasher_fn = |_: &u32| unreachable!();
            indices.insert_unique(hasher.hash_one(&value), index as u32, hasher_fn);
        }

        Self { values, indices }
    }

    fn insert(&mut self, value: T) -> bool {
        use std::hash::BuildHasher;
        let hasher = FnvBuildHasher::default();

        let index = self.values.len() as u32;

        let hasher_fn =
            |&val: &u32| unsafe { hasher.hash_one(self.values.get_unchecked(val as usize)) };

        match self
            .indices
            .entry(hasher.hash_one(&value), |other| *other == index, hasher_fn)
        {
            hashbrown::hash_table::Entry::Vacant(slot) => {
                slot.insert(index);
                self.values.push(value);
                true
            }
            _ => false,
        }
    }

    fn swap_remove(&mut self, value: &T) -> bool {
        use std::hash::BuildHasher;
        let hasher = FnvBuildHasher::default();
        // Search in the table for the value
        let index = if let Ok(entry) = self
            .indices
            .find_entry(hasher.hash_one(value), |&other| unsafe {
                self.values.get_unchecked(other as usize) == value
            }) {
            let (index, _) = entry.remove();
            index
        } else {
            return false;
        };
        // Swap remove it.
        self.values.swap_remove(index as _);
        // If the vec still has items, we need to change it's value.
        if let Some(value) = self.values.get(index as usize) {
            use std::hash::BuildHasher;
            let hasher = FnvBuildHasher::default();
            if let Ok(entry) = self.indices.find_entry(hasher.hash_one(value), |&other| {
                other == self.values.len() as u32
            }) {
                *entry.into_mut() = index;
            }
        }
        true
    }
}

// A priority queue of items where some items have the same priority
// and can be randomly picked.
#[derive(Default, Clone)]
struct SetQueue<T, P: Ord> {
    queue: BinaryHeap<P>,
    sets: HashMap<P, IndexSet<T>, FnvBuildHasher>,
}

impl<T: Hash + Eq + Clone + Debug, P: Copy + Ord + Hash> SetQueue<T, P> {
    #[allow(unused)]
    fn clear(&mut self) {
        self.queue.clear();
        self.sets.clear();
    }

    fn insert_set(&mut self, p: P, set: IndexSet<T>) {
        self.queue.push(p);
        self.sets.insert(p, set);
    }

    fn select_from_first_set_at_random<R: Rng>(&mut self, rng: &mut R) -> Option<T> {
        while let Some(priority) = self.queue.peek_mut() {
            if let hash_map::Entry::Occupied(occupied_entry) = self.sets.entry(*priority) {
                let set = occupied_entry.get();
                if let Some(item) = set.values.choose(rng) {
                    return Some(item.clone());
                } else {
                    occupied_entry.remove();
                }
            }

            binary_heap::PeekMut::pop(priority);
        }

        None
    }

    #[inline]
    fn select_from_lowest_entropy<R: Rng, const WAVE_SIZE: usize>(
        &mut self,
        rng: &mut R,
        func: impl Fn(&T) -> [u64; WAVE_SIZE],
        probabilities: &[f32],
        tile_list: &mut Vec<u8>,
        prefix_summed_probabilities: &mut Vec<OrderedFloat<f32>>,
    ) -> Option<(T, u8)> {
        self.select_from_first_set_at_random(rng).map(|index| {
            let wave = func(&index);
            tile_list_from_wave(&wave, probabilities.len(), tile_list);
            let potential_states = tile_list;
            prefix_summed_probabilities.clear();
            let mut sum = 0.0;
            for &tile in potential_states.iter() {
                sum += probabilities[tile as usize];
                prefix_summed_probabilities.push(OrderedFloat(sum));
            }
            let list_index = sample_prefix_sum(prefix_summed_probabilities, rng);
            let tile = potential_states[list_index];

            (index, tile)
        })
    }

    fn insert(&mut self, p: P, value: T) -> bool {
        let set = match self.sets.entry(p) {
            hash_map::Entry::Occupied(set) => set.into_mut(),
            hash_map::Entry::Vacant(slot) => {
                self.queue.push(p);
                slot.insert(Default::default())
            }
        };
        set.insert(value)
    }

    fn remove(&mut self, p: P, value: &T) -> bool {
        if let Some(set) = self.sets.get_mut(&p) {
            set.swap_remove(value)
        } else {
            false
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
    NegX,
    NegY,
    NegZ,
}

#[inline]
fn tile_list_from_wave(wave: &[u64], wave_size: usize, tile_list: &mut Vec<u8>) {
    tile_list.clear();
    for i in (0..wave_size).filter(|&i| wave_contains(wave, i as _)) {
        tile_list.push(i as _);
    }
}

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

// Specifies connections along the 6 axis (+/-, x/y/z)
#[repr(transparent)]
#[derive(Debug, Clone)]
struct Tile<const WAVE_SIZE: usize> {
    connections: Vec<[u64; WAVE_SIZE]>,
}

impl<const WAVE_SIZE: usize> Tile<WAVE_SIZE> {
    fn new<D: Direction>() -> Self {
        Self {
            connections: vec![[0; WAVE_SIZE]; D::ALL.len()],
        }
    }

    fn connect<D: Direction>(&mut self, other: usize, dir: &D) {
        self.connections[dir.as_index()][other / 64] |= 1 << (other % 64);
    }
}

/// Stores tile connections and their probabilities.
pub struct Tileset<Dir: Direction, const WAVE_SIZE: usize = 4> {
    tiles: Vec<Tile<WAVE_SIZE>>,
    probabilities: Vec<f32>,
    _phantom: std::marker::PhantomData<Dir>,
}

impl<Dir: Direction, const WAVE_SIZE: usize> Default for Tileset<Dir, WAVE_SIZE> {
    fn default() -> Self {
        Self {
            tiles: Default::default(),
            probabilities: Default::default(),
            _phantom: Default::default(),
        }
    }
}

impl<Dir: Direction, const WAVE_SIZE: usize> Clone for Tileset<Dir, WAVE_SIZE> {
    fn clone(&self) -> Self {
        Self {
            tiles: self.tiles.clone(),
            probabilities: self.probabilities.clone(),
            _phantom: self._phantom,
        }
    }
}

impl<Dir: Direction, const WAVE_SIZE: usize> Tileset<Dir, WAVE_SIZE> {
    /// Add a new tine with a given probability.
    #[inline]
    pub fn add(&mut self, probability: f32) -> usize {
        let index = self.tiles.len();
        self.tiles.push(Tile::new::<Dir>());
        self.probabilities.push(probability);
        index
    }

    /// Connect up two tiles on a number of axises
    #[inline]
    pub fn connect(&mut self, from: usize, to: usize, axises: &[Dir]) {
        for axis in axises {
            self.tiles[to].connect(from, &axis.opposite());
            self.tiles[from].connect(to, axis);
        }
    }

    /// Connect a tile to every other tile on every axis.
    #[inline]
    pub fn connect_to_all(&mut self, tile: usize) {
        for other in 0..self.tiles.len() {
            self.connect(tile, other, Dir::ALL)
        }
    }

    fn normalize_probabilities(&mut self) {
        let mut sum = 0.0;
        for &prob in &self.probabilities {
            sum += prob;
        }
        for prob in &mut self.probabilities {
            *prob /= sum;
        }
    }

    /// Build a WFC solver from these tiles.
    #[inline]
    pub fn into_wfc<E: Entropy>(mut self, size: (u32, u32, u32)) -> Wfc<Dir, E, WAVE_SIZE> {
        self.normalize_probabilities();

        let (width, height, depth) = size;
        Wfc {
            state: State::new(
                self.initial_wave(),
                (width * height * depth) as usize,
                &self.probabilities,
            ),
            tiles: self.tiles,
            probabilities: self.probabilities,
            initial_state: None,
            width,
            height,
            scratch_data: Default::default(),
            _phantom: self._phantom,
        }
    }

    /// Build a WFC solver with a partially-collapsed initial state.
    #[inline]
    pub fn into_wfc_with_initial_state<E: Entropy>(
        self,
        size: (u32, u32, u32),
        array: &[[u64; WAVE_SIZE]],
    ) -> Wfc<Dir, E, WAVE_SIZE> {
        let mut wfc = self.into_wfc(size);
        wfc.collapse_initial_state(array);
        wfc
    }

    /// Get the total number of tiles.
    #[inline]
    pub fn num_tiles(&self) -> usize {
        self.tiles.len()
    }

    // Get the initial wave value
    #[inline]
    pub fn initial_wave(&self) -> [u64; WAVE_SIZE] {
        initial_wave(self.tiles.len())
    }
}

#[inline]
fn initial_wave<const WAVE_SIZE: usize>(mut len: usize) -> [u64; WAVE_SIZE] {
    let mut wave = [0; WAVE_SIZE];
    for section in wave.iter_mut() {
        if len >= 64 {
            *section = !0;
            len -= 64;
        } else {
            *section = (!0) >> (64 - len);
            break;
        }
    }
    wave
}

/// A method of determining entropy.
/// Waves that have many possible states have high entropy, while waves that have less have lower entropy
pub trait Entropy: Default + Send + Clone {
    type Type: Ord + Clone + Copy + Default + Hash + Send;
    fn calculate(probabilities: &[f32], wave: &[u64]) -> Self::Type;
}

/// Use the shannon entropy calculation where the probablility of the
/// remaining possible states matters.
#[derive(Clone, Default)]
pub struct ShannonEntropy;

impl Entropy for ShannonEntropy {
    type Type = OrderedFloat<f32>;

    #[inline]
    fn calculate(probabilities: &[f32], wave: &[u64]) -> Self::Type {
        let mut sum = 0.0;
        for (_, &prob) in probabilities
            .iter()
            .enumerate()
            .filter(|&(i, prob)| *prob > 0.0 && wave_contains(wave, i as _))
        {
            sum -= prob * prob.log2();
        }
        OrderedFloat(sum)
    }
}

/// Simply use the number of remaining states in the wave for entropy.
///
/// Doesn't take probability of tiles into account. Slightly faster than shannon entryopy.
#[derive(Clone, Default)]
pub struct LinearEntropy;

impl Entropy for LinearEntropy {
    type Type = u8;

    #[inline]
    fn calculate(_probabilities: &[f32], wave: &[u64]) -> Self::Type {
        count_ones(wave) as _
    }
}

#[derive(Clone, Default)]
struct State<E: Entropy, const WAVE_SIZE: usize> {
    array: Vec<[u64; WAVE_SIZE]>,
    entropy_to_indices: SetQueue<u32, Reverse<<E as Entropy>::Type>>,
}

impl<E: Entropy, const WAVE_SIZE: usize> State<E, WAVE_SIZE> {
    fn new(initial_wave: [u64; WAVE_SIZE], size: usize, probabilities: &[f32]) -> Self {
        let set = IndexSet::new_from_vec((0..size as u32).collect());
        let mut entropy_to_indices = SetQueue::default();
        entropy_to_indices.insert_set(Reverse(E::calculate(probabilities, &initial_wave)), set);

        State {
            array: vec![initial_wave; size],
            entropy_to_indices,
        }
    }
}

#[derive(Default, Clone)]
struct ScratchData<const WAVE_SIZE: usize> {
    stack: Vec<(u32, [u64; WAVE_SIZE])>,
    prefix_summed_probabilities: Vec<OrderedFloat<f32>>,
    tile_list: Vec<u8>,
}

/// The main WFC solver
#[derive(Clone)]
pub struct Wfc<Dir: Direction, E: Entropy, const WAVE_SIZE: usize = 4> {
    tiles: Vec<Tile<WAVE_SIZE>>,
    probabilities: Vec<f32>,
    state: State<E, WAVE_SIZE>,
    initial_state: Option<State<E, WAVE_SIZE>>,
    width: u32,
    height: u32,
    scratch_data: ScratchData<WAVE_SIZE>,
    _phantom: std::marker::PhantomData<Dir>,
}

impl<Dir: Direction, E: Entropy, const WAVE_SIZE: usize> Wfc<Dir, E, WAVE_SIZE> {
    /// Get the initial wave value
    #[inline]
    pub fn initial_wave(&self) -> [u64; WAVE_SIZE] {
        initial_wave(self.tiles.len())
    }

    /// Partially collapse the initial state based on provided input.
    #[inline]
    fn collapse_initial_state(&mut self, state: &[[u64; WAVE_SIZE]]) {
        let initial_wave = self.initial_wave();

        let mut any_contradictions = false;

        // Collapse all provided less-than-initial-wave states.
        for (i, wave) in state.iter().copied().enumerate() {
            if wave != initial_wave && wave != [0; WAVE_SIZE] {
                any_contradictions |= self.partial_collapse(i as u32, wave);
            }
        }

        assert!(!any_contradictions);

        self.initial_state = Some(self.state.clone());
    }

    /// Reset the solver back to the initial state.
    #[inline]
    pub fn reset(&mut self) {
        if let Some(initial_state) = self.initial_state.as_ref() {
            self.state.clone_from(initial_state);
        } else {
            self.state = State::new(
                self.initial_wave(),
                self.state.array.len(),
                &self.probabilities,
            );
        }
    }

    /// Get the number of tiles
    #[inline]
    pub fn num_tiles(&self) -> usize {
        self.tiles.len()
    }

    /// Get the width of the array
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the height of the array
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the depth of the array
    #[inline]
    pub fn depth(&self) -> u32 {
        self.state.array.len() as u32 / self.width() / self.height()
    }

    /// Select the tile with the lowest entropy (randomly if there are multiples with the lowest entropy)
    /// and return its index and the state it should be collapsed to.
    ///
    /// Doesn't actually change the state at all apart from removing unused sets from `entropy_to_indices`.
    #[inline]
    pub fn select_from_lowest_entropy<R: Rng>(&mut self, rng: &mut R) -> Option<(u32, u8)> {
        self.state.entropy_to_indices.select_from_lowest_entropy(
            rng,
            |&index| self.state.array[index as usize],
            &self.probabilities,
            &mut self.scratch_data.tile_list,
            &mut self.scratch_data.prefix_summed_probabilities,
        )
    }

    /// Like `collapse_all_reset_on_contradiction` but uses rayon for parallelism.
    #[cfg(feature = "rayon")]
    #[inline]
    pub fn collapse_all_reset_on_contradiction_par<R: Rng>(&mut self, mut rng: &mut R) -> u32 {
        use rand::{SeedableRng, rngs::SmallRng};

        let states: Vec<_> = (0..rayon::current_num_threads())
            .map(|_| (self.clone(), SmallRng::from_rng(&mut rng)))
            .collect();

        let other_attempts = AtomicU32::new(0);

        let (wfc, local_attempts) =
            find_any_with_early_stop(states, |(mut wfc, mut rng), stop_flag| {
                let mut attempts = 1;
                while let Some((index, tile)) = wfc.select_from_lowest_entropy(&mut rng) {
                    if stop_flag.load(Ordering::Relaxed) {
                        other_attempts.fetch_add(attempts, Ordering::Relaxed);
                        return None;
                    }
                    if wfc.collapse(index, tile) {
                        if attempts % 1000 == 0 {
                            println!("{}", attempts);
                        }
                        wfc.reset();
                        attempts += 1
                    }
                }
                Some((wfc, attempts))
            })
            .unwrap();

        let total_attempts = local_attempts + other_attempts.load(Ordering::Relaxed);

        println!(
            "Found after {} attempts, ({} thread local)",
            total_attempts, local_attempts,
        );

        *self = wfc;
        total_attempts
    }

    /// Like `collapse_all` but resets the state upon a contradiction. Returns the number of attempts it took.
    #[inline]
    pub fn collapse_all_reset_on_contradiction<R: Rng>(&mut self, rng: &mut R) -> u32 {
        let mut attempts = 1;
        while let Some((index, tile)) = self.select_from_lowest_entropy(rng) {
            if self.collapse(index, tile) {
                if attempts % 1000 == 0 {
                    println!("{}", attempts);
                }
                self.reset();
                attempts += 1
            }
        }

        attempts
    }

    /// Collapse all tiles until nothing else can be collapsed.
    ///
    /// Returns whether this caused a contradiction (e.g. a tile has 0 possible states)
    #[inline]
    pub fn collapse_all<R: Rng>(&mut self, rng: &mut R) -> bool {
        let mut any_contradictions = false;
        while let Some((index, tile)) = self.select_from_lowest_entropy(rng) {
            if self.collapse(index, tile) {
                any_contradictions = true;
            }
        }

        any_contradictions
    }

    /// Collapse the wave at `index` down to a single state.
    ///
    /// Returns whether this caused a contradiction (e.g. a tile has 0 possible states)
    #[inline]
    pub fn collapse(&mut self, index: u32, tile: u8) -> bool {
        let mut wave = [0; WAVE_SIZE];
        wave[(tile as usize) / 64] = 1 << (tile as usize % 64);
        self.partial_collapse(index, wave)
    }

    /// Partially collapse the wave at `index` down to a smaller wave.
    ///
    /// Returns whether this caused a contradiction (e.g. a tile has 0 possible states)
    #[inline]
    pub fn partial_collapse(
        &mut self,
        index: u32,
        remaining_possible_states: [u64; WAVE_SIZE],
    ) -> bool {
        self.scratch_data.stack.clear();
        self.scratch_data
            .stack
            .push((index, remaining_possible_states));

        let mut any_contradictions = false;

        while let Some((index, remaining_possible_states)) = self.scratch_data.stack.pop() {
            let old = self.state.array[index as usize];
            for (array, remaining_possible_states) in self.state.array[index as usize]
                .iter_mut()
                .zip(&remaining_possible_states)
            {
                *array &= remaining_possible_states;
            }
            let new = self.state.array[index as usize];

            if old == new {
                continue;
            }

            if count_ones(&old) > 1 {
                let _val = self
                    .state
                    .entropy_to_indices
                    .remove(Reverse(E::calculate(&self.probabilities, &old)), &index);
                debug_assert!(_val);
            }

            if new == [0; WAVE_SIZE] {
                any_contradictions = true;
                continue;
            }

            if count_ones(&new) > 1 {
                let _val = self
                    .state
                    .entropy_to_indices
                    .insert(Reverse(E::calculate(&self.probabilities, &new)), index);
                debug_assert!(_val);
            }

            self.propagate_updated_wave(new, index);
        }

        any_contradictions
    }

    // Intended to be called only inside `partial_collapse`.
    //
    // Given an `updated_wave` at `index`, push all possible new neighbour states onto the stack.
    fn propagate_updated_wave(&mut self, updated_wave: [u64; WAVE_SIZE], index: u32) {
        tile_list_from_wave(
            &updated_wave,
            self.tiles.len(),
            &mut self.scratch_data.tile_list,
        );
        let current_possibilities = &self.scratch_data.tile_list;

        for axis in Dir::ALL {
            let (x, y, z) = (
                index % self.width(),
                (index / self.width()) % self.height(),
                index / self.width() / self.height(),
            );
            let Some((x, y, z)) = axis.apply(x, y, z, self.width(), self.height(), self.depth())
            else {
                continue;
            };

            let index = x + y * self.width() + z * self.width() * self.height();

            let mut valid = [0; WAVE_SIZE];

            for &tile in current_possibilities.iter() {
                let v = self.tiles[tile as usize].connections[axis.as_index()];
                for (valid, v) in valid.iter_mut().zip(&v) {
                    *valid |= v;
                }
            }

            self.scratch_data.stack.push((index, valid));
        }
    }

    /// Get the collapsed values of the array.
    ///
    /// Any non-collapsed or contradictory values are returned as 255.
    #[inline]
    pub fn values(&self) -> Vec<u8> {
        let mut values = vec![0; self.state.array.len()];
        self.set_values(&mut values);
        values
    }

    /// Like `values` but takes in a slice instead.
    #[inline]
    pub fn set_values(&self, values: &mut [u8]) {
        self.state
            .array
            .iter()
            .zip(values)
            .for_each(|(wave, value)| {
                *value = if count_ones(wave) == 1 {
                    trailing_zeros(wave) as u8
                } else {
                    u8::MAX
                }
            });
    }

    #[cfg(test)]
    fn all_collapsed(&self) -> bool {
        self.state
            .array
            .iter()
            .all(|&value| count_ones(&value) == 1)
    }
}

#[inline]
fn sample_prefix_sum<R: rand::Rng>(prefix_sum: &[OrderedFloat<f32>], rng: &mut R) -> usize {
    let num = rng.random_range(0.0..=prefix_sum[prefix_sum.len() - 1].0);
    match prefix_sum.binary_search(&OrderedFloat(num)) {
        Ok(index) => index,
        Err(index) => index,
    }
}

#[cfg(feature = "rayon")]
fn find_any_with_early_stop<
    T,
    O: Send,
    I: IntoParallelIterator<Item = T>,
    F: Sync + Fn(T, &AtomicBool) -> Option<O>,
>(
    iterator: I,
    func: F,
) -> Option<O> {
    let stop_flag = AtomicBool::new(false);
    iterator.into_par_iter().find_map_any(|item| {
        func(item, &stop_flag).inspect(|_| stop_flag.store(true, Ordering::Relaxed))
    })
}

#[cfg(test)]
use rand::{SeedableRng, rngs::SmallRng};

#[test]
fn normal() {
    let mut rng = SmallRng::from_os_rng();

    let mut tileset = Tileset::<_>::default();
    let sea = tileset.add(1.0);
    let beach = tileset.add(0.5);
    let grass = tileset.add(1.0);
    tileset.connect(sea, sea, &Axis::ALL);
    tileset.connect(sea, beach, &Axis::ALL);
    tileset.connect(beach, beach, &Axis::ALL);
    tileset.connect(beach, grass, &Axis::ALL);
    tileset.connect(grass, grass, &Axis::ALL);

    let mut v = [0; 4];
    v[0] = 3;
    assert_eq!(tileset.tiles[sea].connections, [v; 6]);

    let mut wfc = tileset.into_wfc::<ShannonEntropy>((100, 1000, 1));

    assert!(!wfc.all_collapsed());
    assert!(!wfc.collapse_all(&mut rng));
    assert!(
        wfc.all_collapsed(),
        "failed to collapse: {:?}",
        &wfc.state
            .array
            .iter()
            .map(|v| count_ones(v))
            .collect::<Vec<_>>()
    );
}

#[test]
fn normal_2d() {
    let mut rng = SmallRng::from_os_rng();

    let mut tileset = Tileset::<_>::default();
    let sea = tileset.add(1.0);
    let beach = tileset.add(0.5);
    let grass = tileset.add(1.0);
    tileset.connect(sea, sea, &Axis2D::ALL);
    tileset.connect(sea, beach, &Axis2D::ALL);
    tileset.connect(beach, beach, &Axis2D::ALL);
    tileset.connect(beach, grass, &Axis2D::ALL);
    tileset.connect(grass, grass, &Axis2D::ALL);

    let mut v = [0; 4];
    v[0] = 3;
    assert_eq!(tileset.tiles[sea].connections, [v; 4]);

    let mut wfc = tileset.into_wfc::<ShannonEntropy>((100, 1000, 1));

    assert!(!wfc.all_collapsed());
    assert!(!wfc.collapse_all(&mut rng));
    assert!(
        wfc.all_collapsed(),
        "failed to collapse: {:?}",
        &wfc.state
            .array
            .iter()
            .map(|v| count_ones(v))
            .collect::<Vec<_>>()
    );
}

#[test]
fn initial_state() {
    let mut rng = SmallRng::from_os_rng();

    let mut tileset = Tileset::<_>::default();
    let sea = tileset.add(1.0);
    let beach = tileset.add(1.0);
    let grass = tileset.add(1.0);
    tileset.connect(sea, sea, &Axis::ALL);
    tileset.connect(sea, beach, &Axis::ALL);
    tileset.connect(beach, grass, &Axis::ALL);
    tileset.connect(grass, grass, &Axis::ALL);

    let mut v = [0; 4];
    v[0] = 1 << sea | 1 << beach | 1 << grass;
    let mut state = [v; 9];
    assert_eq!(
        tileset.clone()
            .into_wfc_with_initial_state::<LinearEntropy>((3, 3, 1), &state)
            .state
            .array,
        state
    );
    state[4][0] = 1 << sea;
    #[rustfmt::skip]
    let expected = [
        7,3,7,
        3,1,3,
        7,3,7,
        ].map(|v| {
            let mut a = [0; 4];
            a[0] = v;
            a
        });
    let mut wfc = tileset.into_wfc_with_initial_state::<ShannonEntropy>((3, 3, 1), &state);
    assert_eq!(wfc.state.array, expected);
    wfc.collapse_all(&mut rng);
    assert_ne!(wfc.state.array, expected);
    assert_eq!(wfc.values()[4], sea as _);
    wfc.reset();
    assert_eq!(wfc.state.array, expected);
    
}

#[test]
fn initial_state_2d() {
    let mut rng = SmallRng::from_os_rng();

    let mut tileset = Tileset::<_>::default();
    let sea = tileset.add(1.0);
    let beach = tileset.add(1.0);
    let grass = tileset.add(1.0);
    tileset.connect(sea, sea, &Axis2D::ALL);
    tileset.connect(sea, beach, &Axis2D::ALL);
    tileset.connect(beach, grass, &Axis2D::ALL);
    tileset.connect(grass, grass, &Axis2D::ALL);

    let mut v = [0; 4];
    v[0] = 1 << sea | 1 << beach | 1 << grass;
    let mut state = [v; 9];
    assert_eq!(
        tileset
            .clone().into_wfc_with_initial_state::<LinearEntropy>((3, 3, 1), &state)
            .state
            .array,
        state
    );
    state[4][0] = 1 << sea;
    #[rustfmt::skip]
    let expected = [
        7,3,7,
        3,1,3,
        7,3,7,
        ].map(|v| {
            let mut a = [0; 4];
            a[0] = v;
            a
        });
    let mut wfc = tileset.into_wfc_with_initial_state::<ShannonEntropy>((3, 3, 1), &state);
    assert_eq!(wfc.state.array, expected);
    wfc.collapse_all(&mut rng);
    assert_ne!(wfc.state.array, expected);
    assert_eq!(wfc.values()[4], sea as _);
    wfc.reset();
    assert_eq!(wfc.state.array, expected);
}

#[test]
fn verticals() {
    let mut rng = SmallRng::from_os_rng();

    let mut tileset = Tileset::<_>::default();
    let air = tileset.add(1.0);
    let solid = tileset.add(1.0);
    tileset.connect(air, air, &Axis::ALL);
    tileset.connect(solid, solid, &Axis::ALL);
    // solid cant be above air
    tileset.connect(
        solid,
        air,
        &[Axis::X, Axis::Y, Axis::Z, Axis::NegX, Axis::NegY],
    );

    let mut wfc = tileset.into_wfc::<ShannonEntropy>((50, 50, 50));

    assert!(!wfc.all_collapsed());
    assert!(!wfc.collapse_all(&mut rng));
    assert!(
        wfc.all_collapsed(),
        "{:?}",
        &wfc.state
            .array
            .iter()
            .map(|v| count_ones(v))
            .collect::<Vec<_>>()
    );
    let _v = wfc.values();
    //panic!("{:?}",v);
}

#[test]
fn stairs() {
    let mut rng = SmallRng::from_os_rng();

    let mut tileset = Tileset::<_>::default();
    let empty = tileset.add(0.0);
    let ground = tileset.add(1.0);
    tileset.connect(ground, ground, &[Axis::X, Axis::Y]);
    let stairs_top = tileset.add(1.0);
    let stairs_bottom = tileset.add(10.0);
    tileset.connect(stairs_top, stairs_bottom, &[Axis::X, Axis::NegZ]);
    tileset.connect(stairs_top, ground, &[Axis::X]);
    tileset.connect(stairs_bottom, ground, &[Axis::NegX]);
    //tileset.connect(solid, solid, &Axis::ALL);

    tileset.connect_to_all(empty);

    let mut wfc = tileset.into_wfc::<ShannonEntropy>((5, 5, 5));

    assert!(!wfc.collapse_all(&mut rng));
    assert!(wfc.all_collapsed(),);
}

#[test]
fn broken() {
    let mut rng = SmallRng::from_os_rng();

    let mut tileset = Tileset::<_>::default();

    let sea = tileset.add(1.0);
    let beach = tileset.add(1.0);
    let grass = tileset.add(1.0);
    tileset.connect(sea, sea, &Axis::ALL);
    tileset.connect(sea, beach, &Axis::ALL);
    //tileset.connect(beach, beach, &Axis::ALL);
    tileset.connect(beach, grass, &Axis::ALL);
    tileset.connect(grass, grass, &Axis::ALL);

    let mut v = [0; 4];
    v[0] = 3;
    assert_eq!(tileset.tiles[sea].connections, [v; 6]);

    // Wait until there's a collapse failure due to beaches not being able to connect to beaches.
    loop {
        let mut wfc = tileset.clone().into_wfc::<ShannonEntropy>((10, 10, 1));

        assert!(!wfc.all_collapsed());

        if wfc.collapse_all(&mut rng) {
            assert!(!wfc.all_collapsed());
            // Make sure that at least one state has collapsed properly (aka that the error hasn't spread).
            assert!(wfc.state.array.iter().any(|&v| count_ones(&v) == 1));
            break;
        }
    }
}

#[test]
fn broken_2d() {
    let mut rng = SmallRng::from_os_rng();

    let mut tileset = Tileset::<_>::default();

    let sea = tileset.add(1.0);
    let beach = tileset.add(1.0);
    let grass = tileset.add(1.0);
    tileset.connect(sea, sea, &Axis2D::ALL);
    tileset.connect(sea, beach, &Axis2D::ALL);
    //tileset.connect(beach, beach, &Axis2D::ALL);
    tileset.connect(beach, grass, &Axis2D::ALL);
    tileset.connect(grass, grass, &Axis2D::ALL);

    let mut v = [0; 4];
    v[0] = 3;
    assert_eq!(tileset.tiles[sea].connections, [v; 4]);

    // Wait until there's a collapse failure due to beaches not being able to connect to beaches.
    loop {
        let mut wfc = tileset.clone().into_wfc::<ShannonEntropy>((10, 10, 1));

        assert!(!wfc.all_collapsed());

        if wfc.collapse_all(&mut rng) {
            assert!(!wfc.all_collapsed());
            // Make sure that at least one state has collapsed properly (aka that the error hasn't spread).
            assert!(wfc.state.array.iter().any(|&v| count_ones(&v) == 1));
            break;
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum Axis2D {
    X,
    Y,
    NegX,
    NegY,
}

#[cfg(test)]
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

#[test]
fn pipes() {
    let mut rng = SmallRng::from_os_rng();

    let mut tileset = Tileset::<_>::default();

    let empty = tileset.add(1.0);
    let pipe_x = tileset.add(1.0);
    let pipe_y = tileset.add(1.0);
    let t = tileset.add(1.0);
    tileset.connect(empty, empty, &Axis::ALL);
    tileset.connect(pipe_x, pipe_x, &Axis::ALL);
    tileset.connect(pipe_y, pipe_y, &Axis::ALL);
    tileset.connect(empty, pipe_x, &[Axis::X, Axis::NegX]);
    tileset.connect(empty, pipe_y, &[Axis::Y, Axis::NegY]);
    tileset.connect(empty, t, &[Axis::Z, Axis::NegZ, Axis::NegY]);
    tileset.connect(t, pipe_y, &[Axis::Y]);
    tileset.connect(t, pipe_y, &[Axis::X, Axis::NegX]);

    tileset
        .into_wfc::<ShannonEntropy>((10, 10, 10))
        .collapse_all_reset_on_contradiction(&mut rng);
}
