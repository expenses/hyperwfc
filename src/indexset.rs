use fnv::FnvBuildHasher;
use std::fmt::Debug;
use std::hash::Hash;

// Custom IndexSet implementation. Mirrors `indexmap::IndexSet`
// but is a bit faster.
#[derive(Clone)]
pub struct IndexSet<T> {
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
    pub fn new_from_vec(values: Vec<T>) -> Self {
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

    pub fn insert(&mut self, value: T) -> bool {
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

    pub fn swap_remove(&mut self, value: &T) -> bool {
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

    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }
}
