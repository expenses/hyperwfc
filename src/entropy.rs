use crate::wave;
use ordered_float::OrderedFloat;
use std::hash::Hash;

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
            .filter(|&(i, prob)| *prob > 0.0 && wave::contains(wave, i as _))
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
        wave::count_ones(wave) as _
    }
}
