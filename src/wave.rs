#[inline]
pub fn count_ones(wave: &[u64]) -> u32 {
    wave.iter().map(|section| section.count_ones()).sum()
}

#[inline]
pub fn trailing_zeros(wave: &[u64]) -> u32 {
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
pub fn contains(wave: &[u64], bit: u8) -> bool {
    wave[(bit / 64) as usize] >> ((bit % 64) as usize) & 1 != 0
}

#[inline]
pub fn initial<const WAVE_SIZE: usize>(mut len: usize) -> [u64; WAVE_SIZE] {
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

#[inline]
pub fn set_one(wave: &mut [u64], index: usize) {
    wave[index / 64] |= 1 << (index % 64);
}
