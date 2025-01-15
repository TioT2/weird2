use std::num::NonZeroU128;

/// 128bit lightweight random generator.
/// This implementation is based on xorshift128+
/// random integer generation algorithm.
#[derive(Copy, Clone)]
pub struct Xorshift128p {
    /// Low part
    low: u64,

    /// High part
    high: u64,
}

impl Xorshift128p {
    pub const MIN: u64 = 0u64;
    pub const MAX: u64 = !0u64;

    /// Construct random device
    pub fn new(seed: NonZeroU128) -> Self {
        let seed = seed.get();

        Self {
            low: ((seed >>  0) & 0xFFFF_FFFF_FFFF_FFFF) as u64,
            high: ((seed >> 64) & 0xFFFF_FFFF_FFFF_FFFF) as u64,
        }
    }

    /// Get next random number
    pub fn next(&mut self) -> u64 {
        let mut low = self.low;
        let high = self.high;

        self.low = high;

        low ^= low << 23;
        low ^= low >> 18;
        low ^= high ^ (high >> 5);
        self.high = low;

        low.wrapping_add(high)
    }

    pub fn next_unit_f64(&mut self) -> f64 {
        return self.next() as f64 / 0xFFFF_FFFF_FFFF_FFFFu64 as f64
    }
}

impl Iterator for Xorshift128p {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next())
    }
}