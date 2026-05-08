//! Random generator implementation module

use std::num::{NonZeroU32, NonZeroU64, NonZeroU128};

/// Xoshiro256++ random generator
pub struct Xoshiro256pp(u64, u64, u64, u64);

impl Xoshiro256pp {
    /// Minimal generated value
    pub const MIN: u64 = 0u64;

    /// Maximal generated value
    pub const MAX: u64 = !0u64;

    /// Generate next u32
    pub const fn new(seed: NonZeroU64) -> Self {
        Self(seed.get(), 0, 0, 0)
    }

    /// Generate next uint64
    pub const fn next_u64(&mut self) -> u64 {
        let result = self.3.wrapping_add(self.0).rotate_left(23).wrapping_add(self.0);
        let t = self.1 << 17;

        self.2 ^= self.0;
        self.3 ^= self.1;
        self.1 ^= self.2;
        self.0 ^= self.3;

        self.2 ^= t;
        self.3 = self.3.rotate_left(45);

        result
    }

    /// Generate unit float64 number
    pub fn next_unit_f64(&mut self) -> f64 {
        self.next_u64() as f64 / u64::MAX as f64
    }
}

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
    /// Minimal generated value
    pub const MIN: u64 = 0u64;

    /// Maximal generated value
    pub const MAX: u64 = !0u64;

    /// Construct random device
    pub fn new(seed: NonZeroU128) -> Self {
        let seed = seed.get();

        Self {
            low:  ((seed      ) & u64::MAX as u128) as u64,
            high: ((seed >> 64) & u64::MAX as u128) as u64,
        }
    }

    /// Get next random number
    pub fn next_u64(&mut self) -> u64 {
        let mut low = self.low;
        let high = self.high;

        self.low = high;

        low ^= low << 23;
        low ^= low >> 18;
        low ^= high ^ (high >> 5);
        self.high = low;

        low.wrapping_add(high)
    }

    /// Generate unit float64 number
    pub fn next_unit_f64(&mut self) -> f64 {
        self.next_u64() as f64 / u64::MAX as f64
    }
}

impl Iterator for Xorshift128p {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next_u64())
    }
}

/// (VERY) Lightweight random generator
#[derive(Copy, Clone)]
pub struct Xorshift32 {
    /// Current random generator state
    state: u32,
}

impl Xorshift32 {
    /// Generator constructor
    pub fn new(seed: NonZeroU32) -> Self {
        Self { state: seed.get() }
    }

    /// Generate next number
    pub fn next_u32(&mut self) -> u32 {
        self.state = xorshift32(self.state);
        self.state
    }

    /// Generate next F64 in [0..1] range
    pub fn next_unit_f64(&mut self) -> f64 {
        self.next_u32() as f64 / u32::MAX as f64
    }
}

impl Iterator for Xorshift32 {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next_u32())
    }
}

/// Perform 32-bit xorshift step
pub fn xorshift32(mut n: u32) -> u32 {
    n ^= n << 13;
    n ^= n >> 17;
    n ^= n << 5;
    n
}

/// Perform 64-bit xorshift step
pub fn xorshift64(mut n: u64) -> u64 {
    n ^= n << 13;
    n ^= n >> 7;
    n ^= n << 17;
    n
}
