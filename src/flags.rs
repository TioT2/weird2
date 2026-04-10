//! Bit flags implementation file

use std::ops::{BitAnd, BitOr, BitXor, Not};

/// Trait for type capable for being flag internal representation.
pub trait FlagBits: 'static
    + Sized + Copy + Clone + PartialEq + Eq
    + BitAnd<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self> + Not<Output = Self>
{
    /// Zero value, all bits unset
    const ZERO: Self;
}

/// Implement FlagBits trait for integer type
macro_rules! impl_bits_int { ($($i: ty),*) => { $( impl FlagBits for $i { const ZERO: Self = 0; })* }; }
impl_bits_int!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

impl FlagBits for bool { const ZERO: Self = false; }

/// Common flag trait
pub trait Flags: Sized {
    /// Flag bit type
    type Bits: FlagBits;

    /// Convert bits into flag structure
    fn from_bits(bits: Self::Bits) -> Self;

    /// Extract bits from flag structrue
    fn into_bits(self) -> Self::Bits;

    /// Create flags with no bits set
    fn empty() -> Self {
        Self::from_bits(Self::Bits::ZERO)
    }

    /// Conditionally set/clear flag
    fn set_if(self, flag: Self, cond: bool) -> Self {
        if cond {
            self.set(flag)
        } else {
            self.clear(flag)
        }
    }

    /// Set flag
    fn set(self, flag: Self) -> Self {
        Self::from_bits(self.into_bits() | flag.into_bits())
    }

    /// Clear some flag
    fn clear(self, flag: Self) -> Self {
        Self::from_bits(self.into_bits() & !flag.into_bits())
    }

    /// Toggle (e.g. for all bits: check(flag) then set else clear) flag
    fn toggle(self, mask: Self) -> Self {
        let b = self.into_bits();
        let fb = mask.into_bits();

        Self::from_bits(((b ^ fb) & fb) | (b & !fb))
    }

    /// Check [`flag`] for being enabled
    fn check(self, flag: Self) -> bool {
        let fb = flag.into_bits();

        self.into_bits() & fb == fb
    }
}

// Identity implementation
impl<F: FlagBits> Flags for F {
    type Bits = F;

    fn from_bits(bits: Self::Bits) -> Self {
        bits
    }

    fn into_bits(self) -> Self::Bits {
        self
    }
}

/// Define flag structure
#[macro_export]
macro_rules! flags {
    (
        $(#[$flag_attr: meta])*
        $vis: vis struct $Flags: ident: $bits: ty {
            $(
                $(#[$bit_attr: meta])*
                const $BIT: ident = $value: expr;
            )*
        }
    ) => {
        // Define flag structure itself
        $(#[$flag_attr])*
        #[repr(transparent)]
        $vis struct $Flags(pub $bits);

        // Define flag bits
        impl $Flags {
            $(
                $(#[$bit_attr])*
                pub const $BIT: Self = Self($value);
            )*
        }

        impl $Flags where $bits: $crate::flags::FlagBits {
            /// Array containing all defined flags
            pub const ALL_DEFINED_FLAGS: &'static [(&'static str, $Flags)] = &[
                $((stringify!($BIT), Self($value))),*
            ];

            /// Create empty flags
            pub const fn empty() -> Self {
                Self($crate::flags::FlagBits::ZERO)
            }

            /// Conditional set/clear
            pub const fn set_if(self, flag: Self, cond: bool) -> Self {
                if cond {
                    self.set(flag)
                } else {
                    self.clear(flag)
                }
            }

            /// Set some flag
            pub const fn set(self, flag: Self) -> Self {
                Self(self.0 | flag.0)
            }

            /// Clear (e.g. set to false) some flag
            pub const fn clear(self, flag: Self) -> Self {
                Self(self.0 & !flag.0)
            }

            /// Toggle (e.g. for all bits: check(bit) then set else clear) flag
            pub const fn toggle(self, mask: Self) -> Self {
                Self(((self.0 ^ mask.0) & mask.0) | (self.0 & !mask.0))
            }

            /// Check [`flag`] for being enabled
            pub const fn check(self, flag: Self) -> bool {
                self.0 & flag.0 == flag.0
            }

            /// Map flags to another flag type
            pub fn map<T: $crate::flags::Flags>(self, m: impl Iterator<Item = (Self, T)>) -> T {
                let mut flags = T::empty();
                for (f, tf) in m {
                    flags = flags.set_if(tf, self.check(f));
                }
                flags
            }
        }

        impl $crate::flags::Flags for $Flags {
            type Bits = $bits;

            fn from_bits(bits: Self::Bits) -> Self {
                Self(bits)
            }

            fn into_bits(self) -> Self::Bits {
                self.0
            }

            fn empty() -> Self {
                Self::empty()
            }

            fn set_if(self, flag: Self, cond: bool) -> Self {
                self.set_if(flag, cond)
            }

            fn set(self, flag: Self) -> Self {
                self.set(flag)
            }

            fn clear(self, flag: Self) -> Self {
                self.clear(flag)
            }

            fn toggle(self, mask: Self) -> Self {
                self.toggle(mask)
            }

            fn check(self, flag: Self) -> bool {
                self.check(flag)
            }
        }

        impl std::fmt::Debug for $Flags {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, concat!(stringify!($Flags), "("))?;

                let mut enabled_flags = $Flags::ALL_DEFINED_FLAGS.iter().filter_map(|(n, f)| self.check(*f).then_some(n));

                if let Some(first) = enabled_flags.next() {
                    write!(f, "{}", first)?;

                    for flag in enabled_flags {
                        write!(f, " | {}", flag)?;
                    }
                }

                write!(f, ")")
            }
        }
    };
}
