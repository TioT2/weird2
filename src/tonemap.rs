//! Tonemapping implementation

use crate::u64_into_u16;

/// Tonemapping operator kind
#[derive(Copy, Clone)]
pub enum Operator {
    /// Reinhard
    Reinhard {
        allow_simd: bool
    },

    /// Clamping operator
    Clamp,
}

crate::index_enum!(
    Operator
    0 { Self::Reinhard { allow_simd: false } }
    1 { Self::Reinhard { allow_simd: true }  }
    2 { Self::Clamp                          }
);

/// Tonemap clamping function
#[inline]
pub fn tonemap_clamp(hdr: &u64, ldr: &mut u32) {
    let [r, g, b, _] = u64_into_u16(*hdr);
    *ldr = u32::from_le_bytes([
        u16::min(r, 0xFF) as u8,
        u16::min(g, 0xFF) as u8,
        u16::min(b, 0xFF) as u8,
        0
    ]);
}

/// Reinhard tonemapping exposure
pub const REINHARD_EXPOSURE: f32 = 0.5;

/// Default Reinhard
#[inline]
pub fn tonemap_reinhard(hdr: &u64, ldr: &mut u32) {
    let [r, g, b, _] = u64_into_u16(*hdr);
    let [r, g, b] = [r as f32, g as f32, b as f32];

    /// 256 inverse value
    const INV256: f32 = 256.0f32.recip();

    *ldr = u32::from_ne_bytes([
        (r / (r * INV256 + REINHARD_EXPOSURE)) as u8,
        (g / (g * INV256 + REINHARD_EXPOSURE)) as u8,
        (b / (b * INV256 + REINHARD_EXPOSURE)) as u8,
        0,
    ]);
}

/// SSE2-accelerated Reinhard tonemapping
#[cfg(target_feature = "sse2")]
#[inline]
pub fn tonemap_reinhard_sse2(hdr: &u64, ldr: &mut u32) {
    use std::arch::x86_64 as arch;

    unsafe {
        let src16 = std::mem::transmute::<arch::__m128d, arch::__m128i>(
            arch::_mm_load_sd(hdr as *const u64 as *const f64)
        );
        let src32 = arch::_mm_cvtepi16_epi32(src16);
        let src = arch::_mm_cvtepi32_ps(src32);

        // rgba / 256.0 + exposure
        let rgba_norm_aexp = arch::_mm_fmadd_ps(
            src,
            arch::_mm_set1_ps(1.0 / 256.0),
            arch::_mm_set1_ps(REINHARD_EXPOSURE)
        );

        // rgba / (rgba / 256.0 + exposure)
        let mapped = arch::_mm_div_ps(src, rgba_norm_aexp);

        let mapped32 = arch::_mm_cvtps_epi32(mapped);
        let mapped16 = arch::_mm_packus_epi32(mapped32, mapped32);
        let mapped8 = arch::_mm_packus_epi16(mapped16, mapped16);

        arch::_mm_store_ss(
            ldr as *mut u32 as *mut f32,
            std::mem::transmute::<arch::__m128i, arch::__m128>(mapped8)
        );
    }
}

/// Get tonemapping function by parameters
pub fn get_function(op: Operator, swap_bytes: bool) -> &'static dyn Fn(&[u64], &mut [u32]) {
    fn impl_<const OPI: usize, const SWAP: bool>(src: &[u64], dst: &mut [u32]) {
        for (src, dst) in src.iter().zip(dst.iter_mut()) {
            if OPI == 0 {
                tonemap_reinhard(src, dst);
            } else if OPI == 1 {
                cfg_select!(
                    target_feature = "sse2" => tonemap_reinhard_sse2,
                    _ => tonemap_renihard,
                )(src, dst);
            } else if OPI == 2 {
                tonemap_clamp(src, dst);
            }

            if SWAP {
                *dst = dst.swap_bytes() >> 8;
            }
        }
    }

    match (op, swap_bytes) {
        (Operator::Reinhard { allow_simd: false }, false) => &impl_::<0, false>,
        (Operator::Reinhard { allow_simd: false }, true) => &impl_::<0, true>,
        (Operator::Reinhard { allow_simd: true }, false) => &impl_::<1, false>,
        (Operator::Reinhard { allow_simd: true }, true) => &impl_::<1, true>,

        (Operator::Clamp, false) => &impl_::<2, false>,
        (Operator::Clamp, true) => &impl_::<2, true>,
    }
}
