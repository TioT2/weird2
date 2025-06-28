/// Fast vector implementation
/// Thia is is common representation of Vec3/Vec4
/// floating-point vectors based on x86 SSE extension.

use std::arch::x86_64 as arch;

/// SIMD-based float-point 3/4-component vector
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct FVec4(arch::__m128);

impl Default for FVec4 {
    fn default() -> Self {
        FVec4::zero()
    }
}

impl FVec4 {
    /// Construct vector from X, Y and Z. W is zeroed
    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self(unsafe {
            arch::_mm_set_ps(0.0, z, y, x)
        })
    }

    /// Construct vector from X, Y, Z, and W.
    pub fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(unsafe {
            arch::_mm_set_ps(w, z, y, x)
        })
    }

    /// Produce constant zero vector
    pub fn zero() -> Self {
        Self(unsafe { arch::_mm_setzero_ps() })
    }

    /// Construct vector from single number
    pub fn from_single(v: f32) -> Self {
        Self(unsafe {
            arch::_mm_set1_ps(v)
        })
    }

    /// Get field by (compile-time) index
    #[allow(unreachable_code)]
    pub fn get<const I: i32>(self) -> f32 {
        #[cfg(target_feature = "sse4.1")]
        {
            return unsafe {
                std::mem::transmute::<i32, f32>(
                    arch::_mm_extract_ps::<I>(self.0)
                )
            };
        }

        // SSE1/2-based solution
        unsafe {
            arch::_mm_cvtss_f32(
                arch::_mm_shuffle_ps::<I>(self.0, self.0)
            )
        }
    }

    /// Get X coordinate
    pub fn x(self) -> f32 { self.get::<0>() }

    /// Get Y coordinate
    pub fn y(self) -> f32 { self.get::<1>() }

    /// Get Z coordinate
    pub fn z(self) -> f32 { self.get::<2>() }

    /// Get W coordinate
    pub fn w(self) -> f32 { self.get::<3>() }

    /// Convert vector into 3-component tuple
    pub fn into_tuple_3(self) -> (f32, f32, f32) {
        unsafe {
            let arr = std::mem::transmute::<_, [f32; 4]>(self.0);

            (arr[0], arr[1], arr[2])
        }
    }

    /// Convert vector into 4-component tuple
    pub fn into_tuple_4(self) -> (f32, f32, f32, f32) {
        unsafe {
            let arr = std::mem::transmute::<_, [f32; 4]>(self.0);

            (arr[0], arr[1], arr[2], arr[3])
        }
    }

    /// Dot product
    #[allow(unreachable_code)]
    pub fn dot(self, rhs: FVec4) -> f32 {
        #[cfg(target_feature = "sse4.1")]
        {
            // However, this method may be slower on AMD CPUs, but idk.
            return unsafe {
                arch::_mm_cvtss_f32(
                    arch::_mm_dp_ps::<0b1111_0001>(self.0, rhs.0)
                )
            };
        }

        // SSE1/2-based solution
        unsafe {
            let product = arch::_mm_mul_ps(self.0, rhs.0);

            let shuffled = arch::_mm_shuffle_ps::<0b10_11_00_01>(product, product);
            let sums = arch::_mm_add_ps(product, shuffled);
            let shuffled = arch::_mm_movehl_ps(shuffled, sums);
            let sums = arch::_mm_add_ss(sums, shuffled);

            arch::_mm_cvtss_f32(sums)
        }
    }

    /// Multiply and add vectors (self * second + third) (in case of FMA unavailability, fallbacks to separate mul and add)
    pub fn mul_add(self, second: FVec4, third: FVec4) -> Self {
        unsafe {
            // FMA version
            #[cfg(target_feature = "fma")]
            return Self(
                arch::_mm_fmadd_ps(self.0, second.0, third.0)
            );

            // non-FMA version
            #[cfg(not(target_feature = "fma"))]
            return Self(
                arch::_mm_add_ps(arch::_mm_mul_ps(self.0, second.0), third.0)
            )
        }
    }

    /// Calculate 3-dimensional cross product of of self.xyz and rhs.xyz vectors
    pub fn cross(self, othr: FVec4) -> FVec4 {
        // 3 shuffles, 2 products and 1 substraction
        unsafe {
            let tmp0 = arch::_mm_shuffle_ps::<0b11_00_10_01>(self.0, self.0);
            let tmp1 = arch::_mm_shuffle_ps::<0b11_01_00_10>(othr.0, othr.0);
            let tmp2 = arch::_mm_mul_ps(tmp0, othr.0);
            let tmp3 = arch::_mm_mul_ps(tmp0, tmp1);
            let tmp4 = arch::_mm_shuffle_ps::<0b11_00_10_01>(tmp2, tmp2);

            Self(
                arch::_mm_sub_ps(tmp3, tmp4)
            )
        }

        // 4 shuffles, 2 products and 1 substraction
        // unsafe {
        //     let tmp0 = arch::_mm_shuffle_ps::<0b11_00_10_01>(self.0, self.0);
        //     let tmp1 = arch::_mm_shuffle_ps::<0b11_01_00_10>(othr.0, othr.0);
        //     let tmp2 = arch::_mm_shuffle_ps::<0b11_01_00_10>(self.0, self.0);
        //     let tmp3 = arch::_mm_shuffle_ps::<0b11_00_10_01>(othr.0, othr.0);
        //     Self(
        //         arch::_mm_sub_ps(
        //             arch::_mm_mul_ps(tmp0, tmp1),
        //             arch::_mm_mul_ps(tmp2, tmp3)
        //         )
        //     )
        // }
    }
}

macro_rules! impl_binary_operator {
    ($type: ident, $op_trait_name: ident, $op_func_name: ident, $arch_func_name: ident, $set1_func_name: ident) => {

        // vector-vector product
        impl std::ops::$op_trait_name<$type> for $type {
            type Output = $type;

            fn $op_func_name(self, rhs: $type) -> Self::Output {
                Self(unsafe { arch::$arch_func_name(self.0, rhs.0) })
            }
        }

        // vector-number product
        impl std::ops::$op_trait_name<f32> for $type {
            type Output = $type;

            fn $op_func_name(self, rhs: f32) -> Self::Output {
                Self(unsafe { arch::$arch_func_name(self.0, arch::$set1_func_name(rhs)) })
            }
        }
    };
}

impl_binary_operator!(FVec4, Mul, mul, _mm_mul_ps, _mm_set1_ps);
impl_binary_operator!(FVec4, Div, div, _mm_div_ps, _mm_set1_ps);
impl_binary_operator!(FVec4, Add, add, _mm_add_ps, _mm_set1_ps);
impl_binary_operator!(FVec4, Sub, sub, _mm_sub_ps, _mm_set1_ps);

/// Fast 8-component vector
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct FVec8(arch::__m256);

impl FVec8 {
    /// From component set
    pub fn from_components(c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32, c6: f32, c7: f32) -> Self {
        unsafe {
            Self(arch::_mm256_set_ps(c7, c6, c5, c4, c3, c2, c1, c0))
        }
    }

    /// From pair of FVec4's
    pub fn from_fvec4(v1: FVec4, v2: FVec4) -> Self {
        unsafe {
            Self(arch::_mm256_set_m128(v2.0, v1.0))
        }
    }

    /// From single number
    pub fn from_single(v: f32) -> Self {
        unsafe {
            Self(arch::_mm256_set1_ps(v))
        }
    }

    /// From zero components
    pub fn zero() -> Self {
        unsafe {
            Self(arch::_mm256_setzero_ps())
        }
    }

    /// Get element by index
    pub fn get<const I: i32>(self) -> f32 {
        // AVX solution
        unsafe {
            arch::_mm256_cvtss_f32(
                arch::_mm256_shuffle_ps::<I>(self.0, self.0)
            )
        }
    }
}

impl_binary_operator!(FVec8, Mul, mul, _mm256_mul_ps, _mm256_set1_ps);
impl_binary_operator!(FVec8, Div, div, _mm256_div_ps, _mm256_set1_ps);
impl_binary_operator!(FVec8, Add, add, _mm256_add_ps, _mm256_set1_ps);
impl_binary_operator!(FVec8, Sub, sub, _mm256_sub_ps, _mm256_set1_ps);

// fvec_x86.rs
