/// Standard analytic geometry primitive module

use std::ops::{Add, BitXor, Div, Mul, Neg, Rem, RemAssign, Sub};

/// CPU-dependent 3/4-component vector implementation
#[cfg(target_feature = "sse")]
pub mod fvec_x86;

#[cfg(target_feature = "sse")]
impl From<Vec4f> for fvec_x86::FVec4 {
    fn from(value: Vec4f) -> Self {
        Self::new(value.x(), value.y(), value.z(), value.w())
    }
}

#[cfg(target_feature = "sse")]
pub mod fmat4_x86;

#[cfg(target_feature = "sse")]
pub type FVec4 = fvec_x86::FVec4;

#[cfg(not(target_feature = "sse"))]
pub use fallback_fvec4::*;

#[cfg(not(target_feature = "sse"))]
mod fallback_fvec4 {
    use crate::math::Vec4;

    /// Fallback FVec4 value
    #[allow(unused)]
    pub type FVec4 = Vec4<f32>;

    impl Vec4<f32> {
        /// Construct zero vector
        pub fn zero() -> Self {
            Self::broadcast(0.0)
        }

        /// Multiply and add self
        pub fn mul_add(self, mul: Self, add: Self) -> Self {
            self * mul + add
        }
    }

    impl Default for Vec4<f32> {
        fn default() -> Self {
            Self::zero()
        }
    }
}

pub mod numeric_traits {
    pub trait Sqrt {
        fn sqrt(self) -> Self;
    }

    impl Sqrt for f32 {
        fn sqrt(self) -> Self {
            self.sqrt()
        }
    }

    impl Sqrt for f64 {
        fn sqrt(self) -> Self {
            self.sqrt()
        }
    }
}

macro_rules! operator_on_variadic {
    ($operator: tt, $first: expr) => {
        $first
    };

    ($operator: tt, $first: expr, $($rest: expr),*) => {
        $first $operator operator_on_variadic!($operator, $($rest),*)
    };
}

macro_rules! cfoldl1 {
    ($f: expr, $head: expr, $($tail: expr),*) => {
        {
            let mut _r = $head;
            $( _r = $f(_r, $tail); )*
            _r
        }
    };
}

macro_rules! impl_vecn {
    ($DIM: expr, $Vec: ident { $($x: ident),* }) => {
        #[doc = concat!(stringify!($DIM), "-component generic vector")]
        #[derive(Debug)]
        pub struct $Vec<T> {
            $($x: T),*
        }

        impl<T: Clone> Clone for $Vec<T> {
            fn clone(&self) -> Self {
                Self { $($x: self.$x.clone()),* }
            }
        }

        impl<T: Copy> Copy for $Vec<T> {}

        impl<T> $Vec<T> {
            pub const DIM: usize = $DIM;

            /// Construct new vector
            pub const fn new($($x: T),*) -> Self {
                Self { $($x),* }
            }

            /// Construct vector from array
            pub fn from_array(arr: [T; $DIM]) -> Self {
                let [$($x),*] = arr;
                Self { $($x),* }
            }

            /// Convert vector from one type to another
            pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> $Vec<U> {
                $Vec::<U> { $($x: f(self.$x)),* }
            }

            /// Zip two vectors with some function
            pub fn zip<U, V, F: FnMut(T, U) -> V>(self, othr: $Vec<U>, mut f: F) -> $Vec<V> {
                $Vec::<V> { $($x: f(self.$x, othr.$x)),* }
            }

            /// Perform left fold on vector contents
            pub fn fold<U, F: FnMut(U, T) -> U>(self, mut u: U, mut f: F) -> U {
                $( u = f(u, self.$x); )*
                u
            }

            /// Perform left fold without first element
            pub fn fold1<F: FnMut(T, T) -> T>(self, mut f: F) -> T {
                cfoldl1!(f, $(self.$x),*)
            }

            $(
                #[doc = concat!("Extract ", stringify!($x), " component from vector")]
                pub fn $x(self) -> T {
                    self.$x
                }
            )*
        }

        impl<T: Clone> $Vec<T> {
            /// Broadcast value to the vector
            pub fn broadcast(v: T) -> Self {
                Self { $($x: v.clone()),* }
            }
        }

        impl<T: Clone> From<T> for $Vec<T> {
            fn from(v: T) -> $Vec<T> {
                $Vec::<T>::broadcast(v)
            }
        }

        impl<T: Add<T, Output = T> + Mul<T, Output = T>> BitXor for $Vec<T> {
            type Output = T;

            fn bitxor(self, rhs: $Vec<T>) -> Self::Output {
                self.dot(rhs)
            }
        }

        impl<T> $Vec<T> {
            /// Calculate dot product of two vectors
            pub fn dot<U, V>(self, othr: $Vec<U>) -> V
            where
                V: std::ops::Add<V, Output = V>,
                T: std::ops::Mul<U, Output = V>
            {
                operator_on_variadic!(+, $(self.$x * othr.$x),*)
            }
        }

        impl<T: Add<T, Output = T> + Mul<T, Output = T> + Clone> $Vec<T> {
            pub fn length2(&self) -> T {
                self.clone() ^ self.clone()
            }
        }

        impl<T: Add<T, Output = T> + Mul<T, Output = T> + Clone + numeric_traits::Sqrt> $Vec<T> {
            pub fn length(&self) -> T {
                self.length2().sqrt()
            }
        }

        impl<T: Add<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T> + Clone + numeric_traits::Sqrt> $Vec<T> {
            pub fn normalized(&self) -> Self {
                let len = self.length();

                Self { $( $x: self.$x.clone() / len.clone() ),* }
            }

            pub fn normalize(&mut self) {
                let len = self.length();

                $( self.$x = self.$x.clone() / len.clone(); )*
            }
        }

        macro_rules! binary_operator {
            ($Op: ident, $op: ident, $AOp: ident, $aop: ident) => {
                impl<T, U, V> std::ops::$Op<$Vec<U>> for $Vec<T>
                where
                    T: std::ops::$Op<U, Output = V>
                {
                    type Output = $Vec<V>;

                    fn $op(self, othr: $Vec<U>) -> $Vec<V> {
                        $Vec::<V> { $($x: std::ops::$Op::$op(self.$x, othr.$x)),* }
                    }
                }

                impl<T, U> std::ops::$AOp<$Vec<U>> for $Vec<T>
                where
                    T: std::ops::$AOp<U>
                {
                    fn $aop(&mut self, othr: $Vec<U>) {
                        $( std::ops::$AOp::$aop(&mut self.$x, othr.$x); )*
                    }
                }
            };
        }

        binary_operator!(Add, add, AddAssign, add_assign);
        binary_operator!(Sub, sub, SubAssign, sub_assign);
        binary_operator!(Mul, mul, MulAssign, mul_assign);
        binary_operator!(Div, div, DivAssign, div_assign);

        impl<T, U> std::ops::Neg for $Vec<T>
        where
            T: std::ops::Neg<Output = U>
        {
            type Output = $Vec<U>;

            fn neg(self) -> $Vec<U> {
                $Vec::<U> { $($x: -self.$x),* }
            }
        }
    }
}

impl_vecn!(2, Vec2 { x, y });
impl_vecn!(3, Vec3 { x, y, z });
impl_vecn!(4, Vec4 { x, y, z, w });

impl<T: Clone + Mul<T, Output = T> + Sub<T, Output = T>> Vec3<T> {
    pub fn cross(self, rhs: Self) -> Vec3<T> {
        Vec3::<T> {
            x: self.y.clone() * rhs.z.clone() - self.z.clone() * rhs.y.clone(),
            y: self.z * rhs.x.clone() - self.x.clone() * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl<T: Clone + Mul<T, Output = T> + Sub<T, Output = T>> Rem for Vec3<T> {
    type Output = Self;
 
    fn rem(self, rhs: Self) -> Self::Output {
        self.cross(rhs)
    }
}

impl<T: Clone + Mul<T, Output = T> + Sub<T, Output = T>> RemAssign for Vec3<T> {
    fn rem_assign(&mut self, rhs: Self) {
        *self = self.clone() % rhs;
    }
}

impl<T: Clone + Mul<T, Output = T> + Sub<T, Output = T>> Rem for Vec2<T> {
    type Output = T;
    fn rem(self, rhs: Self) -> Self::Output {
        self.x * rhs.y - self.y * rhs.x
    }
}


/// 2-component generic matrix
#[derive(Copy, Clone)]
pub struct Mat2<T> {
    pub e00: T,
    pub e01: T,
    pub e10: T,
    pub e11: T,
}

impl<T: Mul<T, Output = T> + Sub<T, Output = T>> Mat2<T> {
    pub fn new(e00: T, e01: T, e10: T, e11: T) -> Self {
        Self { e00, e01, e10, e11 }
    }

    pub fn determinant(self) -> T {
        self.e00 * self.e11 - self.e01 * self.e10
    }
}

impl Mat2<f32> {
    pub fn inversed(self) -> Option<Self> {
        let determinant = self.determinant();

        if determinant.abs() < f32::EPSILON {
            return None;
        }

        Some(Self {
            e00:  self.e11 / determinant,
            e01: -self.e01 / determinant,
            e10: -self.e10 / determinant,
            e11:  self.e00 / determinant,
        })
    }
}

impl Mul<Vec2f> for Mat2f {
    type Output = Vec2f;

    fn mul(self, rhs: Vec2f) -> Self::Output {
        Self::Output {
            x: rhs.x * self.e00 + rhs.y * self.e10,
            y: rhs.x * self.e01 + rhs.y * self.e11,
        }
    }
}

pub struct Mat3<T> {
    pub e00: T,
    pub e01: T,
    pub e02: T,
    pub e10: T,
    pub e11: T,
    pub e12: T,
    pub e20: T,
    pub e21: T,
    pub e22: T,
}

impl<T> Mat3<T> {
    pub fn new(
        e00: T, e01: T, e02: T,
        e10: T, e11: T, e12: T,
        e20: T, e21: T, e22: T,
    ) -> Self {
        Self {
            e00, e01, e02,
            e10, e11, e12,
            e20, e21, e22,
        }
    }

    pub fn from_rows(r0: Vec3<T>, r1: Vec3<T>, r2: Vec3<T>) -> Self {
        Self {
            e00: r0.x, e01: r0.y, e02: r0.z,
            e10: r1.x, e11: r1.y, e12: r1.z,
            e20: r2.x, e21: r2.y, e22: r2.z,
        }
    }

    pub fn from_cols(c0: Vec3<T>, c1: Vec3<T>, c2: Vec3<T>) -> Self {
        Self {
            e00: c0.x, e01: c1.x, e02: c2.x,
            e10: c0.y, e11: c1.y, e12: c2.y,
            e20: c0.z, e21: c1.z, e22: c2.z,
        }
    }
}

impl<T: Copy + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T>> Mat3<T> {
    pub fn determinant(&self) -> T {
          self.e00 * self.e11 * self.e22 + self.e01 * self.e12 * self.e20 + self.e02 * self.e10 * self.e21
        - self.e10 * self.e01 * self.e22 - self.e02 * self.e11 * self.e20 - self.e12 * self.e21 * self.e00
    }
}

impl<T: Copy + Neg<Output = T> + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Div<T, Output = T>> Mat3<T> {
    unsafe fn inverse_impl(&self, nonzero_determinant: T) -> Mat3<T> {
        Self {
            e00:  (self.e11 * self.e22 - self.e12 * self.e21) / nonzero_determinant,
            e01: -(self.e01 * self.e22 - self.e02 * self.e21) / nonzero_determinant,
            e02:  (self.e01 * self.e12 - self.e02 * self.e11) / nonzero_determinant,
            e10: -(self.e10 * self.e22 - self.e12 * self.e20) / nonzero_determinant,
            e11:  (self.e00 * self.e22 - self.e02 * self.e20) / nonzero_determinant,
            e12: -(self.e00 * self.e12 - self.e02 * self.e10) / nonzero_determinant,
            e20:  (self.e10 * self.e21 - self.e11 * self.e20) / nonzero_determinant,
            e21: -(self.e00 * self.e21 - self.e01 * self.e20) / nonzero_determinant,
            e22:  (self.e00 * self.e11 - self.e01 * self.e10) / nonzero_determinant,
        }
    }

    pub fn inversed_unchecked(&self) -> Mat3<T> {
        unsafe {
            self.inverse_impl(self.determinant())
        }
    }

}

impl Mat3<f32> {
    pub fn inversed(&self) -> Option<Mat3<f32>> {
        let det = self.determinant();
        if det.abs() < f32::EPSILON {
            return None;
        }
        Some(unsafe { self.inverse_impl(det) })
    }
}

impl<T: Copy + Add<T, Output = T> + Mul<T, Output = T>> Mul<Vec3<T>> for Mat3<T> {
    type Output = Vec3<T>;

    fn mul(self, rhs: Vec3<T>) -> Self::Output {
        Self::Output {
            x: self.e00 * rhs.x + self.e01 * rhs.y + self.e02 * rhs.z,
            y: self.e10 * rhs.x + self.e11 * rhs.y + self.e12 * rhs.z,
            z: self.e20 * rhs.x + self.e21 * rhs.y + self.e22 * rhs.z,
        }
    }
}

pub type Vec2f = Vec2<f32>;
pub type Vec3f = Vec3<f32>;
pub type Vec4f = Vec4<f32>;

pub type Mat2f = Mat2<f32>;
pub type Mat3f = Mat3<f32>;
pub type Mat4f = Mat4<f32>;

impl Vec2f {
    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl Vec3f {
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

impl Vec3f {
    /// Get vector component by index
    pub fn get<const I: u32>(self) -> f32 {
        match I {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => panic!("invalid vector component index \"{}\" (expected number in [0, 2] range)", I),
        }
    }
}

pub struct Mat4<T> {
    pub data: [[T; 4]; 4],
}

impl<T: Clone> Clone for Mat4<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl<T: Copy> Copy for Mat4<T> {}

// Compile-time for for 4x4 matrix
macro_rules! mat4_foreach {
    ($action: ident) => {
        $action!(0, 0); $action!(0, 1); $action!(0, 2); $action!(0, 3);
        $action!(1, 0); $action!(1, 1); $action!(1, 2); $action!(1, 3);
        $action!(2, 0); $action!(2, 1); $action!(2, 2); $action!(2, 3);
        $action!(3, 0); $action!(3, 1); $action!(3, 2); $action!(3, 3);
    };
}

impl Mul<Mat4<f32>> for Mat4<f32> {
    type Output = Mat4<f32>;

    fn mul(self, othr: Mat4<f32>) -> Self::Output {
        // Resulting data
        let mut data = [[0f32; 4]; 4];

        macro_rules! mul {
            ($i: expr, $j: expr) => {
                data[$i][$j] = 0.0
                    + self.data[0][$j] * othr.data[$i][0]
                    + self.data[1][$j] * othr.data[$i][1]
                    + self.data[2][$j] * othr.data[$i][2]
                    + self.data[3][$j] * othr.data[$i][3]
                ;
            }
        }
        mat4_foreach!(mul);

        Self { data }
    }
}

/// Pack of operators definition module
impl Mat4<f32>
{
    /// Determinant getting function
    /// * Returns determinant of this matrix
    pub fn determinant(&self) -> f32 {
        0.0
            + self.data[0][0]
                * (self.data[1][1] * self.data[2][2] * self.data[3][3]
                    + self.data[1][2] * self.data[2][3] * self.data[3][1]
                    + self.data[1][3] * self.data[2][1] * self.data[3][2]
                    - self.data[1][1] * self.data[2][3] * self.data[3][2]
                    - self.data[1][2] * self.data[2][1] * self.data[3][3]
                    - self.data[1][3] * self.data[2][2] * self.data[3][1])
            - self.data[0][1]
                * (self.data[0][1] * self.data[2][2] * self.data[3][3]
                    + self.data[0][2] * self.data[2][3] * self.data[3][1]
                    + self.data[0][3] * self.data[2][1] * self.data[3][2]
                    - self.data[0][1] * self.data[2][3] * self.data[3][2]
                    - self.data[0][2] * self.data[2][1] * self.data[3][3]
                    - self.data[0][3] * self.data[2][2] * self.data[3][1])
            + self.data[0][2]
                * (self.data[0][1] * self.data[1][2] * self.data[3][3]
                    + self.data[0][2] * self.data[1][3] * self.data[3][1]
                    + self.data[0][3] * self.data[1][1] * self.data[3][2]
                    - self.data[0][1] * self.data[1][3] * self.data[3][2]
                    - self.data[0][2] * self.data[1][1] * self.data[3][3]
                    - self.data[0][3] * self.data[1][2] * self.data[3][1])
            - self.data[0][3]
                * (self.data[0][1] * self.data[1][2] * self.data[2][3]
                    + self.data[0][2] * self.data[1][3] * self.data[2][1]
                    + self.data[0][3] * self.data[1][1] * self.data[2][2]
                    - self.data[0][1] * self.data[1][3] * self.data[2][2]
                    - self.data[0][2] * self.data[1][1] * self.data[2][3]
                    - self.data[0][3] * self.data[1][2] * self.data[2][1])
    } // fn determinant

    /// Matrix inversion getting function
    /// * Returns this matrix inersed
    pub fn inversed(&self) -> Self {
        let determ_00 = self.data[1][1] * self.data[2][2] * self.data[3][3]
            + self.data[1][2] * self.data[2][3] * self.data[3][1]
            + self.data[1][3] * self.data[2][1] * self.data[3][2]
            - self.data[1][1] * self.data[2][3] * self.data[3][2]
            - self.data[1][2] * self.data[2][1] * self.data[3][3]
            - self.data[1][3] * self.data[2][2] * self.data[3][1];
        let determ_01 = self.data[0][1] * self.data[2][2] * self.data[3][3]
            + self.data[0][2] * self.data[2][3] * self.data[3][1]
            + self.data[0][3] * self.data[2][1] * self.data[3][2]
            - self.data[0][1] * self.data[2][3] * self.data[3][2]
            - self.data[0][2] * self.data[2][1] * self.data[3][3]
            - self.data[0][3] * self.data[2][2] * self.data[3][1];
        let determ_02 = self.data[0][1] * self.data[1][2] * self.data[3][3]
            + self.data[0][2] * self.data[1][3] * self.data[3][1]
            + self.data[0][3] * self.data[1][1] * self.data[3][2]
            - self.data[0][1] * self.data[1][3] * self.data[3][2]
            - self.data[0][2] * self.data[1][1] * self.data[3][3]
            - self.data[0][3] * self.data[1][2] * self.data[3][1];
        let determ_03 = self.data[0][1] * self.data[1][2] * self.data[2][3]
            + self.data[0][2] * self.data[1][3] * self.data[2][1]
            + self.data[0][3] * self.data[1][1] * self.data[2][2]
            - self.data[0][1] * self.data[1][3] * self.data[2][2]
            - self.data[0][2] * self.data[1][1] * self.data[2][3]
            - self.data[0][3] * self.data[1][2] * self.data[2][1];

        let determ = self.data[0][0] * determ_00 - self.data[0][1] * determ_01
            + self.data[0][2] * determ_02
            - self.data[0][3] * determ_03;

        Self {
            data: [
                [
                    self.data[0][0] * determ_00 / determ,
                    -self.data[0][1] * determ_01 / determ,
                    self.data[0][2] * determ_02 / determ,
                    -self.data[0][3] * determ_03 / determ,
                ],
                [
                    -self.data[1][0]
                        * (self.data[0][1] * self.data[2][2] * self.data[3][3]
                            + self.data[0][2] * self.data[2][3] * self.data[3][1]
                            + self.data[0][3] * self.data[2][1] * self.data[3][2]
                            - self.data[0][1] * self.data[2][3] * self.data[3][2]
                            - self.data[0][2] * self.data[2][1] * self.data[3][3]
                            - self.data[0][3] * self.data[2][2] * self.data[3][1])
                        / determ,
                    self.data[1][1]
                        * (self.data[0][0] * self.data[2][2] * self.data[3][3]
                            + self.data[0][2] * self.data[2][3] * self.data[3][0]
                            + self.data[0][3] * self.data[2][0] * self.data[3][2]
                            - self.data[0][0] * self.data[2][3] * self.data[3][2]
                            - self.data[0][2] * self.data[2][0] * self.data[3][3]
                            - self.data[0][3] * self.data[2][2] * self.data[3][0])
                        / determ,
                    -self.data[1][2]
                        * (self.data[0][0] * self.data[2][1] * self.data[3][3]
                            + self.data[0][1] * self.data[2][3] * self.data[3][0]
                            + self.data[0][3] * self.data[2][0] * self.data[3][1]
                            - self.data[0][0] * self.data[2][3] * self.data[3][1]
                            - self.data[0][1] * self.data[2][0] * self.data[3][3]
                            - self.data[0][3] * self.data[2][1] * self.data[3][0])
                        / determ,
                    self.data[1][3]
                        * (self.data[0][0] * self.data[2][1] * self.data[3][2]
                            + self.data[0][1] * self.data[2][2] * self.data[3][0]
                            + self.data[0][2] * self.data[2][0] * self.data[3][1]
                            - self.data[0][0] * self.data[2][2] * self.data[3][1]
                            - self.data[0][1] * self.data[2][0] * self.data[3][2]
                            - self.data[0][2] * self.data[2][1] * self.data[3][0])
                        / determ,
                ],
                [
                    self.data[2][0]
                        * (self.data[0][1] * self.data[1][2] * self.data[3][3]
                            + self.data[0][2] * self.data[1][3] * self.data[3][1]
                            + self.data[0][3] * self.data[1][1] * self.data[3][2]
                            - self.data[0][1] * self.data[1][3] * self.data[3][2]
                            - self.data[0][2] * self.data[1][1] * self.data[3][3]
                            - self.data[0][3] * self.data[1][2] * self.data[3][1])
                        / determ,
                    -self.data[2][1]
                        * (self.data[0][0] * self.data[1][2] * self.data[3][3]
                            + self.data[0][2] * self.data[1][3] * self.data[3][0]
                            + self.data[0][3] * self.data[1][0] * self.data[3][2]
                            - self.data[0][0] * self.data[1][3] * self.data[3][2]
                            - self.data[0][2] * self.data[1][0] * self.data[3][3]
                            - self.data[0][3] * self.data[1][2] * self.data[3][0])
                        / determ,
                    self.data[2][2]
                        * (self.data[0][0] * self.data[1][1] * self.data[3][3]
                            + self.data[0][1] * self.data[1][3] * self.data[3][0]
                            + self.data[0][3] * self.data[1][0] * self.data[3][1]
                            - self.data[0][0] * self.data[1][3] * self.data[3][1]
                            - self.data[0][1] * self.data[1][0] * self.data[3][3]
                            - self.data[0][3] * self.data[1][1] * self.data[3][0])
                        / determ,
                    -self.data[2][3]
                        * (self.data[0][0] * self.data[1][1] * self.data[3][2]
                            + self.data[0][1] * self.data[1][2] * self.data[3][0]
                            + self.data[0][2] * self.data[1][0] * self.data[3][1]
                            - self.data[0][0] * self.data[1][2] * self.data[3][1]
                            - self.data[0][1] * self.data[1][0] * self.data[3][2]
                            - self.data[0][2] * self.data[1][1] * self.data[3][0])
                        / determ,
                ],
                [
                    -self.data[3][0]
                        * (self.data[0][1] * self.data[1][2] * self.data[2][3]
                            + self.data[0][2] * self.data[1][3] * self.data[2][1]
                            + self.data[0][3] * self.data[1][1] * self.data[2][2]
                            - self.data[0][1] * self.data[1][3] * self.data[2][2]
                            - self.data[0][2] * self.data[1][1] * self.data[2][3]
                            - self.data[0][3] * self.data[1][2] * self.data[2][1])
                        / determ,
                    self.data[3][1]
                        * (self.data[0][0] * self.data[1][2] * self.data[2][3]
                            + self.data[0][2] * self.data[1][3] * self.data[2][0]
                            + self.data[0][3] * self.data[1][0] * self.data[2][2]
                            - self.data[0][0] * self.data[1][3] * self.data[2][2]
                            - self.data[0][2] * self.data[1][0] * self.data[2][3]
                            - self.data[0][3] * self.data[1][2] * self.data[2][0])
                        / determ,
                    -self.data[3][2]
                        * (self.data[0][0] * self.data[1][1] * self.data[2][3]
                            + self.data[0][1] * self.data[1][3] * self.data[2][0]
                            + self.data[0][3] * self.data[1][0] * self.data[2][1]
                            - self.data[0][0] * self.data[1][3] * self.data[2][1]
                            - self.data[0][1] * self.data[1][0] * self.data[2][3]
                            - self.data[0][3] * self.data[1][1] * self.data[2][0])
                        / determ,
                    self.data[3][3]
                        * (self.data[0][0] * self.data[1][1] * self.data[2][2]
                            + self.data[0][1] * self.data[1][2] * self.data[2][0]
                            + self.data[0][2] * self.data[1][0] * self.data[2][1]
                            - self.data[0][0] * self.data[1][2] * self.data[2][1]
                            - self.data[0][1] * self.data[1][0] * self.data[2][2]
                            - self.data[0][2] * self.data[1][1] * self.data[2][0])
                        / determ,
                ],
            ],
        }
    } // fn inversed
} // impl Mat4<f32>

/// Default matrices implementation
impl Mat4<f32> {
    /// Identity matrix getting function
    /// * Returns identity matrix
    pub const fn identity() -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    } // fn identity

    /// Rotation matrix getting function
    /// * `angle` - angle to create rotation matrix on
    /// * Returns rotation matrix
    pub fn rotate_x(angle: f32) -> Self {
        let sin = angle.sin();
        let cos = angle.cos();

        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos, sin, 0.0],
                [0.0, -sin, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    } // fn rotate_x

    /// Rotation matrix getting function
    /// * `angle` - angle to create rotation matrix on
    /// * Returns rotation matrix
    pub fn rotate_y(angle: f32) -> Self {
        let sin = angle.sin();
        let cos = angle.cos();

        Self {
            data: [
                [cos, 0.0, -sin, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [sin, 0.0, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    } // fn rotate_y

    /// Rotation matrix getting function
    /// * `angle` - angle to create rotation matrix on
    /// * Returns rotation matrix
    pub fn rotate_z(angle: f32) -> Self {
        let sin = angle.sin();
        let cos = angle.cos();

        Self {
            data: [
                [cos, sin, 0.0, 0.0],
                [-sin, cos, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    } // fn rotate_z
} // impl Mat4x4<f32>
/// Projection functions implementation
impl Mat4<f32> {
    /// Orthographic projection matrix create function
    /// * `l`, `r` - left and right boundaries
    /// * `b`, `t` - bottom and top
    /// * `n`, `f` - near and far
    /// * Returns projection matrix
    pub fn projection_ortho(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4<f32> {
        Self {
            data: [
                [2.0 / (r - l), 0.0, 0.0, 0.0],
                [0.0, 2.0 / (t - b), 0.0, 0.0],
                [0.0, 0.0, -2.0 / (f - n), 0.0],
                [
                    -(r + l) / (r - l),
                    -(t + b) / (t - b),
                    -(f + n) / (f - n),
                    1.0,
                ],
            ],
        }
    } // fn projection_ortho

    /// Frustum projection matrix create function
    /// # Inputs
    /// * `l`, `r` - left and right boundaries
    /// * `b`, `t` - bottom and top
    /// * `n` - near plane
    /// 
    /// # Result
    /// Projection matrix
    /// 
    /// # Note
    /// Far plane is assumed to be infinite. This
    /// matrix may be used with inverse z technique.
    pub fn projection_frustum_inf_far(l: f32, r: f32, b: f32, t: f32, n: f32) -> Mat4<f32> {
        Self {
            data: [
                [2.0 * n / (r - l), 0.0,               0.0,  0.0  ],
                [0.0,               2.0 * n / (t - b), 0.0,  0.0  ],
                [(r + l) / (r - l), (t + b) / (t - b), -1.0, -1.0 ],
                [0.0,               0.0,               0.0,  0.0  ],
            ],
        }
    } // fn projection_frustum

    /// Frustum projection matrix create function
    /// * `l`, `r` - left and right boundaries
    /// * `b`, `t` - bottom and top
    /// * `n`, `f` - near and far
    /// * Returns projection matrix
    pub fn projection_frustum(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4<f32> {
        Self {
            data: [
                [2.0 * n / (r - l), 0.0,               0.0,                    0.0  ],
                [0.0,               2.0 * n / (t - b), 0.0,                    0.0  ],
                [(r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n),     -1.0 ],
                [0.0,               0.0,               -2.0 * n * f / (f - n), 0.0  ],
            ],
        }
    } // fn projection_frustum

    /// View projection matrix create function
    /// `l`, `r` - left and
    pub fn view(loc: Vec3<f32>, at: Vec3<f32>, approx_up: Vec3<f32>) -> Mat4<f32> {
        let dir = (at - loc).normalized();
        let right = dir.cross(approx_up).normalized();
        let up = right.cross(dir).normalized();

        Self {
            data: [
                [ right.x,         up.x,        -dir.x,        0.0],
                [ right.y,         up.y,        -dir.y,        0.0],
                [ right.z,         up.z,        -dir.z,        0.0],
                [-loc.dot(right), -loc.dot(up),  loc.dot(dir), 1.0],
            ],
        }
    } // fn view
} // impl Mat4x4<f32>

impl Mat4<f32> {
    /// Rotation matrix getting function
    /// * `angle` - angle to create rotation matrix on
    /// * `axis` - axis to create rotation matrix based on
    /// * Returns rotation matrix
    pub fn rotate(angle: f32, mut axis: Vec3<f32>) -> Self {
        axis.normalize();

        let sina = angle.sin();
        let cosa = angle.cos();

        Self {
            data: [
                [
                    axis.x * axis.x * (1.0 - cosa) + cosa,
                    axis.x * axis.y * (1.0 - cosa) - axis.z * sina,
                    axis.x * axis.z * (1.0 - cosa) + axis.y * sina,
                    0.0,
                ],
                [
                    axis.y * axis.x * (1.0 - cosa) + axis.z * sina,
                    axis.y * axis.y * (1.0 - cosa) + cosa,
                    axis.y * axis.z * (1.0 - cosa) - axis.x * sina,
                    0.0,
                ],
                [
                    axis.z * axis.x * (1.0 - cosa) - axis.y * sina,
                    axis.z * axis.y * (1.0 - cosa) + axis.x * sina,
                    axis.z * axis.z * (1.0 - cosa) + cosa,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    } // fn rotate

    /// Scaling function.
    /// * `s` - scale vector
    /// * Returns scale matrix
    pub fn scale(s: Vec3<f32>) -> Self {
        Self {
            data: [
                [s.x, 0.0, 0.0, 0.0],
                [0.0, s.y, 0.0, 0.0],
                [0.0, 0.0, s.z, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    } // fn scale

    /// Translating function.
    /// * `t` - translate vector
    /// * Returns scale matrix
    pub fn translate(t: Vec3<f32>) -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [t.x, t.y, t.z, 1.0],
            ],
        }
    } // fn translate

    /// Transform interpreting this matrix as 3x3.
    pub fn transform_vector(&self, v: Vec3<f32>) -> Vec3<f32> {
        Vec3 {
            x: v.x * self.data[0][0] + v.y * self.data[1][0] + v.z * self.data[2][0],
            y: v.x * self.data[0][1] + v.y * self.data[1][1] + v.z * self.data[2][1],
            z: v.x * self.data[0][2] + v.y * self.data[1][2] + v.z * self.data[2][2],
        }
    }

    /// Transform without calculating W coordinate
    pub fn transform_point(&self, v: Vec3<f32>) -> Vec3<f32> {
        Vec3 {
            x: v.x * self.data[0][0] + v.y * self.data[1][0] + v.z * self.data[2][0] + self.data[3][0],
            y: v.x * self.data[0][1] + v.y * self.data[1][1] + v.z * self.data[2][1] + self.data[3][1],
            z: v.x * self.data[0][2] + v.y * self.data[1][2] + v.z * self.data[2][2] + self.data[3][2],
        }
    }

    /// Standard transformation
    pub fn transform(&self, v: Vec4<f32>) -> Vec4<f32> {
        Vec4 {
            x: v.x * self.data[0][0] + v.y * self.data[1][0] + v.z * self.data[2][0] + v.w * self.data[3][0],
            y: v.x * self.data[0][1] + v.y * self.data[1][1] + v.z * self.data[2][1] + v.w * self.data[3][1],
            z: v.x * self.data[0][2] + v.y * self.data[1][2] + v.z * self.data[2][2] + v.w * self.data[3][2],
            w: v.x * self.data[0][3] + v.y * self.data[1][3] + v.z * self.data[2][3] + v.w * self.data[3][3],
        }
    }

    pub fn transform_4x4(&self, v: Vec3<f32>) -> Vec3<f32> {
        let w =
            v.x * self.data[0][3] + v.y * self.data[1][3] + v.z * self.data[2][3] + self.data[3][3];

        Vec3 {
            x: (v.x * self.data[0][0]
                + v.y * self.data[1][0]
                + v.z * self.data[2][0]
                + self.data[3][0])
                / w,
            y: (v.x * self.data[0][1]
                + v.y * self.data[1][1]
                + v.z * self.data[2][1]
                + self.data[3][1])
                / w,
            z: (v.x * self.data[0][2]
                + v.y * self.data[1][2]
                + v.z * self.data[2][2]
                + self.data[3][2])
                / w,
        }
    } // fn transform_4x4
} // impl Mat4x4

/// Quaternion
pub struct Quat<T> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}

// mod.rs
