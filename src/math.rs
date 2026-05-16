//! Standard analytic geometry primitive module

use std::ops::{Add, BitXor, Div, Mul, Neg, Rem, RemAssign, Sub};

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

macro_rules! impl_matn_vecn {
    ($DIM: expr, $Mat: ident, $Vec: ident { $($x: ident),* }) => {
        #[doc = concat!(stringify!($DIM), "-component generic vector")]
        #[derive(Debug)]
        pub struct $Vec<T> {
            $($x: T),*
        }

        impl<T: Clone + From<i8>> $Vec<T> {
            /// Produce vector filled with zeros.
            pub fn zero() -> Self {
                Self::broadcast(From::from(0))
            }
        }

        impl<T: Clone> Clone for $Vec<T> {
            fn clone(&self) -> Self {
                Self { $($x: self.$x.clone()),* }
            }
        }

        impl<T: Copy> Copy for $Vec<T> {}

        impl<T: Default> Default for $Vec<T> {
            fn default() -> Self {
                Self { $($x: T::default()),* }
            }
        }

        impl<T> $Vec<T> {
            /// Amount of vector dimensions
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

            /// Convert vector into array
            pub fn into_array(self) -> [T; $DIM] {
                let Self { $($x),* } = self;
                [$($x),*]
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

        impl<T> From<[T; $DIM]> for $Vec<T> {
            fn from(arr: [T; $DIM]) -> $Vec<T> {
                $Vec::<T>::from_array(arr)
            }
        }

        impl<T> From<$Vec<T>> for [T; $DIM] {
            fn from(v: $Vec<T>) -> [T; $DIM] {
                v.into_array()
            }
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
            /// Vector squared length
            pub fn length2(self) -> T {
                self.clone() ^ self.clone()
            }
        }

        /// Implement float-specific functions
        macro_rules! for_float {
            ($T: ty) => {
                impl $Vec<$T> {
                    /// Calculate vector length
                    pub fn length(self) -> $T {
                        self.length2().sqrt()
                    }

                    /// Get normalized vector
                    pub fn normalized(self) -> Self {
                        self / self.length().into()
                    }

                    /// Per-component mul-add function. Hints compiler to use FMA.
                    pub fn mul_add(self, mul: Self, add: Self) -> Self {
                        Self::new($(self.$x.mul_add(mul.$x, add.$x)),*)
                    }
                }
            };
        }

        for_float!(f32);
        for_float!(f64);

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

        #[doc = concat!(stringify!($DIM), "x", stringify!($DIM), "column-major generic matrix")]
        pub struct $Mat<T> {
            /// Matrix contents
            pub data: [[T; $DIM]; $DIM],
        }

        impl<T: Clone> Clone for $Mat<T> {
            fn clone(&self) -> Self {
                Self {
                    data: self.data.clone(),
                }
            }
        }

        impl<T: Copy> Copy for $Mat<T> {}

        impl<T> $Mat<T> {
            /// Construct matrix from column vector array
            pub fn from_cols(cols: [$Vec<T>; $DIM]) -> Self {
                Self { data: cols.map($Vec::into_array)  }
            }

            /// Transpose matrix
            pub fn transposed(mut self) -> Self where T: Copy {
                for i in 0..$DIM {
                    for j in i + 1..$DIM {
                        let tmp = self.data[i][j];
                        self.data[i][j] = self.data[j][i];
                        self.data[j][i] = tmp;
                    }
                }

                self
            }
        }
    }
}

impl_matn_vecn!(2, Mat2, Vec2 { x, y });
impl_matn_vecn!(3, Mat3, Vec3 { x, y, z });
impl_matn_vecn!(4, Mat4, Vec4 { x, y, z, w });

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

pub type Vec2f = Vec2<f32>;
pub type Vec3f = Vec3<f32>;
pub type Vec4f = Vec4<f32>;

pub type Mat2f = Mat2<f32>;
pub type Mat3f = Mat3<f32>;
pub type Mat4f = Mat4<f32>;

impl<T: Copy + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T>> Mat3<T> {
    /// Calculate generic NxN matrix determinant
    pub fn determinant(&self) -> T {
        let [[e00, e10, e20], [e01, e11, e21], [e02, e12, e22]] = self.data.clone();
          e00 * e11 * e22 + e01 * e12 * e20 + e02 * e10 * e21 - e10 * e01 * e22 - e02 * e11 * e20 - e12 * e21 * e00
    }
}

impl<T: Copy + Neg<Output = T> + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Div<T, Output = T>> Mat3<T> {
    fn inverse_impl(&self, det: T) -> Mat3<T> {
        let [[e00, e10, e20], [e01, e11, e21], [e02, e12, e22]] = self.data.clone();
        Self {
            data: [
                [
                     (e11 * e22 - e12 * e21) / det,
                    -(e10 * e22 - e12 * e20) / det,
                     (e10 * e21 - e11 * e20) / det,
                ],
                [
                    -(e01 * e22 - e02 * e21) / det,
                     (e00 * e22 - e02 * e20) / det,
                    -(e00 * e21 - e01 * e20) / det,
                ],
                [
                     (e01 * e12 - e02 * e11) / det,
                    -(e00 * e12 - e02 * e10) / det,
                     (e00 * e11 - e01 * e10) / det,
                ]
            ],
        }
    }

    pub fn inversed_unchecked(&self) -> Mat3<T> {
        self.inverse_impl(self.determinant())
    }

}

impl Mat3<f32> {
    pub fn inversed(&self) -> Option<Mat3<f32>> {
        let det = self.determinant();
        if det.abs() < f32::EPSILON {
            return None;
        }
        Some(self.inverse_impl(det))
    }
}

impl<T: Copy + Add<T, Output = T> + Mul<T, Output = T>> Mul<Vec3<T>> for Mat3<T> {
    type Output = Vec3<T>;

    fn mul(self, rhs: Vec3<T>) -> Self::Output {
        Vec3::<T> {
            x: self.data[0][0] * rhs.x + self.data[1][0] * rhs.y + self.data[2][0] * rhs.z,
            y: self.data[0][1] * rhs.x + self.data[1][1] * rhs.y + self.data[2][1] * rhs.z,
            z: self.data[0][2] * rhs.x + self.data[1][2] * rhs.y + self.data[2][2] * rhs.z,
        }
    }
}

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
                    + othr.data[0][$j] * self.data[$i][0]
                    + othr.data[1][$j] * self.data[$i][1]
                    + othr.data[2][$j] * self.data[$i][2]
                    + othr.data[3][$j] * self.data[$i][3];
            }
        }

        mat4_foreach!(mul);

        Self { data }
    }
}

/// Pack of operators definition module
impl Mat4<f32> {
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
        // // Minor DETerminant
        // macro_rules! mdet {
        //     ($i0: expr, $j0: expr) => {
        //         {
        //             macro_rules! iskip {
        //                 ($v: expr, $sk: expr) => {
        //                     ($v - ($v >= $sk) as usize)
        //                 };
        //             }

        //             macro_rules! ind {
        //                 ($i: expr, $j: expr) => {
        //                     self.data[iskip!($i, $i0)][iskip!($j, $j0)]
        //                 };
        //             }

        //             ind!(0, 0) * ind!(1, 1) * ind!(2, 2) + ind!(0, 1) * ind!(1, 2) * ind!(2, 0) + ind!(0, 2) * ind!(1, 0) * ind!(2, 1)
        //                 - ind!(0, 0) * ind!(1, 2) * ind!(2, 1) - ind!(0, 1) * ind!(1, 0) * ind!(2, 2) - ind!(0, 2) * ind!(1, 1) * ind!(2, 0)
        //         }
        //     };
        // }

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
    /// Construct identity matrix
    pub const fn identity() -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Get matrix of rotation along X axis
    pub fn rotate_x(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();

        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cos, sin, 0.0],
                [0.0, -sin, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Get matrix of rotation along Y axis
    pub fn rotate_y(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();

        Self {
            data: [
                [cos, 0.0, -sin, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [sin, 0.0, cos, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Get matrix of rotation along Z axis
    pub fn rotate_z(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();

        Self {
            data: [
                [cos, sin, 0.0, 0.0],
                [-sin, cos, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

impl Mat4<f32> {
    /// Create orthographic projection matrix
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
    }

    /// Create frustum projection matrix assuming far plane being located on infinity
    pub fn projection_frustum_inf_far(l: f32, r: f32, b: f32, t: f32, n: f32) -> Mat4<f32> {
        Self {
            data: [
                [2.0 * n / (r - l), 0.0,               0.0,  0.0  ],
                [0.0,               2.0 * n / (t - b), 0.0,  0.0  ],
                [(r + l) / (r - l), (t + b) / (t - b), -1.0, -1.0 ],
                [0.0,               0.0,               0.0,  0.0  ],
            ],
        }
    }

    /// Create standard frustum projection matrix
    pub fn projection_frustum(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4<f32> {
        Self {
            data: [
                [2.0 * n / (r - l), 0.0,               0.0,                    0.0  ],
                [0.0,               2.0 * n / (t - b), 0.0,                    0.0  ],
                [(r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n),     -1.0 ],
                [0.0,               0.0,               -2.0 * n * f / (f - n), 0.0  ],
            ],
        }
    }

    /// Create view matrix
    pub fn view(loc: Vec3<f32>, at: Vec3<f32>, approx_up: Vec3<f32>) -> Mat4<f32> {
        let dir = (at - loc).normalized();
        let right = dir.cross(approx_up).normalized();
        let up = right.cross(dir).normalized();

        let ld = Vec3::new(right, up, dir).map(|v| -v.dot(loc));

        Self {
            data: [
                [right.x, up.x, -dir.x, 0.0],
                [right.y, up.y, -dir.y, 0.0],
                [right.z, up.z, -dir.z, 0.0],
                [   ld.x, ld.y, - ld.z, 1.0],
            ],
        }
    }
}

impl Mat4<f32> {
    /// Calculate matrix of rotation along any axis
    pub fn rotate(angle: f32, axis: Vec3<f32>) -> Self {
        let a = axis.normalized();
        let (s, c) = angle.sin_cos();

        Self {
            data: [
                [
                    a.x * a.x * (1.0 - c) + c,
                    a.x * a.y * (1.0 - c) - a.z * s,
                    a.x * a.z * (1.0 - c) + a.y * s,
                    0.0,
                ],
                [
                    a.y * a.x * (1.0 - c) + a.z * s,
                    a.y * a.y * (1.0 - c) + c,
                    a.y * a.z * (1.0 - c) - a.x * s,
                    0.0,
                ],
                [
                    a.z * a.x * (1.0 - c) - a.y * s,
                    a.z * a.y * (1.0 - c) + a.x * s,
                    a.z * a.z * (1.0 - c) + c,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Build scale matrix
    pub fn scale(s: Vec3<f32>) -> Self {
        Self {
            data: [
                [s.x, 0.0, 0.0, 0.0],
                [0.0, s.y, 0.0, 0.0],
                [0.0, 0.0, s.z, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Build translation matrix
    pub fn translate(t: Vec3<f32>) -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [t.x, t.y, t.z, 1.0],
            ],
        }
    }

    /// Transform vector using affine part only
    pub fn transform_aff(&self, v: Vec3<f32>) -> Vec3<f32> {
        Vec3 {
            x: v.x * self.data[0][0] + v.y * self.data[1][0] + v.z * self.data[2][0],
            y: v.x * self.data[0][1] + v.y * self.data[1][1] + v.z * self.data[2][1],
            z: v.x * self.data[0][2] + v.y * self.data[1][2] + v.z * self.data[2][2],
        }
    }

    /// Transform vector using homogeneous part only assuming W=1
    pub fn transform_hom(&self, v: Vec3<f32>) -> Vec3<f32> {
        Vec3 {
            x: v.x * self.data[0][0] + v.y * self.data[1][0] + v.z * self.data[2][0] + self.data[3][0],
            y: v.x * self.data[0][1] + v.y * self.data[1][1] + v.z * self.data[2][1] + self.data[3][1],
            z: v.x * self.data[0][2] + v.y * self.data[1][2] + v.z * self.data[2][2] + self.data[3][2],
        }
    }

    /// Transform vector dividing result by W assuming initial W=1
    pub fn transform(&self, v: Vec3<f32>) -> Vec3<f32> {
        let w = v.x * self.data[0][3] + v.y * self.data[1][3] + v.z * self.data[2][3] + self.data[3][3];

        Vec3::new(
            v.x * self.data[0][0] + v.y * self.data[1][0] + v.z * self.data[2][0] + self.data[3][0],
            v.x * self.data[0][1] + v.y * self.data[1][1] + v.z * self.data[2][1] + self.data[3][1],
            v.x * self.data[0][2] + v.y * self.data[1][2] + v.z * self.data[2][2] + self.data[3][2],
        ) / w.into()
    }

    /// Multiply matrix by vector
    pub fn mul_v(&self, v: Vec4<f32>) -> Vec4<f32> {
        Vec4 {
            x: v.x * self.data[0][0] + v.y * self.data[1][0] + v.z * self.data[2][0] + v.w * self.data[3][0],
            y: v.x * self.data[0][1] + v.y * self.data[1][1] + v.z * self.data[2][1] + v.w * self.data[3][1],
            z: v.x * self.data[0][2] + v.y * self.data[1][2] + v.z * self.data[2][2] + v.w * self.data[3][2],
            w: v.x * self.data[0][3] + v.y * self.data[1][3] + v.z * self.data[2][3] + v.w * self.data[3][3],
        }
    }
}

/// Quaternion
pub struct Quat<T> {
    pub w: T,
    pub x: T,
    pub y: T,
    pub z: T,
}
