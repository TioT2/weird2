use std::ops::{Add, AddAssign, BitXor, Div, DivAssign, Mul, MulAssign, Neg, Range, Rem, RemAssign, Sub, SubAssign};

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

macro_rules! consume_ident {
    ($type: ty, $i: ident) => { $type };
}

macro_rules! impl_vecn_base {
    ($struct_name: ident, $template_type: ident, $value_type: ty, $($x: ident),*) => {
        #[derive(Debug, Default, PartialEq)]
        pub struct $struct_name<$template_type> {
            $( pub $x : $value_type, )*
        }

        impl<$template_type: Clone> Clone for $struct_name<$template_type> where $value_type: Clone {
            fn clone(&self) -> Self {
                Self {
                    $( $x: self.$x.clone() ),*
                }
            }
        }

        impl<$template_type: Copy> Copy for $struct_name<$template_type> where $value_type: Copy {

        }

        impl<$template_type> $struct_name<$template_type> {
            pub fn new($($x: $value_type,)*) -> Self {
                Self { $($x,)* }
            }

            pub fn from_tuple(t: ( $( consume_ident!($value_type, $x) ),* )) -> Self {
                Self::from(t)
            }

            pub fn into_tuple(self) -> ( $( consume_ident!($value_type, $x) ),* ) {
                self.into()
            }
        }

        impl<$template_type> Into<( $( consume_ident!($value_type, $x) ),* )> for $struct_name<$template_type> {
            fn into(self) -> ( $( consume_ident!($value_type, $x) ),* ) {
                ( $( self.$x ),* )
            }
        }

        impl<$template_type> From<( $( consume_ident!($value_type, $x) ),* )> for $struct_name<$template_type> {
            fn from(t: ( $( consume_ident!($value_type, $x) ),* )) -> Self {
                let ($($x),*) = t;

                Self { $($x),* }
            }
        }
    }
}

macro_rules! impl_vecn_binary_operator {
    ($op_name: ident, $op_fn_name: ident, $struct_name: ident, $($x: ident),*) => {
        impl<A: $op_name<Output = A>> $op_name<$struct_name<A>> for $struct_name<A> {
            type Output = $struct_name<A>;

            fn $op_fn_name(self, rhs: $struct_name<A>) -> Self::Output {
                Self::Output {
                    $( $x: $op_name::$op_fn_name(self.$x, rhs.$x), )*
                }
            }
        }

        impl<T: Clone + $op_name<Output = T>> $op_name<T> for $struct_name<T> {
            type Output = $struct_name<T>;

            fn $op_fn_name(self, rhs: T) -> Self::Output {
                Self::Output {
                    $( $x: $op_name::$op_fn_name(self.$x, rhs.clone()), )*
                }
            }
        }
    }
}

macro_rules! impl_vecn_assignment_operator {
    ($op_name: ident, $op_fn_name: ident, $struct_name: ident, $($x: ident),*) => {
        impl<T: $op_name> $op_name<$struct_name<T>> for $struct_name<T> {
            fn $op_fn_name(&mut self, rhs: $struct_name<T>) {
                $( $op_name::<T>::$op_fn_name(&mut self.$x, rhs.$x); )*
            }
        }

        impl<T: Clone + $op_name> $op_name<T> for $struct_name<T> {
            fn $op_fn_name(&mut self, rhs: T) {
                $( $op_name::<T>::$op_fn_name(&mut self.$x, rhs.clone()); )*
            }
        }
    }
}

macro_rules! impl_vecn_unary_operator {
    ($op_name: ident, $op_fn_name: ident, $struct_name: ident, $($x: ident),*) => {
        impl<T: $op_name<Output = T>> $op_name for $struct_name<T> {
            type Output = $struct_name<T>;

            fn $op_fn_name(self) -> Self::Output {
                Self::Output {
                    $( $x: $op_name::$op_fn_name(self.$x), )*
                }
            }
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

macro_rules! impl_vecn {
    ($struct_name: ident, $($x: ident),*) => {
        impl_vecn_base!($struct_name, T, T, $($x),*);

        impl<T: Add<T, Output = T> + Mul<T, Output = T>> BitXor for $struct_name<T> {
            type Output = T;

            fn bitxor(self, rhs: $struct_name<T>) -> Self::Output {
                self.dot(rhs)
            }
        }

        impl<T: Add<T, Output = T> + Mul<T, Output = T>> $struct_name<T> {
            fn dot(self, rhs: $struct_name<T>) -> T {
                operator_on_variadic!(+, $(self.$x * rhs.$x),*)
            }
        }

        impl<T: Add<T, Output = T> + Mul<T, Output = T> + Clone> $struct_name<T> {
            pub fn length2(&self) -> T {
                self.clone() ^ self.clone()
            }
        }

        impl<T: Add<T, Output = T> + Mul<T, Output = T> + Clone + numeric_traits::Sqrt> $struct_name<T> {
            pub fn length(&self) -> T {
                self.length2().sqrt()
            }
        }

        impl<T: Add<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T> + Clone + numeric_traits::Sqrt> $struct_name<T> {
            pub fn normalized(&self) -> Self {
                let len = self.length();

                Self { $( $x: self.$x.clone() / len.clone() ),* }
            }

            pub fn normalize(&mut self) {
                let len = self.length();

                $( self.$x = self.$x.clone() / len.clone(); )*
            }
        }

        impl_vecn_binary_operator!(Add, add, $struct_name, $($x),*);
        impl_vecn_binary_operator!(Sub, sub, $struct_name, $($x),*);
        impl_vecn_binary_operator!(Mul, mul, $struct_name, $($x),*);
        impl_vecn_binary_operator!(Div, div, $struct_name, $($x),*);

        impl_vecn_unary_operator!(Neg, neg, $struct_name, $($x),*);

        impl_vecn_assignment_operator!(AddAssign, add_assign, $struct_name, $($x),*);
        impl_vecn_assignment_operator!(SubAssign, sub_assign, $struct_name, $($x),*);
        impl_vecn_assignment_operator!(MulAssign, mul_assign, $struct_name, $($x),*);
        impl_vecn_assignment_operator!(DivAssign, div_assign, $struct_name, $($x),*);
    }
}

macro_rules! impl_extn {
    ($struct_name: ident, $($x: ident),*) => {
        impl_vecn_base!($struct_name, T, T, $($x),*);
    }
}

macro_rules! impl_rectn {
    ($struct_name: ident, $point_name: ident, $ext_name: ident, $($x: ident),*) => {
        impl_vecn_base!($struct_name, T, Range<T>, $($x),*);

        impl<T> $struct_name<T> where Range<T>: ExactSizeIterator {
            pub fn extent(&self) -> $ext_name<usize> {
                $ext_name::<usize>::new($( self.$x.len() ),*)
            }
        }

        impl<T: Clone> $struct_name<T> {
            pub fn start(&self) -> $point_name<T> {
                $point_name::<T>::new($( self.$x.start.clone() ),*)
            }

            pub fn end(&self) -> $point_name<T> {
                $point_name::<T>::new($( self.$x.end.clone() ),*)
            }
        }
    }
}

impl_vecn!(Vec2, x, y);
impl_vecn!(Vec3, x, y, z);
impl_vecn!(Vec4, x, y, z, w);

impl_extn!(Ext2, w, h);
impl_extn!(Ext3, w, h, d);

impl_rectn!(Rect, Vec2, Ext2, x, y);
impl_rectn!(Box, Vec3, Ext3, x, y, z);

impl Vec3f {
    pub fn checked_normalized(&self) -> Option<Self> {
        let len2 = self.length2();

        if len2 > f32::EPSILON {
            Some(*self / len2.sqrt())
        } else {
            None
        }
    }
}

#[macro_export]
macro_rules! vec3f {
    ($x: expr, $y: expr, $z: expr $(,)?) => {
        math::Vec3f::new($x, $y, $z)
    };
}

#[macro_export]
macro_rules! vec2f {
    ($x: expr, $y: expr $(,)?) => {
        math::Vec2f::new($x, $y)
    };
}

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

pub type Ext2f = Ext2<f32>;
pub type Vec2f = Vec2<f32>;
pub type Vec3f = Vec3<f32>;

pub type Vec2u32 = Vec2<u32>;
pub type Ext2u32 = Ext2<u32>;

pub type Vec2us = Vec2<usize>;
pub type Ext2us = Ext2<usize>;

pub type Mat2f = Mat2<f32>;
pub type Mat3f = Mat3<f32>;
pub type Mat4f = Mat4<f32>;

impl Vec3f {
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
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

impl Mul<Mat4<f32>> for Mat4<f32> {
    type Output = Mat4<f32>;

    fn mul(self, rhs: Mat4<f32>) -> Self::Output {
        Self {
            data: [
                [
                    self.data[0][0].clone() * rhs.data[0][0].clone()
                        + self.data[0][1].clone() * rhs.data[1][0].clone()
                        + self.data[0][2].clone() * rhs.data[2][0].clone()
                        + self.data[0][3].clone() * rhs.data[3][0].clone(),
                    self.data[0][0].clone() * rhs.data[0][1].clone()
                        + self.data[0][1].clone() * rhs.data[1][1].clone()
                        + self.data[0][2].clone() * rhs.data[2][1].clone()
                        + self.data[0][3].clone() * rhs.data[3][1].clone(),
                    self.data[0][0].clone() * rhs.data[0][2].clone()
                        + self.data[0][1].clone() * rhs.data[1][2].clone()
                        + self.data[0][2].clone() * rhs.data[2][2].clone()
                        + self.data[0][3].clone() * rhs.data[3][2].clone(),
                    self.data[0][0].clone() * rhs.data[0][3].clone()
                        + self.data[0][1].clone() * rhs.data[1][3].clone()
                        + self.data[0][2].clone() * rhs.data[2][3].clone()
                        + self.data[0][3].clone() * rhs.data[3][3].clone(),
                ],
                [
                    self.data[1][0].clone() * rhs.data[0][0].clone()
                        + self.data[1][1].clone() * rhs.data[1][0].clone()
                        + self.data[1][2].clone() * rhs.data[2][0].clone()
                        + self.data[1][3].clone() * rhs.data[3][0].clone(),
                    self.data[1][0].clone() * rhs.data[0][1].clone()
                        + self.data[1][1].clone() * rhs.data[1][1].clone()
                        + self.data[1][2].clone() * rhs.data[2][1].clone()
                        + self.data[1][3].clone() * rhs.data[3][1].clone(),
                    self.data[1][0].clone() * rhs.data[0][2].clone()
                        + self.data[1][1].clone() * rhs.data[1][2].clone()
                        + self.data[1][2].clone() * rhs.data[2][2].clone()
                        + self.data[1][3].clone() * rhs.data[3][2].clone(),
                    self.data[1][0].clone() * rhs.data[0][3].clone()
                        + self.data[1][1].clone() * rhs.data[1][3].clone()
                        + self.data[1][2].clone() * rhs.data[2][3].clone()
                        + self.data[1][3].clone() * rhs.data[3][3].clone(),
                ],
                [
                    self.data[2][0].clone() * rhs.data[0][0].clone()
                        + self.data[2][1].clone() * rhs.data[1][0].clone()
                        + self.data[2][2].clone() * rhs.data[2][0].clone()
                        + self.data[2][3].clone() * rhs.data[3][0].clone(),
                    self.data[2][0].clone() * rhs.data[0][1].clone()
                        + self.data[2][1].clone() * rhs.data[1][1].clone()
                        + self.data[2][2].clone() * rhs.data[2][1].clone()
                        + self.data[2][3].clone() * rhs.data[3][1].clone(),
                    self.data[2][0].clone() * rhs.data[0][2].clone()
                        + self.data[2][1].clone() * rhs.data[1][2].clone()
                        + self.data[2][2].clone() * rhs.data[2][2].clone()
                        + self.data[2][3].clone() * rhs.data[3][2].clone(),
                    self.data[2][0].clone() * rhs.data[0][3].clone()
                        + self.data[2][1].clone() * rhs.data[1][3].clone()
                        + self.data[2][2].clone() * rhs.data[2][3].clone()
                        + self.data[2][3].clone() * rhs.data[3][3].clone(),
                ],
                [
                    self.data[3][0].clone() * rhs.data[0][0].clone()
                        + self.data[3][1].clone() * rhs.data[1][0].clone()
                        + self.data[3][2].clone() * rhs.data[2][0].clone()
                        + self.data[3][3].clone() * rhs.data[3][0].clone(),
                    self.data[3][0].clone() * rhs.data[0][1].clone()
                        + self.data[3][1].clone() * rhs.data[1][1].clone()
                        + self.data[3][2].clone() * rhs.data[2][1].clone()
                        + self.data[3][3].clone() * rhs.data[3][1].clone(),
                    self.data[3][0].clone() * rhs.data[0][2].clone()
                        + self.data[3][1].clone() * rhs.data[1][2].clone()
                        + self.data[3][2].clone() * rhs.data[2][2].clone()
                        + self.data[3][3].clone() * rhs.data[3][2].clone(),
                    self.data[3][0].clone() * rhs.data[0][3].clone()
                        + self.data[3][1].clone() * rhs.data[1][3].clone()
                        + self.data[3][2].clone() * rhs.data[2][3].clone()
                        + self.data[3][3].clone() * rhs.data[3][3].clone(),
                ],
            ],
        }
    }
}

/// Pack of operators definition module
impl Mat4<f32>
{
    /// Determinant getting function
    /// * Returns determinant of this matrix
    pub fn determinant(&self) -> f32 {
        self.data[0][0]
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
    /// * `l`, `r` - left and right boundaries
    /// * `b`, `t` - bottom and top
    /// * `n`, `f` - near and far
    /// * Returns projection matrix
    pub fn projection_frustum(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4<f32> {
        Self {
            data: [
                [2.0 * n / (r - l), 0.0, 0.0, 0.0],
                [0.0, 2.0 * n / (t - b), 0.0, 0.0],
                [
                    (r + l) / (r - l),
                    (t + b) / (t - b),
                    -(f + n) / (f - n),
                    -1.0,
                ],
                [0.0, 0.0, -2.0 * n * f / (f - n), 0.0],
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

    pub fn transform_vector(&self, v: Vec3<f32>) -> Vec3<f32> {
        Vec3 {
            x: v.x * self.data[0][0] + v.y * self.data[1][0] + v.z * self.data[2][0],
            y: v.x * self.data[0][1] + v.y * self.data[1][1] + v.z * self.data[2][1],
            z: v.x * self.data[0][2] + v.y * self.data[1][2] + v.z * self.data[2][2],
        }
    } // fn transform_vector

    pub fn transform_point(&self, v: Vec3<f32>) -> Vec3<f32> {
        Vec3 {
            x: v.x * self.data[0][0]
                + v.y * self.data[1][0]
                + v.z * self.data[2][0]
                + self.data[3][0],
            y: v.x * self.data[0][1]
                + v.y * self.data[1][1]
                + v.z * self.data[2][1]
                + self.data[3][1],
            z: v.x * self.data[0][2]
                + v.y * self.data[1][2]
                + v.z * self.data[2][2]
                + self.data[3][2],
        }
    } // fn transform_point

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
