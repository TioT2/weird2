/// 

pub use std::arch::x86_64 as arch;

#[derive(Copy, Clone)]
pub struct FMat4(
    arch::__m128,
    arch::__m128,
    arch::__m128,
    arch::__m128,
);

impl FMat4 {
    pub fn zero() -> Self {
        unsafe {
            Self(
                arch::_mm_setzero_ps(),
                arch::_mm_setzero_ps(),
                arch::_mm_setzero_ps(),
                arch::_mm_setzero_ps(),
            )
        }
    }

    pub fn identity() -> Self {
        unsafe {
            Self(
                arch::_mm_set_ps(0.0, 0.0, 0.0, 1.0),
                arch::_mm_set_ps(0.0, 0.0, 1.0, 0.0),
                arch::_mm_set_ps(0.0, 1.0, 0.0, 0.0),
                arch::_mm_set_ps(1.0, 0.0, 0.0, 0.0),
            )
        }
    }
}

impl std::ops::Add<FMat4> for FMat4 {
    type Output = FMat4;

    fn add(self, rhs: FMat4) -> Self::Output {
        unsafe {
            Self(
                arch::_mm_add_ps(self.0, rhs.0),
                arch::_mm_add_ps(self.1, rhs.1),
                arch::_mm_add_ps(self.2, rhs.2),
                arch::_mm_add_ps(self.3, rhs.3),
            )
        }
    }
}

impl std::ops::AddAssign<FMat4> for FMat4 {
    fn add_assign(&mut self, rhs: FMat4) {
        unsafe {
            self.0 = arch::_mm_add_ps(self.0, rhs.0);
            self.1 = arch::_mm_add_ps(self.1, rhs.1);
            self.2 = arch::_mm_add_ps(self.2, rhs.2);
            self.3 = arch::_mm_add_ps(self.3, rhs.3);
        }
    }
}