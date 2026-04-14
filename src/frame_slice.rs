//! 2D image slice common structure implementation module

use std::{marker::PhantomData, ptr::NonNull};

/// 2D image immutable slice
#[derive(Copy, Clone)]
pub struct FrameSlice<'t, T> {
    /// Frame slice width
    width: usize,

    /// Frame slice height
    height: usize,

    /// Frame stride
    stride: usize,

    /// Slice data
    data: NonNull<T>,

    /// Phantom data
    _phantom: PhantomData<&'t T>,
}

impl<'t, T> FrameSlice<'t, T> {
    /// Construct new from w, h, s parameters
    pub const fn new(width: usize, height: usize, stride: usize, data: &'t [T]) -> Self {
        assert!(height * stride <= data.len(), "Data array width must be equal to content width");
        assert!(width <= stride, "Slice width must be less or equal to frame stride");

        unsafe { Self::from_raw_parts(width, height, stride, data.as_ptr()) }
    }

    /// Construct new for raw structures
    pub const unsafe fn from_raw_parts(width: usize, height: usize, stride: usize, ptr: *const T) -> Self {
        assert!(width <= stride, "Slice width must be less or equal to frame stride");
        Self {
            width,
            height,
            stride,
            data: NonNull::new(ptr.cast_mut()).unwrap(),
            _phantom: PhantomData,
        }
    }

    /// Create empty frame slice
    pub const fn empty() -> Self {
        unsafe { Self::from_raw_parts(0, 0, 0, std::ptr::dangling()) }
    }

    /// Get frame width
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get frame height
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get frame stride (in size_of::<T>())
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Get data mutable pointer
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Split framebuffer by vertical line at `x` to disjoint subsets.
    /// `x` is truncated to buffer width.
    pub fn split_vertical(self, x: usize) -> (Self, Self) {
        let x = usize::min(x, self.width);

        (
            Self {
                width: x,
                data: self.data,
                ..self
            },
            Self {
                width: self.width - x,
                data: unsafe { self.data.add(x) },
                ..self
            }
        )
    }

    /// Split framebuffer by horizontal line at `y` to disjoint subsets.
    /// `y` is truncated to buffer height.
    pub fn split_horizontal(self, y: usize) -> (Self, Self) {
        let y = usize::min(y, self.height);

        (
            Self {
                height: y,
                data: self.data,
                ..self
            },
            Self {
                height: self.height - y,
                data: unsafe { self.data.add(y * self.stride) },
                ..self
            }
        )
    }

    /// Try to interpret self as a flat slice
    pub fn as_flat(&self) -> Option<&'t [T]> {
        (self.width == self.stride).then(|| unsafe {
            std::slice::from_raw_parts(self.data.as_ptr(), self.width * self.stride)
        })
    }

    /// Unchecked get function
    pub unsafe fn get_unchecked(&self, y: usize) -> &'t [T] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.add(y * self.stride).as_ptr(),
                self.width
            )
        }
    }

    /// Get horizontal line by it's Y coordinate
    pub fn get(&self, y: usize) -> Option<&'t [T]> {
        (y < self.height).then(|| unsafe { self.get_unchecked(y) })
    }

    /// 2-dimensional unchecked access
    pub unsafe fn get2_unchecked(&self, y: usize, x: usize) -> &'t T {
        unsafe { self.data.add(y * self.stride + x).as_ref() }
    }

    /// 2-dimensional access
    pub unsafe fn get2(&self, y: usize, x: usize) -> Option<&'t T> {
        (y < self.height && x < self.width).then(|| unsafe { self.get2_unchecked(y, x) })
    }
}

/// 2D image slicing structure that allows safe mutable access to disjoint image subsets
pub struct FrameSliceMut<'t, T> {
    /// Width of region allowed to write
    width: usize,

    /// Height of region allowed to write
    height: usize,

    /// Region stride (e.g. number of pixels to jump to next line, **must be** greater than width)
    stride: usize,

    /// Pointer to top-left part of frame
    data: NonNull<T>,

    /// `data` contents lifetime holder
    _phantom: PhantomData<&'t mut ()>,
}

impl<'t, T> FrameSliceMut<'t, T> {
    /// Construct new from w, h, s parameters
    pub const fn new(width: usize, height: usize, stride: usize, data: &'t mut [T]) -> Self {
        assert!(height * stride <= data.len(), "Data array width must be equal to content width");
        unsafe { Self::from_raw_parts(width, height, stride, data.as_mut_ptr()) }
    }

    /// Construct new for raw structures
    pub const unsafe fn from_raw_parts(width: usize, height: usize, stride: usize, ptr: *mut T) -> Self {
        assert!(width <= stride, "Slice width must be less or equal to frame stride");
        Self {
            width,
            height,
            stride,
            data: NonNull::new(ptr).unwrap(),
            _phantom: PhantomData,
        }
    }

    /// Create empty frame slice
    pub const fn empty() -> Self {
        unsafe { Self::from_raw_parts(0, 0, 0, std::ptr::dangling_mut()) }
    }

    /// Get frame width
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get frame height
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get frame stride (in size_of::<T>())
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Get data mutable pointer
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.data.as_ptr()
    }

    /// Reborrow frame slice contents from some shorter lifetime
    pub fn reborrow<'r>(&'r mut self) -> FrameSliceMut<'r, T> {
        FrameSliceMut::<'r> {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: self.data,
            _phantom: PhantomData,
        }
    }

    /// Try to interpret self as a flat mutable slice
    pub fn as_flat<'f>(&'f mut self) -> Option<&'f mut [T]> {
        (self.width == self.stride).then(|| unsafe {
            std::slice::from_raw_parts_mut(self.data.as_ptr(), self.stride * self.height)
        })
    }

    /// Split framebuffer by vertical line at `x` to disjoint subsets.
    /// `x` is truncated to buffer width.
    pub fn split_vertical(self, x: usize) -> (Self, Self) {
        let x = usize::min(x, self.width);

        (
            Self {
                width: x,
                data: self.data,
                ..self
            },
            Self {
                width: self.width - x,
                data: unsafe { self.data.add(x) },
                ..self
            }
        )
    }

    /// Split framebuffer by horizontal line at `y` to disjoint subsets.
    /// `y` is truncated to buffer height.
    pub fn split_horizontal(self, y: usize) -> (Self, Self) {
        let y = usize::min(y, self.height);

        (
            Self {
                height: y,
                data: self.data,
                ..self
            },
            Self {
                height: self.height - y,
                data: unsafe { self.data.add(y * self.stride) },
                ..self
            }
        )
    }

    /// Get mutable line reference without check
    pub unsafe fn get_unchecked_mut<'l>(&'l mut self, y: usize) -> &'l mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.add(y * self.stride).as_ptr(),
                self.width
            )
        }
    }

    /// Get mutable line reference
    pub fn get_mut<'l>(&'l mut self, y: usize) -> Option<&'l mut [T]> {
        (y < self.height).then(|| unsafe { self.get_unchecked_mut(y) })
    }

    /// Get constant line reference without check
    pub unsafe fn get_unchecked<'l>(&'l self, y: usize) -> &'l [T] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.add(y * self.stride).as_ptr(),
                self.width
            )
        }
    }

    /// Get constnat line reference
    pub fn get<'l>(&'l self, y: usize) -> Option<&'l [T]> {
        (y < self.height).then(|| unsafe { self.get_unchecked(y) })
    }


    /// 2-dimensional unchecked mutable get
    pub unsafe fn get2_unchecked_mut<'l>(&'l mut self, y: usize, x: usize) -> &'l mut T {
        unsafe { self.data.add(y * self.stride + x).as_mut() }
    }

    /// 2-dimensional mutable get
    pub fn get2_mut<'l>(&'l mut self, y: usize, x: usize) -> Option<&'l mut T> {
        (y < self.height && x < self.width).then(|| unsafe { self.get2_unchecked_mut(y, x) })
    }

    /// 2-dimensional unchecked get
    pub unsafe fn get2_unchecked<'l>(&'l self, y: usize, x: usize) -> &'l T {
        unsafe { self.data.add(y * self.stride + x).as_ref() }
    }

    /// 2-dimensional get
    pub fn get2<'l>(&'l self, y: usize, x: usize) -> Option<&'l T> {
        (y < self.height && x < self.width).then(|| unsafe { self.get2_unchecked(y, x) })
    }
}

// Implement conversion
impl<'t, T> From<FrameSliceMut<'t, T>> for FrameSlice<'t, T> {
    fn from(f: FrameSliceMut<'t, T>) -> FrameSlice<'t, T> {
        unsafe { FrameSlice::from_raw_parts(f.width, f.height, f.stride, f.data.as_ptr()) }
    }
}
