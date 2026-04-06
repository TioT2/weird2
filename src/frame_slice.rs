//! 2D image slice common structure implementation module

use std::marker::PhantomData;

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
    data: *const T,

    /// Phantom data
    _phantom: PhantomData<&'t T>,
}

impl<'t, T> FrameSlice<'t, T> {
    /// Construct new from w, h, s parameters
    pub const fn new(width: usize, height: usize, stride: usize, data: &'t [T]) -> Self {
        assert!(height * stride <= data.len(), "Data array width must be equal to content width");
        assert!(width <= stride, "Slice width must be less or equal to frame stride");

        Self {
            width,
            height,
            stride,
            data: data.as_ptr(),
            _phantom: PhantomData,
        }
    }

    /// Construct new for raw structures
    pub const unsafe fn from_raw_parts(width: usize, height: usize, stride: usize, ptr: *const T) -> Self {
        assert!(width <= stride, "Slice width must be less or equal to frame stride");
        Self {
            width,
            height,
            stride,
            data: ptr,
            _phantom: PhantomData,
        }
    }

    /// Create empty frame slice
    pub const fn empty() -> Self {
        Self {
            width: 0,
            height: 0,
            stride: 0,
            data: std::ptr::dangling_mut(),
            _phantom: PhantomData,
        }
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
        self.data
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
            std::slice::from_raw_parts(self.data, self.width * self.stride)
        })
    }

    /// Get horizontal line by it's y coordinate
    pub fn getline(&self, y: usize) -> Option<&'t [T]> {
        (y < self.height).then(|| unsafe {
            std::slice::from_raw_parts(
                self.data.add(y * self.stride),
                self.width
            )
        })
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
    data: *mut T,

    /// `data` contents lifetime holder
    _phantom: PhantomData<&'t mut ()>,
}

impl<'t, T> FrameSliceMut<'t, T> {
    /// Construct new from w, h, s parameters
    pub const fn new(width: usize, height: usize, stride: usize, data: &'t mut [T]) -> Self {
        assert!(height * stride <= data.len(), "Data array width must be equal to content width");
        assert!(width <= stride, "Slice width must be less or equal to frame stride");

        Self {
            width,
            height,
            stride,
            data: data.as_mut_ptr(),
            _phantom: PhantomData,
        }
    }

    /// Construct new for raw structures
    pub const unsafe fn from_raw_parts(width: usize, height: usize, stride: usize, ptr: *mut T) -> Self {
        assert!(width <= stride, "Slice width must be less or equal to frame stride");
        Self {
            width,
            height,
            stride,
            data: ptr,
            _phantom: PhantomData,
        }
    }

    /// Create empty frame slice
    pub const fn empty() -> Self {
        Self {
            width: 0,
            height: 0,
            stride: 0,
            data: std::ptr::dangling_mut(),
            _phantom: PhantomData,
        }
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
        self.data
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
            std::slice::from_raw_parts_mut(self.data, self.stride * self.height)
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

    /// Get horizontal line by it's y coordinate
    pub fn getline<'l>(&'l mut self, y: usize) -> Option<&'l mut [T]> {
        (y < self.height).then(|| unsafe {
            std::slice::from_raw_parts_mut(
                self.data.add(y * self.stride),
                self.width
            )
        })
    }
}
