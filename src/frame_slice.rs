//! 2D image slice common structure implementation module

use std::{marker::PhantomData, ptr::NonNull};

/// Unsafe frame slice iterator
#[derive(Copy, Clone)]
struct UnsafeIter<'t, T> {
    /// Current pointer
    ptr: NonNull<T>,

    /// Range end
    end: NonNull<T>,

    /// Row width
    width: usize,

    /// Step
    step: usize,

    /// Phantom data
    _phantom: PhantomData<&'t T>,
}

impl<'t, T> UnsafeIter<'t, T> {
    /// Next pointer
    fn next_ptr(&mut self) -> Option<NonNull<T>> {
        if self.ptr < self.end {
            let ptr = self.ptr;
            self.ptr = unsafe { self.ptr.add(self.step) };
            Some(ptr)
        } else {
            None
        }
    }

    /// Next constant slice
    unsafe fn next(&mut self) -> Option<&'t [T]> {
        self.next_ptr().map(|ptr| unsafe {
            std::slice::from_raw_parts(
                ptr.as_ptr().cast_const(),
                self.width
            )
        })
    }

    /// Next mutable slice
    unsafe fn next_mut(&mut self) -> Option<&'t mut [T]> {
        self.next_ptr().map(|ptr| unsafe {
            std::slice::from_raw_parts_mut(
                ptr.as_ptr(),
                self.width
            )
        })
    }
}

/// Frame slice constant iterator
#[derive(Copy, Clone)]
pub struct Iter<'t, T>(UnsafeIter<'t, T>);

impl<'t, T> Iterator for Iter<'t, T> {
    type Item = &'t [T];

    fn next(&mut self) -> Option<&'t [T]> {
        unsafe { self.0.next() }
    }
}

/// Frame slice mutable iterator
pub struct IterMut<'t, T>(UnsafeIter<'t, T>);

impl<'t, T> Iterator for IterMut<'t, T> {
    type Item = &'t mut [T];

    fn next(&mut self) -> Option<&'t mut [T]> {
        unsafe { self.0.next_mut() }
    }
}

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

impl<'t, T> Default for FrameSlice<'t, T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<'t, T> FrameSlice<'t, T> {
    /// Construct new from w, h, s parameters
    pub const fn new(width: usize, height: usize, stride: usize, data: &'t [T]) -> Self {
        assert!(height * stride <= data.len(), "Data array width must be equal to content width");
        assert!(width <= stride, "Slice width must be less or equal to frame stride");

        unsafe { Self::from_raw_parts(width, height, stride, data.as_ptr()) }
    }

    /// Construct new for raw structures
    /// # Safety
    /// [`ptr`] must point to start of WxH frame with S stride (W rows of size S where first H elements are borrowed for 't by resulting [`FrameSlice`]).
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
    /// # Safety
    /// [`y`] must be less, than [`height`][FrameSlice::height]
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
    /// # Safety
    /// [`y`] must be less, than [`height`][FrameSlice::height] and
    /// [`x`] must be less, than [`width`][FrameSlice::width]
    pub unsafe fn get2_unchecked(&self, y: usize, x: usize) -> &'t T {
        unsafe { self.data.add(y * self.stride + x).as_ref() }
    }

    /// 2-dimensional access
    pub fn get2(&self, y: usize, x: usize) -> Option<&'t T> {
        (y < self.height && x < self.width).then(|| unsafe { self.get2_unchecked(y, x) })
    }

    /// Create frame slice iterator
    pub fn iter(&self) -> Iter<'t, T> {
        Iter(UnsafeIter {
            ptr: self.data,
            end: unsafe { self.data.add(self.stride * self.height) },
            step: self.stride,
            width: self.width,
            _phantom: PhantomData,
        })
    }
}

impl<'t, T> std::ops::Index<usize> for FrameSlice<'t, T> {
    type Output = [T];

    fn index(&self, index: usize) -> &'t Self::Output {
        self.get(index).unwrap()
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

impl<'t, T> Default for FrameSliceMut<'t, T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<'t, T> FrameSliceMut<'t, T> {
    /// Construct new from w, h, s parameters
    pub const fn new(width: usize, height: usize, stride: usize, data: &'t mut [T]) -> Self {
        assert!(height * stride <= data.len(), "Data array width must be equal to content width");
        unsafe { Self::from_raw_parts(width, height, stride, data.as_mut_ptr()) }
    }

    /// Construct new for raw structures
    /// # Safety
    /// [`ptr`] must point to start of WxH frame with S stride (W rows of size S where first H elements are mutably borrowed for 't by resulting [`FrameSlice`]).
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
    pub fn as_flat(&mut self) -> Option<&mut [T]> {
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
    /// # Safety
    /// [`y`] must be less, than [`height`][FrameSliceMut::height]
    pub unsafe fn get_unchecked_mut(&mut self, y: usize) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.add(y * self.stride).as_ptr(),
                self.width
            )
        }
    }

    /// Get mutable line reference
    pub fn get_mut(&mut self, y: usize) -> Option<&mut [T]> {
        (y < self.height).then(|| unsafe { self.get_unchecked_mut(y) })
    }

    /// Get constant line reference without check
    /// # Safety
    /// [`y`] must be less, than [`height`][FrameSliceMut::height]
    pub unsafe fn get_unchecked(&self, y: usize) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.add(y * self.stride).as_ptr(),
                self.width
            )
        }
    }

    /// Get constnat line reference
    pub fn get(&self, y: usize) -> Option<&[T]> {
        (y < self.height).then(|| unsafe { self.get_unchecked(y) })
    }


    /// 2-dimensional unchecked mutable get
    /// # Safety
    /// [`y`] must be less than [`height`][FrameSliceMut::height] and
    /// [`x`] must be less than [`width`][FrameSliceMut::width]
    pub unsafe fn get2_unchecked_mut(&mut self, y: usize, x: usize) -> &mut T {
        unsafe { self.data.add(y * self.stride + x).as_mut() }
    }

    /// 2-dimensional mutable get
    pub fn get2_mut(&mut self, y: usize, x: usize) -> Option<&mut T> {
        (y < self.height && x < self.width).then(|| unsafe { self.get2_unchecked_mut(y, x) })
    }

    /// 2-dimensional unchecked get
    /// # Safety
    /// [`y`] must be less than [`height`][FrameSliceMut::height] and
    /// [`x`] must be less than [`width`][FrameSliceMut::width]
    pub unsafe fn get2_unchecked(&self, y: usize, x: usize) -> &T {
        unsafe { self.data.add(y * self.stride + x).as_ref() }
    }

    /// 2-dimensional get
    pub fn get2(&self, y: usize, x: usize) -> Option<&T> {
        (y < self.height && x < self.width).then(|| unsafe { self.get2_unchecked(y, x) })
    }

    /// Create unsafe iterator
    unsafe fn iter_unsafe<'i>(&self) -> UnsafeIter<'i, T> {
        UnsafeIter {
            ptr: self.data,
            end: unsafe { self.data.add(self.stride * self.height) },
            step: self.stride,
            width: self.width,
            _phantom: PhantomData,
        }
    }

    /// Create frame slice iterator
    pub fn iter<'i>(&'i self) -> Iter<'i, T> {
        unsafe { Iter(self.iter_unsafe()) }
    }

    /// Create frame slice mutable iterator
    pub fn iter_mut<'i>(&'i mut self) -> IterMut<'i, T> {
        unsafe { IterMut(self.iter_unsafe()) }
    }
}

impl<'t, T> std::ops::Index<usize> for FrameSliceMut<'t, T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<'t, T> std::ops::IndexMut<usize> for FrameSliceMut<'t, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

// Implement conversion
impl<'t, T> From<FrameSliceMut<'t, T>> for FrameSlice<'t, T> {
    fn from(f: FrameSliceMut<'t, T>) -> FrameSlice<'t, T> {
        unsafe { FrameSlice::from_raw_parts(f.width, f.height, f.stride, f.data.as_ptr()) }
    }
}
