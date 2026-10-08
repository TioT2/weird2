//! Bump allocator (for frame buffers)

use std::{alloc::Layout, cell::Cell, marker::PhantomData, mem::MaybeUninit, ptr::NonNull};

/// Single allocation chunk
#[repr(align(16))]
struct Chunk {
    /// Next chunk pointer
    next: Option<NonNull<Chunk>>,

    /// Current pointer
    ptr: NonNull<u8>,

    /// Chunk allocation layout
    layout: Layout,
}

/// Initial chunk length
const DEFAULT_INIT_LENGTH: usize = 32768;

/// Chunk alignment
const CHUNK_ALIGN: usize = std::mem::align_of::<Chunk>();

impl Chunk {
    /// Allocate new chunk of given content length
    unsafe fn alloc_new(len: usize, next: Option<NonNull<Chunk>>) -> NonNull<Self> {
        // Align content length by chunk type alignment
        let len = (len + CHUNK_ALIGN - 1) / CHUNK_ALIGN * CHUNK_ALIGN;
        let layout = Layout::from_size_align(
            len + std::mem::size_of::<Self>(),
            CHUNK_ALIGN
        ).unwrap();

        let Some(ptr) = NonNull::new(unsafe { std::alloc::alloc(layout) }) else {
            std::alloc::handle_alloc_error(layout);
        };

        let v = unsafe { ptr.add(len).cast::<MaybeUninit<Self>>().as_mut() };
        v.write(Chunk { next, ptr, layout });

        unsafe { ptr.add(len).cast() }
    }

    /// Allocate memory by layout
    unsafe fn alloc(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        let begin = unsafe { self.ptr.add(self.ptr.align_offset(layout.align())) };
        let end = unsafe { begin.add(layout.size()) };

        if end > NonNull::from_ref(self).cast::<u8>() {
            return None;
        }

        self.ptr = end;

        Some(begin)
    }
}

// I think bump allocator really does not need a flush mechanism.
/// Bump allocator
pub struct Bump {
    /// Top chunk pointer
    chunk: Cell<NonNull<Chunk>>,
}

impl Bump {
    /// Create new bump
    pub fn new(length: Option<usize>) -> Self {
        let length = length.unwrap_or(DEFAULT_INIT_LENGTH);
        let chk = unsafe { Chunk::alloc_new(length, None) };

        Self {
            chunk: Cell::new(chk),
        }
    }

    /// Allocate new memory portion from layout
    pub unsafe fn alloc_ptr(&self, layout: Layout) -> NonNull<u8> {
        // Yes, I know that this tactic is kind of... interesting.
        loop {
            let chk = unsafe { self.chunk.get().as_mut() };

            if let Some(ptr) = unsafe { chk.alloc(layout) } {
                return ptr;
            }

            // Calculate length of the next chunk
            self.chunk.set(unsafe {
                Chunk::alloc_new(
                    (chk.layout.size() - std::mem::size_of::<Chunk>()) * 2,
                    Some(self.chunk.get()))
            });
        }
    }

    /// Allocate uninit value of T
    pub fn alloc_uninit<'t, T>(&'t self) -> &'t mut MaybeUninit<T> {
        unsafe {
            self.alloc_ptr(Layout::new::<T>())
                .cast::<MaybeUninit<T>>()
                .as_mut()
        }
    }

    /// Allocate a given value
    pub fn alloc<'t, T>(&'t self, value: T) -> &'t mut T {
        self.alloc_uninit().write(value)
    }

    /// Allocate boxed value
    pub fn alloc_boxed<'t, T>(&'t self, value: T) -> BBox<'t, T> {
        BBox {
            ptr: NonNull::from_mut(self.alloc(value)),
            phantom: PhantomData::default(),
        }
    }

    /// Allocate slice of uninitialized values
    pub fn alloc_uninit_slice<'t, T>(&'t self, len: usize) -> &'t mut [MaybeUninit<T>] {
        unsafe {
            let ptr = self.alloc_ptr(Layout::array::<T>(len).unwrap()).cast::<MaybeUninit<T>>();
            std::slice::from_raw_parts_mut(ptr.as_ptr(), len)
        }
    }

    /// Allocate slice with some kind of pointer
    pub fn alloc_slice<'t, T>(&'t self, len: usize, value: T) -> &'t mut [T]
    where T: Copy
    {
        let uslice = self.alloc_uninit_slice(len);
        uslice.fill(MaybeUninit::new(value));
        unsafe { uslice.assume_init_mut() }
    }

    /// Allocate a boxed slice
    pub fn alloc_boxed_slice<'t, T>(&'t self, len: usize, value: T) -> BBox<'t, [T]>
    where T: Copy
    {
        BBox {
            ptr: NonNull::from_mut(self.alloc_slice(len, value)),
            phantom: PhantomData::default(),
        }
    }
}

impl Drop for Bump {
    fn drop(&mut self) {
        let mut chunk_ptr_opt = Some(self.chunk.get());

        while let Some(chunk_ptr) = chunk_ptr_opt {
            let chunk = unsafe { chunk_ptr.as_ref() };
            chunk_ptr_opt = chunk.next;
            let layout = chunk.layout;

            // Drop chunk structure
            unsafe { std::ptr::drop_in_place(chunk_ptr.as_ptr()) };

            // Calculate pointer to allocation
            let allocation = unsafe { chunk_ptr.add(1).byte_sub(layout.size()).cast::<u8>() };

            // Deallocate allocation
            unsafe { std::alloc::dealloc(allocation.as_ptr(), layout) };
        }
    }
}

/// Bump-allocated value box
pub struct BBox<'t, T: ?Sized> {
    ptr: NonNull<T>,
    phantom: PhantomData<&'t T>,
}

impl<'t, T> std::ops::Deref for BBox<'t, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }
}

impl<'t, T> std::ops::DerefMut for BBox<'t, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}

impl<'t, T: ?Sized> Drop for BBox<'t, T> {
    fn drop(&mut self) {
        unsafe { std::ptr::drop_in_place(self.ptr.as_ptr()) };
    }
}
