//! Bump allocator (for frame buffers)

use std::{alloc::Layout, cell::RefCell, marker::PhantomData, mem::MaybeUninit, ptr::NonNull};

/// Chunk alignment
const CHUNK_ALIGN: usize = std::mem::align_of::<ChunkData>();

/// Allocatino chunk contents
#[repr(align(16))]
struct ChunkData {
    /// Next chunk pointer
    next: Option<NonNull<ChunkData>>,

    /// Current pointer
    ptr: NonNull<u8>,

    /// Chunk allocation layout
    layout: Layout,
}

/// Fallible bump allocator
pub struct Chunk {
    /// Underlying data pointer
    data: NonNull<ChunkData>,
}

impl Chunk {
    /// Take underlying chunk data
    unsafe fn take_data(self) -> NonNull<ChunkData> {
        let result = self.data;
        std::mem::forget(self);
        result
    }

    /// Align length by
    fn align_length(length: usize) -> usize {
        (length + CHUNK_ALIGN - 1) / CHUNK_ALIGN * CHUNK_ALIGN
    }

    /// Create new chunk
    pub fn new(len: usize, next: Option<Chunk>) -> Self {
        // Align content length by chunk type alignment
        let len = Self::align_length(len);
        let layout = Layout::from_size_align(
            len + std::mem::size_of::<ChunkData>(),
            CHUNK_ALIGN
        ).unwrap();

        let Some(ptr) = NonNull::new(unsafe { std::alloc::alloc(layout) }) else {
            std::alloc::handle_alloc_error(layout);
        };

        let v = unsafe { ptr.add(len).cast::<MaybeUninit<ChunkData>>().as_mut() };
        v.write(ChunkData {
            next: next.map(|v| unsafe { v.take_data() }),
            ptr,
            layout
        });

        Self { data: unsafe { ptr.add(len).cast() } }
    }

    /// Allocate a pointer range by layout
    pub unsafe fn alloc_ptr(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        let data = unsafe { self.data.as_mut() };

        let begin = unsafe { data.ptr.add(data.ptr.align_offset(layout.align())) };
        let end = unsafe { begin.add(layout.size()) };

        if end > self.data.cast() {
            return None;
        }

        data.ptr = end;

        Some(begin)
    }

    /// Get amount of useful chunk data
    pub fn length(&self) -> usize {
        unsafe { self.data.as_ref() }.layout.size() - std::mem::size_of::<ChunkData>()
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        let mut chunk_ptr_opt = Some(self.data);

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

/// Initial chunk length
const DEFAULT_INIT_LENGTH: usize = 32768;

// I think bump allocator really does not need a flush mechanism.
/// Bump allocator
pub struct Bump {
    /// Top chunk pointer
    chunk: RefCell<Option<Chunk>>,
}

impl Bump {
    /// Create new bump
    pub fn new() -> Self {
        Self { chunk: RefCell::new(None) }
    }

    /// Allocate new memory portion from layout
    pub unsafe fn alloc_ptr(&self, layout: Layout) -> NonNull<u8> {
        // Yes, I know that this tactic is kind of... interesting.
        loop {
            let mut chunk_opt = self.chunk.borrow_mut();

            // Create a chunk then not present
            let Some(chunk) = chunk_opt.as_mut() else {
                *chunk_opt = Some(Chunk::new(DEFAULT_INIT_LENGTH, None));
                continue;
            };

            // Try to allocate
            if let Some(data) = unsafe { chunk.alloc_ptr(layout) } {
                return data;
            }

            // Chain a larger chunk on failure
            *chunk_opt = chunk_opt
                .take()
                .map(|prev| Chunk::new(
                    prev.length() * 2,
                    Some(prev)
                ));
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

/// Bump-allocated value box
pub struct BBox<'t, T: ?Sized> {
    /// Underlying value pointer
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
