use std::mem::MaybeUninit;

///! Arena allocator

struct Chunk {
    size: usize,
    memory: *mut u8,
    next: *mut Chunk,
}

pub struct Arena {
    first_free_chunk: *mut Chunk,
    first_used_chunk: *mut Chunk,
    current_ptr: *mut u8,
    current_end: *mut u8,
}

impl Arena {
    pub fn new() -> Self {
        Self {
            first_free_chunk: std::ptr::null_mut(),
            first_used_chunk: std::ptr::null_mut(),
            current_ptr: std::ptr::null_mut(),
            current_end: std::ptr::null_mut(),
        }
    }

    /// Build chunk allocation layout
    const unsafe fn get_chunk_layout(size: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align_unchecked(
            std::mem::size_of::<Chunk>() + size,
            std::mem::align_of::<Chunk>(),
        )
    }

    /// Allocate chunk
    unsafe fn alloc_chunk(size: usize) -> *mut Chunk {
        let ptr = std::alloc::alloc(Self::get_chunk_layout(size));

        if ptr.is_null() {
            std::alloc::handle_alloc_error(Self::get_chunk_layout(size));
        }

        let chunk = ptr as *mut MaybeUninit<Chunk>;
        let memory = ptr.add(std::mem::size_of::<Chunk>());

        let chunk_ref = chunk.as_mut().unwrap();

        chunk_ref.write(Chunk {
            memory,
            next: std::ptr::null_mut(),
            size,
        });

        return chunk_ref.assume_init_mut();
    }

    /// Deallocate chunk
    unsafe fn dealloc_chunk(chunk: *mut Chunk) {
        let chunk_ptr = chunk.as_mut().unwrap_unchecked();
        let size = chunk_ptr.size;

        std::alloc::dealloc(chunk as *mut u8, Self::get_chunk_layout(size));
    }

    /// Allocate
    pub unsafe fn alloc(&mut self, layout: std::alloc::Layout) {
        let mut start_ptr = self.current_ptr.add(self.current_ptr.align_offset(layout.align()));
        let mut end_ptr = self.current_ptr.add(layout.size());

        // Allocate new chunk if current's finished.
        if end_ptr > self.current_end {
            // Select chunk size

            // Allocate new chunk
            let new_chunk = Self::alloc_chunk(65536);
        }
    }
}
