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

    unsafe fn alloc_chunk(size: usize) -> *mut Chunk {
        let layout = std::alloc::Layout::from_size_align_unchecked(
            std::mem::size_of::<Chunk>() + size,
            std::mem::align_of::<Chunk>(),
        );
        let ptr = std::alloc::alloc(layout);

        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
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

    unsafe fn dealloc_chunk(chunk: *mut Chunk) {
        // std::alloc::dealloc(chunk as *mut u8, layout);
    }
}
