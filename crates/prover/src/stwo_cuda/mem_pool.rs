// GPU Memory Pool for CUDA Backend - Improved version
// DEPRECATED: This Rust-side memory pool is deprecated in favor of CUDA's built-in memory pool (cudaMemPool_t).
// The new implementation is entirely in CUDA code for better performance and integration.
// TODO: Remove this file once all dependent code has been migrated.

#[deprecated(note = "Use CUDA's built-in memory pool (cudaMemPool_t) instead. See cuda_mem_pool.cuh")]
use std::sync::{Mutex, OnceLock};
use std::ffi::c_void;
use std::collections::HashMap;

use crate::stwo_cuda::bindings;

#[derive(Debug)]
pub struct CudaMemPool {
    device_id: usize,
    // Use HashMap to store pools for different sizes
    // Key is the size, value is a vector of free pointers
    pool: OnceLock<Mutex<HashMap<usize, Vec<*const c_void>>>>,
    // Track total allocated memory
    total_allocated: OnceLock<Mutex<usize>>,
}

impl CudaMemPool {
    pub fn new(device_id: usize) -> Self {
        Self {
            device_id,
            pool: OnceLock::new(),
            total_allocated: OnceLock::new(),
        }
    }

    fn get_pool(&self) -> &Mutex<HashMap<usize, Vec<*const c_void>>> {
        self.pool.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn get_total_allocated(&self) -> &Mutex<usize> {
        self.total_allocated.get_or_init(|| Mutex::new(0))
    }

    // Dynamic pre-allocation based on log_size
    pub fn preallocate_for_log_size(&self, log_size: u32) {
        // Don't preallocate for large sizes to avoid memory issues
        if log_size > 20 {
            return;
        }
        
        let mut pool = self.get_pool().lock().unwrap();
        let mut total_allocated = self.get_total_allocated().lock().unwrap();
        
        // Calculate reasonable allocation sizes based on log_size
        let base_size = 1usize << (log_size.min(10) as usize);
        
        // Only preallocate small to medium blocks
        let sizes_to_preallocate = vec![
            (base_size, 2),         // Base size blocks
            (base_size * 2, 2),     // 2x base size blocks
            (base_size * 4, 1),     // 4x base size blocks
        ];
        
        for (size, count) in sizes_to_preallocate {
            let blocks = pool.entry(size).or_insert_with(Vec::new);
            
            for _ in 0..count {
                if blocks.len() < count {
                    let ptr = unsafe { bindings::cuda_malloc_uint32_t(size as u32) };
                    if !ptr.is_null() {
                        blocks.push(ptr as *const c_void);
                        *total_allocated += size * 4; // size is in u32, convert to bytes
                    }
                }
            }
        }
        
        crate::bench_println!(
            "Memory pool initialized for log_size={}, allocated {} MB",
            log_size,
            *total_allocated / 1024 / 1024
        );
    }

    pub fn allocate(&self, size: usize) -> *const u32 {
        let mut pool = self.get_pool().lock().unwrap();
        
        // Try to find exact size match first
        if let Some(blocks) = pool.get_mut(&size) {
            if let Some(ptr) = blocks.pop() {
                return ptr as *const u32;
            }
        }
        
        // For large allocations (>64MB), always allocate fresh to avoid memory waste
        if size > 16 * 1024 * 1024 {
            return unsafe { bindings::cuda_malloc_uint32_t(size as u32) };
        }
        
        // Try to find a slightly larger block (up to 25% larger)
        let max_acceptable_size = size + size / 4;
        for (&block_size, blocks) in pool.iter_mut() {
            if block_size >= size && block_size <= max_acceptable_size {
                if let Some(ptr) = blocks.pop() {
                    return ptr as *const u32;
                }
            }
        }
        
        // Fallback: allocate new memory
        let mut total_allocated = self.get_total_allocated().lock().unwrap();
        let ptr = unsafe { bindings::cuda_malloc_uint32_t(size as u32) };
        if !ptr.is_null() {
            *total_allocated += size * 4; // size is in u32, convert to bytes
        }
        ptr
    }

    pub fn deallocate(&self, ptr: *const u32, size: usize) {
        if ptr.is_null() {
            return;
        }
        
        let mut pool = self.get_pool().lock().unwrap();
        
        // For very large allocations, free immediately
        if size > 16 * 1024 * 1024 {
            unsafe { bindings::cuda_free_memory(ptr as *const c_void) };
            let mut total_allocated = self.get_total_allocated().lock().unwrap();
            *total_allocated = total_allocated.saturating_sub(size * 4);
            return;
        }
        
        // Add to appropriate size pool
        let blocks = pool.entry(size).or_insert_with(Vec::new);
        
        // Limit pool size based on block size
        let max_blocks = match size {
            s if s < 1024 => 32,           // Small blocks: keep more
            s if s < 1024 * 1024 => 16,    // Medium blocks
            _ => 8,                         // Large blocks: keep fewer
        };
        
        if blocks.len() < max_blocks {
            blocks.push(ptr as *const c_void);
        } else {
            // If pool is full, free the memory
            unsafe { bindings::cuda_free_memory(ptr as *const c_void) };
            let mut total_allocated = self.get_total_allocated().lock().unwrap();
            *total_allocated = total_allocated.saturating_sub(size * 4);
        }
    }

    pub fn allocate_zeroes(&self, size: usize) -> *const u32 {
        // For zeroed memory, we can potentially reuse pooled memory
        let ptr = self.allocate(size);
        if !ptr.is_null() {
            // Zero the memory using CUDA memset
            unsafe {
                let cuda_ptr = ptr as *mut c_void;
                let result = cudaMemset(cuda_ptr, 0, size * 4);
                if result != 0 {
                    // If memset fails, fall back to cuda_alloc_zeroes
                    self.deallocate(ptr, size);
                    return bindings::cuda_alloc_zeroes_uint32_t(size as u32);
                }
            }
        }
        ptr
    }

    pub fn clear(&self) {
        let mut pool = self.get_pool().lock().unwrap();
        let mut total_allocated = self.get_total_allocated().lock().unwrap();
        
        // Free all pooled memory
        for (size, blocks) in pool.iter() {
            for ptr in blocks {
                unsafe { bindings::cuda_free_memory(*ptr) };
                *total_allocated = total_allocated.saturating_sub(size * 4);
            }
        }
        pool.clear();
        
        crate::bench_println!(
            "Memory pool cleared, freed {} MB",
            *total_allocated / 1024 / 1024
        );
        *total_allocated = 0;
    }

    pub fn get_stats(&self) -> HashMap<usize, usize> {
        let pool = self.get_pool().lock().unwrap();
        let total_allocated = self.get_total_allocated().lock().unwrap();
        
        let mut stats = pool.iter().map(|(&size, blocks)| (size, blocks.len())).collect::<HashMap<_, _>>();
        // Add total allocated memory as a special entry
        stats.insert(usize::MAX, *total_allocated / 1024 / 1024); // Total in MB
        stats
    }
}

impl Drop for CudaMemPool {
    fn drop(&mut self) {
        self.clear();
    }
}

unsafe impl Send for CudaMemPool {}
unsafe impl Sync for CudaMemPool {}

// External CUDA function for memset
extern "C" {
    fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
}

// Global memory pool instance
static GLOBAL_MEMORY_POOL: OnceLock<CudaMemPool> = OnceLock::new();

pub fn get_global_memory_pool() -> &'static CudaMemPool {
    GLOBAL_MEMORY_POOL.get_or_init(|| {
        CudaMemPool::new(0) // Default to device 0
    })
}

pub fn init_memory_pool(device_id: usize) {
    let _ = GLOBAL_MEMORY_POOL.set(CudaMemPool::new(device_id));
}

pub fn init_memory_pool_with_preallocation(device_id: usize, log_size: u32) {
    let pool = CudaMemPool::new(device_id);
    pool.preallocate_for_log_size(log_size);
    let _ = GLOBAL_MEMORY_POOL.set(pool);
}

// Convenience functions for global pool
pub fn pool_allocate(size: usize) -> *const u32 {
    get_global_memory_pool().allocate(size)
}

pub fn pool_deallocate(ptr: *const u32, size: usize) {
    get_global_memory_pool().deallocate(ptr, size)
}

pub fn pool_allocate_zeroes(size: usize) -> *const u32 {
    get_global_memory_pool().allocate_zeroes(size)
}

pub fn pool_clear() {
    get_global_memory_pool().clear()
}

pub fn pool_preallocate_for_log_size(log_size: u32) {
    get_global_memory_pool().preallocate_for_log_size(log_size)
}

pub fn pool_get_stats() -> HashMap<usize, usize> {
    get_global_memory_pool().get_stats()
}
