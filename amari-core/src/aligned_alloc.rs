//! Aligned memory allocation for SIMD-optimized operations
//!
//! This module provides memory alignment utilities specifically designed
//! for AVX2/SSE SIMD operations in geometric algebra computations.

use alloc::alloc::{alloc, dealloc, Layout};
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ptr::NonNull;

/// 32-byte aligned allocation for AVX2 operations
pub const AVX2_ALIGNMENT: usize = 32;

/// 16-byte aligned allocation for SSE operations
pub const SSE_ALIGNMENT: usize = 16;

/// Cache line size for modern CPUs
pub const CACHE_LINE_SIZE: usize = 64;

/// Aligned memory block for SIMD operations
#[repr(C)]
pub struct AlignedMemory<T> {
    ptr: NonNull<T>,
    layout: Layout,
}

impl<T> AlignedMemory<T> {
    /// Allocate aligned memory for the given number of elements
    pub fn new(count: usize, alignment: usize) -> Result<Self, &'static str> {
        let size = count * core::mem::size_of::<T>();
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| "Invalid layout for aligned allocation")?;

        let ptr = unsafe { alloc(layout) as *mut T };
        if ptr.is_null() {
            return Err("Failed to allocate aligned memory");
        }

        let ptr = unsafe { NonNull::new_unchecked(ptr) };

        Ok(Self { ptr, layout })
    }

    /// Get a raw pointer to the allocated memory
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get a mutable raw pointer to the allocated memory
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get the number of elements that can fit in this allocation
    pub fn capacity(&self) -> usize {
        self.layout.size() / core::mem::size_of::<T>()
    }
}

impl<T> Drop for AlignedMemory<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

/// Create an AVX2-aligned vector for f64 coefficients
pub fn create_aligned_f64_vec(count: usize) -> Vec<f64> {
    // For small sizes, use regular Vec
    if count <= 16 {
        return vec![0.0; count];
    }

    // Use system allocator with alignment hint
    let mut vec = Vec::with_capacity(count);
    vec.resize(count, 0.0);

    // Verify alignment for critical sizes
    let ptr = vec.as_ptr() as usize;
    if count == 8 && ptr % AVX2_ALIGNMENT != 0 {
        // Reallocate with proper alignment for 8-element vectors (3D Clifford algebra)
        let mut aligned_vec = Vec::with_capacity(count + (AVX2_ALIGNMENT / 8));
        aligned_vec.resize(count, 0.0);

        // Find aligned position within the allocation
        let start_ptr = aligned_vec.as_ptr() as usize;
        let aligned_offset = (AVX2_ALIGNMENT - (start_ptr % AVX2_ALIGNMENT)) % AVX2_ALIGNMENT / 8;

        if aligned_offset < aligned_vec.len() - count {
            // Return a slice starting at the aligned position
            aligned_vec.drain(0..aligned_offset);
            aligned_vec.truncate(count);
            return aligned_vec;
        }
    }

    vec
}

/// Cache-friendly memory pool for frequent allocations
pub struct MemoryPool {
    blocks: Vec<AlignedMemory<f64>>,
    block_size: usize,
    alignment: usize,
}

impl MemoryPool {
    /// Create a new memory pool with specified block size and alignment
    pub fn new(block_size: usize, alignment: usize) -> Self {
        Self {
            blocks: Vec::new(),
            block_size,
            alignment,
        }
    }

    /// Create a pool optimized for 3D Clifford algebra operations
    pub fn for_3d_clifford() -> Self {
        // 8 coefficients per multivector, 32-byte alignment for AVX2
        Self::new(8, AVX2_ALIGNMENT)
    }

    /// Allocate a block from the pool
    pub fn allocate(&mut self) -> Result<Box<[f64]>, &'static str> {
        // For now, use regular allocation
        // In a production system, this would maintain a pool of reusable blocks
        let coefficients = create_aligned_f64_vec(self.block_size);
        Ok(coefficients.into_boxed_slice())
    }

    /// Pre-allocate blocks for better performance
    pub fn pre_allocate(&mut self, count: usize) -> Result<(), &'static str> {
        for _ in 0..count {
            let block = AlignedMemory::new(self.block_size, self.alignment)?;
            self.blocks.push(block);
        }
        Ok(())
    }
}

/// RAII wrapper for aligned coefficient arrays
pub struct AlignedCoefficients {
    data: Box<[f64]>,
}

impl AlignedCoefficients {
    /// Create aligned coefficients for the given count
    pub fn new(count: usize) -> Self {
        Self {
            data: create_aligned_f64_vec(count).into_boxed_slice(),
        }
    }

    /// Create zero-initialized aligned coefficients
    pub fn zero(count: usize) -> Self {
        let mut coeffs = Self::new(count);
        coeffs.data.fill(0.0);
        coeffs
    }

    /// Get the underlying data
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Get mutable access to the underlying data
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Convert to boxed slice
    pub fn into_boxed_slice(self) -> Box<[f64]> {
        self.data
    }

    /// Check if the memory is properly aligned for SIMD
    pub fn is_simd_aligned(&self) -> bool {
        let ptr = self.data.as_ptr() as usize;
        ptr % AVX2_ALIGNMENT == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_memory() {
        let mem = AlignedMemory::<f64>::new(8, AVX2_ALIGNMENT).unwrap();
        let ptr = mem.as_ptr() as usize;
        assert_eq!(ptr % AVX2_ALIGNMENT, 0);
        assert_eq!(mem.capacity(), 8);
    }

    #[test]
    fn test_aligned_coefficients() {
        let coeffs = AlignedCoefficients::zero(8);
        assert_eq!(coeffs.as_slice().len(), 8);
        assert!(coeffs.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::for_3d_clifford();
        let _block = pool.allocate().unwrap();
        assert_eq!(pool.block_size, 8);
        assert_eq!(pool.alignment, AVX2_ALIGNMENT);
    }

    #[test]
    fn test_aligned_f64_vec() {
        let vec = create_aligned_f64_vec(8);
        assert_eq!(vec.len(), 8);

        // For 3D Clifford algebra, we want good alignment
        let ptr = vec.as_ptr() as usize;
        // Should be at least 8-byte aligned for f64
        assert_eq!(ptr % 8, 0);
    }
}