//! Cayley table generation and caching for efficient geometric product computation
//!
//! The Cayley table precomputes the multiplication rules for all basis blade pairs,
//! encoding both the resulting blade index and sign based on the algebra's signature.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicPtr, Ordering};

/// Entry in the Cayley table: (sign, result_index)
type CayleyEntry = (f64, usize);

/// Cayley table for Clifford algebra Cl(P,Q,R)
pub struct CayleyTable<const P: usize, const Q: usize, const R: usize> {
    table: Vec<CayleyEntry>,
    dim: usize,
}

impl<const P: usize, const Q: usize, const R: usize> CayleyTable<P, Q, R> {
    const DIM: usize = P + Q + R;
    const BASIS_COUNT: usize = 1 << Self::DIM;
    
    /// Get or create the singleton Cayley table
    pub fn get() -> &'static Self {
        static mut TABLE: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());
        
        unsafe {
            let ptr = TABLE.load(Ordering::Acquire);
            if ptr.is_null() {
                let table = Box::leak(Box::new(Self::generate()));
                TABLE.store(table as *mut _ as *mut (), Ordering::Release);
                table
            } else {
                &*(ptr as *const Self)
            }
        }
    }
    
    /// Generate the Cayley table for this algebra
    fn generate() -> Self {
        let mut table = Vec::with_capacity(Self::BASIS_COUNT * Self::BASIS_COUNT);
        
        for i in 0..Self::BASIS_COUNT {
            for j in 0..Self::BASIS_COUNT {
                let (sign, index) = Self::compute_product(i, j);
                table.push((sign, index));
            }
        }
        
        Self {
            table,
            dim: Self::DIM,
        }
    }
    
    /// Get the product of two basis blades by their indices
    #[inline(always)]
    pub fn get_product(&self, i: usize, j: usize) -> CayleyEntry {
        self.table[i * Self::BASIS_COUNT + j]
    }
    
    /// Compute the product of two basis blades
    fn compute_product(blade_a: usize, blade_b: usize) -> CayleyEntry {
        // Result blade is XOR of the two input blades
        let result_blade = blade_a ^ blade_b;
        
        // Compute sign from reordering and metric signature
        let mut sign = 1.0;
        
        // Count swaps needed to reorder basis vectors
        sign *= Self::compute_reorder_sign(blade_a, blade_b);
        
        // Apply metric signature for squared basis vectors
        let common = blade_a & blade_b;
        if common != 0 {
            sign *= Self::compute_metric_sign(common);
        }
        
        (sign, result_blade)
    }
    
    /// Compute sign from reordering basis vectors
    fn compute_reorder_sign(blade_a: usize, blade_b: usize) -> f64 {
        let mut swaps = 0;
        let mut b = blade_b;
        
        while b != 0 {
            // Get rightmost set bit in b
            let lowest_b = b & (!b + 1);
            b ^= lowest_b;
            
            // Count how many basis vectors in a are to the right of this one
            let mask = lowest_b - 1;
            let count = (blade_a & mask).count_ones();
            swaps += count;
        }
        
        if swaps % 2 == 0 { 1.0 } else { -1.0 }
    }
    
    /// Compute sign from metric signature for squared basis vectors
    fn compute_metric_sign(common_bits: usize) -> f64 {
        let mut sign = 1.0;
        
        for i in 0..Self::DIM {
            if (common_bits >> i) & 1 == 1 {
                // Determine signature of basis vector i
                if i < P {
                    // Positive signature: e_i^2 = +1
                    sign *= 1.0;
                } else if i < P + Q {
                    // Negative signature: e_i^2 = -1
                    sign *= -1.0;
                } else {
                    // Null signature: e_i^2 = 0
                    return 0.0;
                }
            }
        }
        
        sign
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_euclidean_3d() {
        type Table3D = CayleyTable<3, 0, 0>;
        let table = Table3D::get();
        
        // e1 * e1 = 1 (blade 1 * blade 1 = blade 0 with sign +1)
        assert_eq!(table.get_product(1, 1), (1.0, 0));
        
        // e1 * e2 = e12 (blade 1 * blade 2 = blade 3 with sign +1)
        assert_eq!(table.get_product(1, 2), (1.0, 3));
        
        // e2 * e1 = -e12 (blade 2 * blade 1 = blade 3 with sign -1)
        assert_eq!(table.get_product(2, 1), (-1.0, 3));
        
        // e12 * e12 = -1 (blade 3 * blade 3 = blade 0 with sign -1)
        assert_eq!(table.get_product(3, 3), (-1.0, 0));
    }
    
    #[test]
    fn test_minkowski_spacetime() {
        type TableST = CayleyTable<1, 3, 0>; // Spacetime signature (+,-,-,-)
        let table = TableST::get();
        
        // e0 * e0 = +1 (timelike, blade 1 * blade 1)
        assert_eq!(table.get_product(1, 1), (1.0, 0));
        
        // e1 * e1 = -1 (spacelike, blade 2 * blade 2)
        assert_eq!(table.get_product(2, 2), (-1.0, 0));
    }
    
    #[test]
    fn test_null_vectors() {
        type TableNull = CayleyTable<1, 1, 1>; // One null vector
        let table = TableNull::get();
        
        // e2 * e2 = 0 (null vector, blade 4 * blade 4)
        assert_eq!(table.get_product(4, 4), (0.0, 0));
    }
}