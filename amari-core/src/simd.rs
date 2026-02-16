//! SIMD optimizations for geometric algebra operations
//!
//! This module provides vectorized implementations of critical geometric algebra
//! operations using CPU SIMD instruction sets (AVX2, SSE) for maximum performance.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::*;

use crate::Multivector;

/// SIMD-optimized geometric product for 3D Euclidean algebra (most common case)
#[cfg(target_feature = "avx2")]
#[inline(always)]
pub fn geometric_product_3d_avx2(
    lhs: &Multivector<3, 0, 0>,
    rhs: &Multivector<3, 0, 0>,
) -> Multivector<3, 0, 0> {
    unsafe {
        let _result = Multivector::<3, 0, 0>::zero();

        // Load coefficients into AVX2 registers (256-bit, 4 doubles each)
        let lhs_low = _mm256_loadu_pd(lhs.as_slice().as_ptr());
        let lhs_high = _mm256_loadu_pd(lhs.as_slice().as_ptr().add(4));
        let rhs_low = _mm256_loadu_pd(rhs.as_slice().as_ptr());
        let rhs_high = _mm256_loadu_pd(rhs.as_slice().as_ptr().add(4));

        // Result accumulator
        let mut result_low = _mm256_setzero_pd();
        let mut result_high = _mm256_setzero_pd();

        // Unrolled computation for 8x8 geometric product
        // This manually implements the geometric product using
        // precomputed multiplication patterns for 3D Euclidean space

        // Scalar * all components
        let scalar_lhs = _mm256_set1_pd(lhs.get(0));
        result_low = _mm256_fmadd_pd(scalar_lhs, rhs_low, result_low);
        result_high = _mm256_fmadd_pd(scalar_lhs, rhs_high, result_high);

        // e1 products
        let e1_lhs = _mm256_set1_pd(lhs.get(1));
        let e1_pattern_low = _mm256_set_pd(-rhs.get(3), rhs.get(2), rhs.get(0), rhs.get(1));
        let e1_pattern_high = _mm256_set_pd(-rhs.get(7), -rhs.get(6), rhs.get(5), rhs.get(4));
        result_low = _mm256_fmadd_pd(e1_lhs, e1_pattern_low, result_low);
        result_high = _mm256_fmadd_pd(e1_lhs, e1_pattern_high, result_high);

        // e2 products
        let e2_lhs = _mm256_set1_pd(lhs.get(2));
        let e2_pattern_low = _mm256_set_pd(rhs.get(1), rhs.get(0), -rhs.get(3), rhs.get(2));
        let e2_pattern_high = _mm256_set_pd(rhs.get(6), -rhs.get(7), rhs.get(4), -rhs.get(5));
        result_low = _mm256_fmadd_pd(e2_lhs, e2_pattern_low, result_low);
        result_high = _mm256_fmadd_pd(e2_lhs, e2_pattern_high, result_high);

        // e3 products
        let e3_lhs = _mm256_set1_pd(lhs.get(4));
        let e3_pattern_low = _mm256_set_pd(-rhs.get(2), rhs.get(1), rhs.get(0), rhs.get(4));
        let e3_pattern_high = _mm256_set_pd(-rhs.get(5), rhs.get(4), -rhs.get(7), rhs.get(6));
        result_low = _mm256_fmadd_pd(e3_lhs, e3_pattern_low, result_low);
        result_high = _mm256_fmadd_pd(e3_lhs, e3_pattern_high, result_high);

        // e12 products
        let e12_lhs = _mm256_set1_pd(lhs.get(3));
        let e12_pattern_low = _mm256_set_pd(rhs.get(0), -rhs.get(4), rhs.get(1), -rhs.get(2));
        let e12_pattern_high = _mm256_set_pd(rhs.get(4), rhs.get(7), -rhs.get(6), rhs.get(5));
        result_low = _mm256_fmadd_pd(e12_lhs, e12_pattern_low, result_low);
        result_high = _mm256_fmadd_pd(e12_lhs, e12_pattern_high, result_high);

        // e13 products
        let e13_lhs = _mm256_set1_pd(lhs.get(5));
        let e13_pattern_low = _mm256_set_pd(rhs.get(4), rhs.get(0), -rhs.get(2), rhs.get(1));
        let e13_pattern_high = _mm256_set_pd(-rhs.get(7), rhs.get(6), rhs.get(4), -rhs.get(5));
        result_low = _mm256_fmadd_pd(e13_lhs, e13_pattern_low, result_low);
        result_high = _mm256_fmadd_pd(e13_lhs, e13_pattern_high, result_high);

        // e23 products
        let e23_lhs = _mm256_set1_pd(lhs.get(6));
        let e23_pattern_low = _mm256_set_pd(-rhs.get(1), rhs.get(0), rhs.get(4), rhs.get(2));
        let e23_pattern_high = _mm256_set_pd(rhs.get(5), -rhs.get(4), rhs.get(7), rhs.get(6));
        result_low = _mm256_fmadd_pd(e23_lhs, e23_pattern_low, result_low);
        result_high = _mm256_fmadd_pd(e23_lhs, e23_pattern_high, result_high);

        // e123 products
        let e123_lhs = _mm256_set1_pd(lhs.get(7));
        let e123_pattern_low = _mm256_set_pd(rhs.get(1), rhs.get(2), rhs.get(4), -rhs.get(0));
        let e123_pattern_high = _mm256_set_pd(-rhs.get(5), -rhs.get(6), -rhs.get(7), rhs.get(4));
        result_low = _mm256_fmadd_pd(e123_lhs, e123_pattern_low, result_low);
        result_high = _mm256_fmadd_pd(e123_lhs, e123_pattern_high, result_high);

        // Store results back to memory
        let mut coeffs = [0.0; 8];
        _mm256_storeu_pd(coeffs.as_mut_ptr(), result_low);
        _mm256_storeu_pd(coeffs.as_mut_ptr().add(4), result_high);

        Multivector::from_coefficients(coeffs.to_vec())
    }
}

/// Optimized batch geometric product for processing multiple multivector pairs
#[cfg(target_feature = "avx2")]
pub fn batch_geometric_product_avx2(
    lhs_batch: &[f64],
    rhs_batch: &[f64],
    result_batch: &mut [f64],
) {
    const COEFFS_PER_MV: usize = 8;
    let num_pairs = lhs_batch.len() / COEFFS_PER_MV;

    for i in 0..num_pairs {
        let lhs_offset = i * COEFFS_PER_MV;
        let rhs_offset = i * COEFFS_PER_MV;
        let result_offset = i * COEFFS_PER_MV;

        // Create temporary multivectors from slices
        let lhs_coeffs = lhs_batch[lhs_offset..lhs_offset + COEFFS_PER_MV].to_vec();
        let rhs_coeffs = rhs_batch[rhs_offset..rhs_offset + COEFFS_PER_MV].to_vec();

        let lhs_mv = Multivector::<3, 0, 0>::from_coefficients(lhs_coeffs);
        let rhs_mv = Multivector::<3, 0, 0>::from_coefficients(rhs_coeffs);

        // Compute product using SIMD optimization
        let result_mv = geometric_product_3d_avx2(&lhs_mv, &rhs_mv);

        // Copy result back to batch array
        result_batch[result_offset..result_offset + COEFFS_PER_MV]
            .copy_from_slice(result_mv.as_slice());
    }
}

/// CPU feature detection for optimal code path selection
/// In std environments, uses runtime detection; in no_std, uses compile-time detection
pub fn select_geometric_product_impl(
) -> fn(&Multivector<3, 0, 0>, &Multivector<3, 0, 0>) -> Multivector<3, 0, 0> {
    #[cfg(all(not(feature = "std"), target_feature = "avx2"))]
    {
        return geometric_product_3d_avx2;
    }

    #[cfg(all(feature = "std", target_feature = "avx2"))]
    {
        if is_x86_feature_detected!("avx2") {
            return geometric_product_3d_avx2;
        }
    }

    // Fallback to scalar implementation
    |lhs, rhs| lhs.geometric_product(rhs)
}

/// Memory-aligned buffer for SIMD operations
#[repr(C, align(32))]
pub struct AlignedBuffer<const N: usize> {
    pub data: [f64; N],
}

impl<const N: usize> AlignedBuffer<N> {
    pub fn new() -> Self {
        Self { data: [0.0; N] }
    }

    pub fn as_ptr(&self) -> *const f64 {
        self.data.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        self.data.as_mut_ptr()
    }
}

impl<const N: usize> Default for AlignedBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Multivector;
    use approx::assert_relative_eq;

    type Cl3 = Multivector<3, 0, 0>;

    #[test]
    #[cfg(target_feature = "avx2")]
    fn test_simd_geometric_product_correctness() {
        let e1 = Cl3::basis_vector(0);
        let e2 = Cl3::basis_vector(1);

        // Test against scalar implementation
        let scalar_result = e1.geometric_product(&e2);
        let simd_result = geometric_product_3d_avx2(&e1, &e2);

        for i in 0..8 {
            assert_relative_eq!(scalar_result.get(i), simd_result.get(i), epsilon = 1e-14);
        }
    }

    #[test]
    fn test_aligned_buffer() {
        let mut buffer = AlignedBuffer::<8>::new();
        buffer.data[0] = 1.0;
        assert_eq!(buffer.data[0], 1.0);

        // Verify alignment
        let ptr = buffer.as_ptr() as usize;
        assert_eq!(ptr % 32, 0);
    }

    #[test]
    fn test_runtime_feature_detection() {
        let impl_fn = select_geometric_product_impl();

        let e1 = Cl3::basis_vector(0);
        let e2 = Cl3::basis_vector(1);
        let result = impl_fn(&e1, &e2);

        // Should match scalar implementation
        let expected = e1.geometric_product(&e2);
        for i in 0..8 {
            assert_relative_eq!(result.get(i), expected.get(i), epsilon = 1e-14);
        }
    }
}
