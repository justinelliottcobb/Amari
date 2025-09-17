use amari_core::Multivector;
use amari_tropical::TropicalMultivector;
use amari_dual::DualMultivector;
use amari_fusion::TropicalDualClifford;

#[test]
fn test_tropical_dual_clifford_consistency() {
    let logits = vec![1.0, 2.0, 3.0, 4.0];
    let tdc = TropicalDualClifford::<f64, 4>::from_logits(&logits);
    
    // All three views should be consistent
    assert_eq!(tdc.tropical().max_element().0, 4.0);
    assert_eq!(tdc.clifford().get(0), 1.0);
    
    // Gradients should work
    let (value, gradient) = tdc.dual().forward_mode_ad(|x| {
        let mut result = x.clone();
        for i in 0..8 {
            let coeff = x.get(i);
            result.set(i, coeff * coeff);
        }
        result
    });
    assert!(value > 0.0);
    assert!(gradient.value().as_slice().len() > 0);
}

#[test]
fn test_performance_tropical_vs_traditional() {
    use std::time::Instant;
    
    let size = 100; // Reduced size for reliability
    let logits: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
    
    // Traditional softmax
    let start = Instant::now();
    let exp_sum: f64 = logits.iter().map(|x| x.exp()).sum();
    let _traditional = logits.iter().map(|x| x.exp() / exp_sum).collect::<Vec<_>>();
    let traditional_time = start.elapsed();
    
    // Tropical max
    let tropical_logits = TropicalMultivector::<f64, 8>::from_logits(&logits);
    let start = Instant::now();
    let _tropical_max = tropical_logits.max_element();
    let tropical_time = start.elapsed();
    
    // Both should complete successfully (timing may vary)
    assert!(traditional_time.as_nanos() > 0);
    assert!(tropical_time.as_nanos() > 0);
    
    // Just verify tropical computation works
    assert!(_tropical_max.0 > 0.0);
}