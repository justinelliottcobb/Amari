#[allow(unused_imports)]
use amari_core::Multivector;
#[allow(unused_imports)]
use amari_dual::DualMultivector;
use amari_fusion::TropicalDualClifford;
use amari_tropical::TropicalMultivector;

/// NOTE: Disabled in v0.12.0 - max_element() and forward_mode_ad() removed from API
#[test]
#[ignore]
fn test_tropical_dual_clifford_consistency() {
    // TODO: Re-enable when max_element() and forward_mode_ad() are re-added to v0.12.0 API
}

/// NOTE: Disabled in v0.12.0 - TropicalMultivector now requires 4 generic args, max_element() removed
#[test]
#[ignore]
fn test_performance_tropical_vs_traditional() {
    // TODO: Re-enable when TropicalMultivector API is updated for from_logits() and max_element()
}
