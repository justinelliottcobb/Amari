//! Concentration inequalities and probability bounds

/// Hoeffding's inequality bound
///
/// For n independent samples, probability that sample mean deviates
/// from true mean by more than epsilon is bounded by 2*exp(-2*n*epsilon^2)
pub fn hoeffding_bound(n: usize, epsilon: f64) -> f64 {
    2.0 * (-2.0 * n as f64 * epsilon * epsilon).exp()
}

/// Chernoff bound for tail probabilities
///
/// Upper bound on probability that sum of independent random variables
/// deviates from expected value
pub fn chernoff_bound(_n: usize, mu: f64, delta: f64) -> f64 {
    if delta <= 0.0 {
        return 1.0;
    }

    // Simplified Chernoff bound for 0/1 random variables
    let exp_term = -(mu * delta * delta) / 3.0;
    exp_term.exp()
}

/// Compute confidence level from sample statistics
///
/// Given samples and observed deviation, compute confidence that
/// true probability is within epsilon of observed probability
pub fn compute_confidence(samples: usize, epsilon: f64) -> f64 {
    let bound = hoeffding_bound(samples, epsilon);
    (1.0 - bound).max(0.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hoeffding_bound() {
        let bound = hoeffding_bound(100, 0.1);
        assert!(bound > 0.0);
        assert!(bound < 1.0);

        // More samples should give tighter bound
        let bound2 = hoeffding_bound(1000, 0.1);
        assert!(bound2 < bound);
    }

    #[test]
    fn test_chernoff_bound() {
        let bound = chernoff_bound(100, 50.0, 0.1);
        assert!(bound > 0.0);
        assert!(bound <= 1.0);
    }

    #[test]
    fn test_compute_confidence() {
        let conf = compute_confidence(1000, 0.05);
        assert!(conf > 0.9);
        assert!(conf <= 1.0);
    }
}
