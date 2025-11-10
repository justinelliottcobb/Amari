//! Probability estimation utilities

/// Estimate probability from sample data
pub fn estimate_probability(successes: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    successes as f64 / total as f64
}

/// Compute confidence interval for probability estimate
///
/// Returns (lower_bound, upper_bound) for given confidence level
pub fn confidence_interval(successes: usize, total: usize, confidence: f64) -> (f64, f64) {
    if total == 0 {
        return (0.0, 1.0);
    }

    let p = estimate_probability(successes, total);
    let n = total as f64;

    // Wilson score interval (better for edge cases than normal approximation)
    let z = confidence_z_score(confidence);
    let z_sq = z * z;

    let denominator = 1.0 + z_sq / n;
    let center = (p + z_sq / (2.0 * n)) / denominator;
    let margin = (z / denominator) * ((p * (1.0 - p) / n) + (z_sq / (4.0 * n * n))).sqrt();

    let lower = (center - margin).max(0.0);
    let upper = (center + margin).min(1.0);

    (lower, upper)
}

/// Compute required sample size for desired precision
pub fn sample_size_for_precision(precision: f64, confidence: f64) -> usize {
    let z = confidence_z_score(confidence);
    let n = (z * z) / (4.0 * precision * precision);
    n.ceil() as usize
}

/// Get z-score for confidence level
fn confidence_z_score(confidence: f64) -> f64 {
    // Approximate z-scores for common confidence levels
    match confidence {
        c if (c - 0.90).abs() < 0.01 => 1.645,
        c if (c - 0.95).abs() < 0.01 => 1.960,
        c if (c - 0.99).abs() < 0.01 => 2.576,
        _ => 1.960, // Default to 95%
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_probability() {
        assert_eq!(estimate_probability(5, 10), 0.5);
        assert_eq!(estimate_probability(0, 10), 0.0);
        assert_eq!(estimate_probability(10, 10), 1.0);
    }

    #[test]
    fn test_confidence_interval() {
        let (lower, upper) = confidence_interval(50, 100, 0.95);
        assert!(lower < 0.5);
        assert!(upper > 0.5);
        assert!(lower >= 0.0);
        assert!(upper <= 1.0);
    }

    #[test]
    fn test_sample_size() {
        let n = sample_size_for_precision(0.05, 0.95);
        assert!(n > 0);
        assert!(n < 1000); // Should be reasonable
    }
}
