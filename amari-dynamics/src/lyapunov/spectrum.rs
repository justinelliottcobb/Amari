//! Lyapunov exponent spectrum
//!
//! This module provides data structures for representing and analyzing
//! the Lyapunov spectrum of a dynamical system.
//!
//! # Overview
//!
//! Lyapunov exponents measure the rates of exponential divergence or
//! convergence of nearby trajectories. The full spectrum characterizes:
//!
//! - **Chaos**: Positive largest exponent indicates chaotic dynamics
//! - **Dimension**: Kaplan-Yorke dimension from the spectrum
//! - **Entropy**: Kolmogorov-Sinai entropy from positive exponents
//!
//! # Ordering
//!
//! By convention, Lyapunov exponents are ordered from largest to smallest:
//! λ₁ ≥ λ₂ ≥ ... ≥ λₙ
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::lyapunov::{compute_lyapunov_spectrum, LyapunovSpectrum};
//!
//! let spectrum = compute_lyapunov_spectrum(&lorenz_system, &initial, 10000, 0.01)?;
//!
//! if spectrum.is_chaotic() {
//!     println!("Largest exponent: {:.4}", spectrum.largest());
//!     println!("Kaplan-Yorke dimension: {:.4}", spectrum.kaplan_yorke_dimension().unwrap());
//!     println!("Kolmogorov-Sinai entropy: {:.4}", spectrum.ks_entropy());
//! }
//! ```

use std::fmt;

/// Lyapunov spectrum of a dynamical system
///
/// Contains the full set of Lyapunov exponents ordered from largest to smallest,
/// along with computed properties like dimension and entropy.
#[derive(Debug, Clone)]
pub struct LyapunovSpectrum {
    /// Lyapunov exponents, ordered λ₁ ≥ λ₂ ≥ ... ≥ λₙ
    pub exponents: Vec<f64>,

    /// Standard errors for each exponent (from time averaging)
    pub errors: Option<Vec<f64>>,

    /// Convergence history: (time, exponents) pairs
    pub convergence_history: Option<Vec<(f64, Vec<f64>)>>,

    /// Total integration time used
    pub integration_time: f64,

    /// Number of renormalization steps
    pub num_renormalizations: usize,
}

impl LyapunovSpectrum {
    /// Create a new Lyapunov spectrum from exponents
    ///
    /// Automatically sorts exponents in descending order.
    pub fn new(
        mut exponents: Vec<f64>,
        integration_time: f64,
        num_renormalizations: usize,
    ) -> Self {
        // Sort in descending order
        exponents.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        Self {
            exponents,
            errors: None,
            convergence_history: None,
            integration_time,
            num_renormalizations,
        }
    }

    /// Create spectrum with error estimates
    pub fn with_errors(
        exponents: Vec<f64>,
        errors: Vec<f64>,
        integration_time: f64,
        num_renormalizations: usize,
    ) -> Self {
        let mut spectrum = Self::new(exponents, integration_time, num_renormalizations);

        // Sort errors to match exponent ordering
        let mut indexed_errors: Vec<(usize, f64)> = errors.into_iter().enumerate().collect();
        indexed_errors.sort_by(|a, b| {
            spectrum.exponents[a.0]
                .partial_cmp(&spectrum.exponents[b.0])
                .unwrap_or(std::cmp::Ordering::Equal)
                .reverse()
        });

        spectrum.errors = Some(indexed_errors.into_iter().map(|(_, e)| e).collect());
        spectrum
    }

    /// Get the dimension of the system
    pub fn dimension(&self) -> usize {
        self.exponents.len()
    }

    /// Get the largest Lyapunov exponent (λ₁)
    pub fn largest(&self) -> f64 {
        self.exponents.first().copied().unwrap_or(0.0)
    }

    /// Get the smallest Lyapunov exponent (λₙ)
    pub fn smallest(&self) -> f64 {
        self.exponents.last().copied().unwrap_or(0.0)
    }

    /// Check if the system is chaotic (λ₁ > 0)
    pub fn is_chaotic(&self) -> bool {
        self.largest() > 0.0
    }

    /// Check if the system is hyperchaotic (at least two positive exponents)
    pub fn is_hyperchaotic(&self) -> bool {
        self.positive_count() >= 2
    }

    /// Count positive Lyapunov exponents
    pub fn positive_count(&self) -> usize {
        self.exponents.iter().filter(|&&x| x > 0.0).count()
    }

    /// Count negative Lyapunov exponents
    pub fn negative_count(&self) -> usize {
        self.exponents.iter().filter(|&&x| x < 0.0).count()
    }

    /// Count zero Lyapunov exponents (within tolerance)
    pub fn zero_count(&self, tolerance: f64) -> usize {
        self.exponents
            .iter()
            .filter(|&&x| x.abs() < tolerance)
            .count()
    }

    /// Sum of all Lyapunov exponents
    ///
    /// For Hamiltonian systems, this should be zero.
    /// For dissipative systems, this is typically negative.
    pub fn sum(&self) -> f64 {
        self.exponents.iter().sum()
    }

    /// Sum of positive Lyapunov exponents
    ///
    /// This equals the Kolmogorov-Sinai entropy for certain systems
    /// (Pesin's identity).
    pub fn positive_sum(&self) -> f64 {
        self.exponents.iter().filter(|&&x| x > 0.0).sum()
    }

    /// Kolmogorov-Sinai (KS) entropy estimate
    ///
    /// By Pesin's identity, for smooth systems with an SRB measure,
    /// h_KS = Σ λᵢ for λᵢ > 0
    pub fn ks_entropy(&self) -> f64 {
        self.positive_sum()
    }

    /// Kaplan-Yorke (Lyapunov) dimension
    ///
    /// The Kaplan-Yorke dimension is defined as:
    /// D_KY = j + (λ₁ + λ₂ + ... + λⱼ) / |λⱼ₊₁|
    ///
    /// where j is the largest integer such that λ₁ + ... + λⱼ ≥ 0.
    ///
    /// Returns `None` if the dimension cannot be computed (e.g., all exponents positive).
    pub fn kaplan_yorke_dimension(&self) -> Option<f64> {
        if self.exponents.is_empty() {
            return None;
        }

        // Find j: largest index where cumulative sum is non-negative
        let mut cumsum = 0.0;
        let mut j = 0;

        for (i, &lambda) in self.exponents.iter().enumerate() {
            cumsum += lambda;
            if cumsum >= 0.0 {
                j = i + 1;
            }
        }

        if j == 0 {
            // All exponents negative - dimension is 0
            return Some(0.0);
        }

        if j >= self.exponents.len() {
            // All exponents sum to non-negative - can't compute
            return None;
        }

        // Compute cumulative sum up to j
        let sum_j: f64 = self.exponents.iter().take(j).sum();
        let lambda_j_plus_1 = self.exponents[j];

        if lambda_j_plus_1.abs() < 1e-15 {
            return None; // Avoid division by zero
        }

        Some(j as f64 + sum_j / lambda_j_plus_1.abs())
    }

    /// Check if the system appears Hamiltonian (sum ≈ 0)
    pub fn is_hamiltonian(&self, tolerance: f64) -> bool {
        self.sum().abs() < tolerance
    }

    /// Check if exponents come in ± pairs (symplectic structure)
    pub fn is_symplectic(&self, tolerance: f64) -> bool {
        let n = self.exponents.len();
        if n % 2 != 0 {
            return false;
        }

        // Check if λᵢ ≈ -λₙ₋ᵢ₊₁
        for i in 0..n / 2 {
            let sum = self.exponents[i] + self.exponents[n - 1 - i];
            if sum.abs() > tolerance {
                return false;
            }
        }

        true
    }

    /// Classify the attractor type based on spectrum
    pub fn classify_attractor(&self, tolerance: f64) -> LyapunovClassification {
        let pos = self.positive_count();
        let zero = self.zero_count(tolerance);
        let _neg = self.negative_count();

        match (pos, zero) {
            (0, 0) => LyapunovClassification::FixedPoint,
            (0, 1) => LyapunovClassification::LimitCycle,
            (0, 2) => LyapunovClassification::Torus2,
            (0, n) if n >= 3 => LyapunovClassification::TorusN(n),
            (1, _) => LyapunovClassification::StrangeAttractor,
            (n, _) if n >= 2 => LyapunovClassification::HyperchaosN(n),
            _ => LyapunovClassification::Unknown,
        }
    }

    /// Get convergence rate (how quickly exponents stabilized)
    ///
    /// Returns the relative change in exponents over the last portion of integration.
    pub fn convergence_rate(&self) -> Option<f64> {
        let history = self.convergence_history.as_ref()?;
        if history.len() < 2 {
            return None;
        }

        let last = &history[history.len() - 1].1;
        let prev = &history[history.len() - 2].1;

        let mut max_change = 0.0_f64;
        for (l, p) in last.iter().zip(prev.iter()) {
            let change = (l - p).abs() / (p.abs().max(1e-10));
            max_change = max_change.max(change);
        }

        Some(max_change)
    }
}

impl fmt::Display for LyapunovSpectrum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Lyapunov Spectrum (n={})", self.dimension())?;
        writeln!(f, "  Exponents: {:?}", self.exponents)?;

        if let Some(errors) = &self.errors {
            writeln!(f, "  Errors: {:?}", errors)?;
        }

        writeln!(f, "  Chaotic: {}", self.is_chaotic())?;

        if let Some(d_ky) = self.kaplan_yorke_dimension() {
            writeln!(f, "  Kaplan-Yorke dimension: {:.4}", d_ky)?;
        }

        writeln!(f, "  KS entropy: {:.4}", self.ks_entropy())?;
        writeln!(f, "  Sum: {:.4}", self.sum())?;

        Ok(())
    }
}

/// Classification of attractors based on Lyapunov spectrum
#[derive(Debug, Clone, PartialEq)]
pub enum LyapunovClassification {
    /// Fixed point: all exponents negative
    FixedPoint,

    /// Limit cycle: one zero exponent, rest negative
    LimitCycle,

    /// 2-torus (quasiperiodic): two zero exponents
    Torus2,

    /// n-torus: n zero exponents
    TorusN(usize),

    /// Strange attractor: one positive exponent
    StrangeAttractor,

    /// Hyperchaos with n positive exponents
    HyperchaosN(usize),

    /// Cannot classify
    Unknown,
}

impl LyapunovClassification {
    /// Check if classification indicates chaotic behavior
    pub fn is_chaotic(&self) -> bool {
        matches!(self, Self::StrangeAttractor | Self::HyperchaosN(_))
    }

    /// Check if classification indicates regular (non-chaotic) behavior
    pub fn is_regular(&self) -> bool {
        matches!(
            self,
            Self::FixedPoint | Self::LimitCycle | Self::Torus2 | Self::TorusN(_)
        )
    }
}

impl fmt::Display for LyapunovClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FixedPoint => write!(f, "Fixed Point"),
            Self::LimitCycle => write!(f, "Limit Cycle"),
            Self::Torus2 => write!(f, "2-Torus"),
            Self::TorusN(n) => write!(f, "{}-Torus", n),
            Self::StrangeAttractor => write!(f, "Strange Attractor"),
            Self::HyperchaosN(n) => write!(f, "Hyperchaos ({}+)", n),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Configuration for Lyapunov computation
#[derive(Debug, Clone)]
pub struct LyapunovConfig {
    /// Integration time per renormalization
    pub renorm_time: f64,

    /// Number of renormalization steps
    pub num_renormalizations: usize,

    /// Initial transient time to skip
    pub transient_time: f64,

    /// Time step for integration
    pub dt: f64,

    /// Initial perturbation magnitude
    pub initial_perturbation: f64,

    /// Threshold for considering exponent zero
    pub zero_tolerance: f64,

    /// Whether to compute convergence history
    pub track_convergence: bool,

    /// How often to record convergence (every N renormalizations)
    pub convergence_interval: usize,
}

impl Default for LyapunovConfig {
    fn default() -> Self {
        Self {
            renorm_time: 1.0,
            num_renormalizations: 1000,
            transient_time: 100.0,
            dt: 0.01,
            initial_perturbation: 1e-8,
            zero_tolerance: 1e-4,
            track_convergence: true,
            convergence_interval: 10,
        }
    }
}

impl LyapunovConfig {
    /// Configuration for fast but rough estimates
    pub fn fast() -> Self {
        Self {
            renorm_time: 0.5,
            num_renormalizations: 100,
            transient_time: 50.0,
            track_convergence: false,
            ..Default::default()
        }
    }

    /// Configuration for accurate computation
    pub fn accurate() -> Self {
        Self {
            renorm_time: 2.0,
            num_renormalizations: 5000,
            transient_time: 500.0,
            convergence_interval: 50,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectrum_ordering() {
        let exponents = vec![-2.0, 0.9, 0.0, -14.5];
        let spectrum = LyapunovSpectrum::new(exponents, 1000.0, 100);

        assert_eq!(spectrum.exponents, vec![0.9, 0.0, -2.0, -14.5]);
        assert_eq!(spectrum.largest(), 0.9);
        assert_eq!(spectrum.smallest(), -14.5);
    }

    #[test]
    fn test_chaotic_detection() {
        let chaotic = LyapunovSpectrum::new(vec![0.9, 0.0, -14.5], 1000.0, 100);
        assert!(chaotic.is_chaotic());
        assert!(!chaotic.is_hyperchaotic());

        let hyperchaotic = LyapunovSpectrum::new(vec![0.9, 0.5, 0.0, -15.0], 1000.0, 100);
        assert!(hyperchaotic.is_hyperchaotic());
        assert_eq!(hyperchaotic.positive_count(), 2);

        let regular = LyapunovSpectrum::new(vec![0.0, -1.0, -2.0], 1000.0, 100);
        assert!(!regular.is_chaotic());
    }

    #[test]
    fn test_kaplan_yorke_dimension() {
        // Lorenz-like: λ = (0.9, 0, -14.5)
        let lorenz = LyapunovSpectrum::new(vec![0.9, 0.0, -14.5], 1000.0, 100);
        let d_ky = lorenz.kaplan_yorke_dimension().unwrap();

        // D_KY = 2 + (0.9 + 0) / 14.5 ≈ 2.062
        assert!((d_ky - 2.062).abs() < 0.01);
    }

    #[test]
    fn test_ks_entropy() {
        let spectrum = LyapunovSpectrum::new(vec![0.9, 0.3, 0.0, -14.5], 1000.0, 100);
        assert!((spectrum.ks_entropy() - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_symplectic_check() {
        // Symplectic: exponents come in ± pairs
        let symplectic = LyapunovSpectrum::new(vec![0.5, -0.5], 1000.0, 100);
        assert!(symplectic.is_symplectic(1e-10));

        let non_symplectic = LyapunovSpectrum::new(vec![0.5, -0.3], 1000.0, 100);
        assert!(!non_symplectic.is_symplectic(1e-10));
    }

    #[test]
    fn test_classification() {
        let fixed_point = LyapunovSpectrum::new(vec![-1.0, -2.0], 1000.0, 100);
        assert_eq!(
            fixed_point.classify_attractor(1e-4),
            LyapunovClassification::FixedPoint
        );

        let limit_cycle = LyapunovSpectrum::new(vec![0.0, -1.0], 1000.0, 100);
        assert_eq!(
            limit_cycle.classify_attractor(1e-4),
            LyapunovClassification::LimitCycle
        );

        let strange = LyapunovSpectrum::new(vec![0.9, 0.0, -14.5], 1000.0, 100);
        assert_eq!(
            strange.classify_attractor(1e-4),
            LyapunovClassification::StrangeAttractor
        );
    }

    #[test]
    fn test_sum() {
        // Dissipative system: negative sum
        let dissipative = LyapunovSpectrum::new(vec![0.9, 0.0, -14.5], 1000.0, 100);
        assert!(dissipative.sum() < 0.0);

        // Hamiltonian: sum ≈ 0
        let hamiltonian = LyapunovSpectrum::new(vec![0.5, -0.5], 1000.0, 100);
        assert!(hamiltonian.is_hamiltonian(1e-10));
    }

    #[test]
    fn test_config_presets() {
        let fast = LyapunovConfig::fast();
        assert_eq!(fast.num_renormalizations, 100);
        assert!(!fast.track_convergence);

        let accurate = LyapunovConfig::accurate();
        assert_eq!(accurate.num_renormalizations, 5000);
    }
}
