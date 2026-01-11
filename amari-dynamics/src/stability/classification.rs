//! Stability classification for fixed points
//!
//! This module provides the classification of fixed points based on their
//! local stability properties, determined by eigenvalue analysis.
//!
//! # Classification Scheme
//!
//! For continuous-time systems (ODEs), stability is determined by the
//! eigenvalues of the Jacobian matrix at the fixed point:
//!
//! - **Stable**: All eigenvalues have negative real parts
//! - **Unstable**: At least one eigenvalue has positive real part
//! - **Saddle**: Both positive and negative real parts exist
//! - **Center**: All eigenvalues are purely imaginary

use core::fmt;

/// Stability classification of a fixed point
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StabilityType {
    /// Asymptotically stable: all eigenvalues have Re(λ) < 0
    ///
    /// Trajectories converge to the fixed point exponentially.
    AsymptoticallyStable,

    /// Stable node: all eigenvalues are real and negative
    ///
    /// Trajectories approach along straight lines.
    StableNode,

    /// Stable focus (spiral): complex eigenvalues with Re(λ) < 0
    ///
    /// Trajectories spiral inward toward the fixed point.
    StableFocus,

    /// Unstable: at least one eigenvalue has Re(λ) > 0
    ///
    /// Trajectories diverge from the fixed point.
    Unstable,

    /// Unstable node: all eigenvalues are real and positive
    ///
    /// Trajectories diverge along straight lines.
    UnstableNode,

    /// Unstable focus (spiral): complex eigenvalues with Re(λ) > 0
    ///
    /// Trajectories spiral outward from the fixed point.
    UnstableFocus,

    /// Saddle point: eigenvalues with both positive and negative real parts
    ///
    /// Trajectories approach along stable manifold, diverge along unstable.
    Saddle,

    /// Center: all eigenvalues are purely imaginary
    ///
    /// Trajectories form closed orbits around the fixed point.
    /// Note: This is structurally unstable in nonlinear systems.
    Center,

    /// Degenerate: at least one eigenvalue is zero
    ///
    /// Requires higher-order analysis to determine stability.
    Degenerate,

    /// Lyapunov stable but not asymptotically stable
    ///
    /// Trajectories remain bounded but don't converge.
    LyapunovStable,

    /// Could not determine stability
    Unknown,
}

impl StabilityType {
    /// Check if the fixed point is stable (asymptotically or Lyapunov)
    pub fn is_stable(&self) -> bool {
        matches!(
            self,
            StabilityType::AsymptoticallyStable
                | StabilityType::StableNode
                | StabilityType::StableFocus
                | StabilityType::LyapunovStable
        )
    }

    /// Check if the fixed point is asymptotically stable
    pub fn is_asymptotically_stable(&self) -> bool {
        matches!(
            self,
            StabilityType::AsymptoticallyStable
                | StabilityType::StableNode
                | StabilityType::StableFocus
        )
    }

    /// Check if the fixed point is unstable
    pub fn is_unstable(&self) -> bool {
        matches!(
            self,
            StabilityType::Unstable
                | StabilityType::UnstableNode
                | StabilityType::UnstableFocus
                | StabilityType::Saddle
        )
    }

    /// Check if the fixed point is a saddle
    pub fn is_saddle(&self) -> bool {
        matches!(self, StabilityType::Saddle)
    }

    /// Check if the fixed point is a center
    pub fn is_center(&self) -> bool {
        matches!(self, StabilityType::Center)
    }

    /// Check if trajectories spiral (complex eigenvalues)
    pub fn is_spiral(&self) -> bool {
        matches!(
            self,
            StabilityType::StableFocus | StabilityType::UnstableFocus | StabilityType::Center
        )
    }

    /// Check if classification is degenerate or unknown
    pub fn is_degenerate(&self) -> bool {
        matches!(self, StabilityType::Degenerate | StabilityType::Unknown)
    }

    /// Get a short description of the stability type
    pub fn description(&self) -> &'static str {
        match self {
            StabilityType::AsymptoticallyStable => "asymptotically stable",
            StabilityType::StableNode => "stable node",
            StabilityType::StableFocus => "stable focus (spiral sink)",
            StabilityType::Unstable => "unstable",
            StabilityType::UnstableNode => "unstable node",
            StabilityType::UnstableFocus => "unstable focus (spiral source)",
            StabilityType::Saddle => "saddle point",
            StabilityType::Center => "center",
            StabilityType::Degenerate => "degenerate (zero eigenvalue)",
            StabilityType::LyapunovStable => "Lyapunov stable (not asymptotically)",
            StabilityType::Unknown => "unknown",
        }
    }
}

impl fmt::Display for StabilityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Classify stability based on eigenvalues
///
/// # Arguments
///
/// * `eigenvalues` - Vector of (real_part, imaginary_part) pairs
/// * `tolerance` - Threshold for considering a value as zero
///
/// # Returns
///
/// The stability classification
pub fn classify_from_eigenvalues(eigenvalues: &[(f64, f64)], tolerance: f64) -> StabilityType {
    if eigenvalues.is_empty() {
        return StabilityType::Unknown;
    }

    let mut has_positive_real = false;
    let mut has_negative_real = false;
    let mut has_zero_real = false;
    let mut all_real = true;
    let mut has_nonzero_imaginary = false;

    for &(re, im) in eigenvalues {
        // Check real part
        if re > tolerance {
            has_positive_real = true;
        } else if re < -tolerance {
            has_negative_real = true;
        } else {
            has_zero_real = true;
        }

        // Check imaginary part
        if im.abs() > tolerance {
            all_real = false;
            has_nonzero_imaginary = true;
        }
    }

    // Degenerate case: zero eigenvalue
    if has_zero_real && !has_positive_real && !has_negative_real {
        if has_nonzero_imaginary {
            return StabilityType::Center;
        }
        return StabilityType::Degenerate;
    }

    // Saddle: both positive and negative real parts
    if has_positive_real && has_negative_real {
        return StabilityType::Saddle;
    }

    // All negative real parts: stable
    if has_negative_real && !has_positive_real && !has_zero_real {
        if all_real {
            return StabilityType::StableNode;
        } else {
            return StabilityType::StableFocus;
        }
    }

    // All positive real parts: unstable
    if has_positive_real && !has_negative_real && !has_zero_real {
        if all_real {
            return StabilityType::UnstableNode;
        } else {
            return StabilityType::UnstableFocus;
        }
    }

    // Center: zero real parts with nonzero imaginary
    if has_zero_real && has_nonzero_imaginary && !has_positive_real && !has_negative_real {
        return StabilityType::Center;
    }

    // Mixed with zero eigenvalues
    if has_zero_real {
        return StabilityType::Degenerate;
    }

    StabilityType::Unknown
}

/// Discrete-time stability classification
///
/// For discrete maps x_{n+1} = f(x_n), stability is determined by
/// the magnitude of eigenvalues:
///
/// - Stable: all |λ| < 1
/// - Unstable: at least one |λ| > 1
/// - Saddle: some |λ| < 1 and some |λ| > 1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiscreteStabilityType {
    /// Asymptotically stable: all |λ| < 1
    AsymptoticallyStable,

    /// Stable node: all eigenvalues real, |λ| < 1
    StableNode,

    /// Stable focus: complex eigenvalues with |λ| < 1
    StableFocus,

    /// Unstable: at least one |λ| > 1
    Unstable,

    /// Unstable node: all eigenvalues real, |λ| > 1
    UnstableNode,

    /// Unstable focus: complex eigenvalues with |λ| > 1
    UnstableFocus,

    /// Saddle: some |λ| < 1 and some |λ| > 1
    Saddle,

    /// On stability boundary: some |λ| = 1
    Marginal,

    /// Cannot determine
    Unknown,
}

impl DiscreteStabilityType {
    /// Check if the fixed point is stable
    pub fn is_stable(&self) -> bool {
        matches!(
            self,
            DiscreteStabilityType::AsymptoticallyStable
                | DiscreteStabilityType::StableNode
                | DiscreteStabilityType::StableFocus
        )
    }

    /// Get a short description
    pub fn description(&self) -> &'static str {
        match self {
            DiscreteStabilityType::AsymptoticallyStable => "asymptotically stable",
            DiscreteStabilityType::StableNode => "stable node",
            DiscreteStabilityType::StableFocus => "stable focus",
            DiscreteStabilityType::Unstable => "unstable",
            DiscreteStabilityType::UnstableNode => "unstable node",
            DiscreteStabilityType::UnstableFocus => "unstable focus",
            DiscreteStabilityType::Saddle => "saddle",
            DiscreteStabilityType::Marginal => "marginal (on stability boundary)",
            DiscreteStabilityType::Unknown => "unknown",
        }
    }
}

impl fmt::Display for DiscreteStabilityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Classify discrete-time stability based on eigenvalues
///
/// # Arguments
///
/// * `eigenvalues` - Vector of (real_part, imaginary_part) pairs
/// * `tolerance` - Threshold for considering |λ| - 1 as zero
pub fn classify_discrete_from_eigenvalues(
    eigenvalues: &[(f64, f64)],
    tolerance: f64,
) -> DiscreteStabilityType {
    if eigenvalues.is_empty() {
        return DiscreteStabilityType::Unknown;
    }

    let mut has_inside = false; // |λ| < 1
    let mut has_outside = false; // |λ| > 1
    let mut has_boundary = false; // |λ| ≈ 1
    let mut all_real = true;

    for &(re, im) in eigenvalues {
        let magnitude = (re * re + im * im).sqrt();

        if magnitude < 1.0 - tolerance {
            has_inside = true;
        } else if magnitude > 1.0 + tolerance {
            has_outside = true;
        } else {
            has_boundary = true;
        }

        if im.abs() > tolerance {
            all_real = false;
        }
    }

    // Marginal stability
    if has_boundary {
        return DiscreteStabilityType::Marginal;
    }

    // Saddle
    if has_inside && has_outside {
        return DiscreteStabilityType::Saddle;
    }

    // Stable
    if has_inside && !has_outside {
        if all_real {
            return DiscreteStabilityType::StableNode;
        } else {
            return DiscreteStabilityType::StableFocus;
        }
    }

    // Unstable
    if has_outside && !has_inside {
        if all_real {
            return DiscreteStabilityType::UnstableNode;
        } else {
            return DiscreteStabilityType::UnstableFocus;
        }
    }

    DiscreteStabilityType::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_node() {
        // Two negative real eigenvalues
        let eigenvalues = vec![(-1.0, 0.0), (-2.0, 0.0)];
        let result = classify_from_eigenvalues(&eigenvalues, 1e-10);
        assert_eq!(result, StabilityType::StableNode);
        assert!(result.is_stable());
        assert!(result.is_asymptotically_stable());
    }

    #[test]
    fn test_stable_focus() {
        // Complex eigenvalues with negative real part
        let eigenvalues = vec![(-1.0, 1.0), (-1.0, -1.0)];
        let result = classify_from_eigenvalues(&eigenvalues, 1e-10);
        assert_eq!(result, StabilityType::StableFocus);
        assert!(result.is_stable());
        assert!(result.is_spiral());
    }

    #[test]
    fn test_unstable_node() {
        // Two positive real eigenvalues
        let eigenvalues = vec![(1.0, 0.0), (2.0, 0.0)];
        let result = classify_from_eigenvalues(&eigenvalues, 1e-10);
        assert_eq!(result, StabilityType::UnstableNode);
        assert!(result.is_unstable());
    }

    #[test]
    fn test_unstable_focus() {
        // Complex eigenvalues with positive real part
        let eigenvalues = vec![(1.0, 1.0), (1.0, -1.0)];
        let result = classify_from_eigenvalues(&eigenvalues, 1e-10);
        assert_eq!(result, StabilityType::UnstableFocus);
        assert!(result.is_unstable());
        assert!(result.is_spiral());
    }

    #[test]
    fn test_saddle() {
        // One positive, one negative
        let eigenvalues = vec![(1.0, 0.0), (-1.0, 0.0)];
        let result = classify_from_eigenvalues(&eigenvalues, 1e-10);
        assert_eq!(result, StabilityType::Saddle);
        assert!(result.is_saddle());
        assert!(result.is_unstable());
    }

    #[test]
    fn test_center() {
        // Purely imaginary eigenvalues
        let eigenvalues = vec![(0.0, 1.0), (0.0, -1.0)];
        let result = classify_from_eigenvalues(&eigenvalues, 1e-10);
        assert_eq!(result, StabilityType::Center);
        assert!(result.is_center());
    }

    #[test]
    fn test_degenerate() {
        // Zero eigenvalue
        let eigenvalues = vec![(0.0, 0.0), (-1.0, 0.0)];
        let result = classify_from_eigenvalues(&eigenvalues, 1e-10);
        assert_eq!(result, StabilityType::Degenerate);
        assert!(result.is_degenerate());
    }

    #[test]
    fn test_discrete_stable() {
        // |λ| < 1
        let eigenvalues = vec![(0.5, 0.0), (0.3, 0.0)];
        let result = classify_discrete_from_eigenvalues(&eigenvalues, 1e-10);
        assert_eq!(result, DiscreteStabilityType::StableNode);
        assert!(result.is_stable());
    }

    #[test]
    fn test_discrete_unstable() {
        // |λ| > 1
        let eigenvalues = vec![(1.5, 0.0), (2.0, 0.0)];
        let result = classify_discrete_from_eigenvalues(&eigenvalues, 1e-10);
        assert_eq!(result, DiscreteStabilityType::UnstableNode);
    }

    #[test]
    fn test_discrete_saddle() {
        // One inside, one outside
        let eigenvalues = vec![(0.5, 0.0), (1.5, 0.0)];
        let result = classify_discrete_from_eigenvalues(&eigenvalues, 1e-10);
        assert_eq!(result, DiscreteStabilityType::Saddle);
    }

    #[test]
    fn test_stability_type_display() {
        let st = StabilityType::StableFocus;
        assert_eq!(format!("{}", st), "stable focus (spiral sink)");
    }
}
