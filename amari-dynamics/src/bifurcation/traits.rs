//! Bifurcation detection traits
//!
//! This module defines the core trait for bifurcation detection algorithms.

use amari_core::Multivector;

use crate::error::Result;
use crate::flow::ParametricSystem;

use super::types::{BifurcationConfig, BifurcationPoint, BifurcationType};

/// Trait for bifurcation detection algorithms
///
/// Implementations can detect different types of bifurcations in
/// parametric dynamical systems.
pub trait BifurcationDetector<const P: usize, const Q: usize, const R: usize> {
    /// Detect bifurcations in a parameter range
    ///
    /// # Arguments
    ///
    /// * `system` - The parametric dynamical system
    /// * `param_range` - (start, end) parameter values to scan
    /// * `config` - Detection configuration
    ///
    /// # Returns
    ///
    /// A vector of detected bifurcation points, sorted by parameter value.
    fn detect<S>(
        &self,
        system: &mut S,
        param_range: (f64, f64),
        config: &BifurcationConfig,
    ) -> Result<Vec<BifurcationPoint<P, Q, R>>>
    where
        S: ParametricSystem<P, Q, R, Parameter = f64>;

    /// Refine a suspected bifurcation point to higher precision
    ///
    /// Given an approximate bifurcation location, use bisection or
    /// other methods to locate it more precisely.
    fn refine<S>(
        &self,
        system: &mut S,
        approximate: &BifurcationPoint<P, Q, R>,
        tolerance: f64,
    ) -> Result<BifurcationPoint<P, Q, R>>
    where
        S: ParametricSystem<P, Q, R, Parameter = f64>;

    /// Classify the bifurcation type from eigenvalue data
    fn classify(&self, eigenvalues: &[(f64, f64)], config: &BifurcationConfig) -> BifurcationType;
}

/// Result of tracking a solution branch
#[derive(Debug, Clone)]
pub struct BranchPoint<const P: usize, const Q: usize, const R: usize> {
    /// Parameter value
    pub parameter: f64,

    /// State at this parameter value (e.g., fixed point)
    pub state: Multivector<P, Q, R>,

    /// Eigenvalues at this point
    pub eigenvalues: Vec<(f64, f64)>,

    /// Whether this point is stable
    pub is_stable: bool,
}

impl<const P: usize, const Q: usize, const R: usize> BranchPoint<P, Q, R> {
    /// Create a new branch point
    pub fn new(parameter: f64, state: Multivector<P, Q, R>, eigenvalues: Vec<(f64, f64)>) -> Self {
        let is_stable = eigenvalues.iter().all(|(re, _)| *re < 0.0);
        Self {
            parameter,
            state,
            eigenvalues,
            is_stable,
        }
    }
}

/// A solution branch in parameter space
///
/// Represents a continuous curve of solutions (fixed points, periodic orbits)
/// parameterized by the bifurcation parameter.
#[derive(Debug, Clone)]
pub struct SolutionBranch<const P: usize, const Q: usize, const R: usize> {
    /// Points along the branch, sorted by parameter
    pub points: Vec<BranchPoint<P, Q, R>>,

    /// Branch identifier for distinguishing multiple branches
    pub branch_id: usize,

    /// Parameter range covered by this branch
    pub param_range: (f64, f64),
}

impl<const P: usize, const Q: usize, const R: usize> SolutionBranch<P, Q, R> {
    /// Create a new solution branch
    pub fn new(branch_id: usize) -> Self {
        Self {
            points: Vec::new(),
            branch_id,
            param_range: (f64::INFINITY, f64::NEG_INFINITY),
        }
    }

    /// Add a point to the branch
    pub fn push(&mut self, point: BranchPoint<P, Q, R>) {
        if point.parameter < self.param_range.0 {
            self.param_range.0 = point.parameter;
        }
        if point.parameter > self.param_range.1 {
            self.param_range.1 = point.parameter;
        }
        self.points.push(point);
    }

    /// Sort points by parameter value
    pub fn sort_by_parameter(&mut self) {
        self.points
            .sort_by(|a, b| a.parameter.partial_cmp(&b.parameter).unwrap());
    }

    /// Find points where stability changes
    pub fn stability_changes(&self) -> Vec<(f64, f64)> {
        let mut changes = Vec::new();

        for window in self.points.windows(2) {
            if window[0].is_stable != window[1].is_stable {
                // Interpolate parameter value
                let p0 = window[0].parameter;
                let p1 = window[1].parameter;
                changes.push((p0, p1));
            }
        }

        changes
    }

    /// Get the number of points in this branch
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the branch is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Interpolate state at a given parameter value
    pub fn interpolate(&self, parameter: f64) -> Option<Multivector<P, Q, R>> {
        if self.points.is_empty() {
            return None;
        }

        // Find bracketing points
        for window in self.points.windows(2) {
            let p0 = window[0].parameter;
            let p1 = window[1].parameter;

            if parameter >= p0 && parameter <= p1 {
                let t = (parameter - p0) / (p1 - p0);
                // Linear interpolation
                let s0 = &window[0].state;
                let s1 = &window[1].state;
                return Some(&(s0 * (1.0 - t)) + &(s1 * t));
            }
        }

        // Check if parameter is before first point
        if parameter < self.points[0].parameter {
            return Some(self.points[0].state.clone());
        }

        // After last point
        if parameter > self.points.last()?.parameter {
            return Some(self.points.last()?.state.clone());
        }

        None
    }
}

/// Trait for parameter continuation methods
///
/// Continuation follows solution curves as parameters vary, detecting
/// bifurcations when stability changes or branches collide.
pub trait ParameterContinuation<const P: usize, const Q: usize, const R: usize> {
    /// Continue a solution branch from an initial point
    ///
    /// # Arguments
    ///
    /// * `system` - The parametric system
    /// * `initial_state` - Starting state (should be a fixed point)
    /// * `initial_param` - Starting parameter value
    /// * `param_range` - (min, max) parameter values to explore
    /// * `config` - Continuation configuration
    ///
    /// # Returns
    ///
    /// The traced solution branch and any detected bifurcations.
    fn continue_branch<S>(
        &self,
        system: &mut S,
        initial_state: Multivector<P, Q, R>,
        initial_param: f64,
        param_range: (f64, f64),
        config: &BifurcationConfig,
    ) -> Result<(SolutionBranch<P, Q, R>, Vec<BifurcationPoint<P, Q, R>>)>
    where
        S: ParametricSystem<P, Q, R, Parameter = f64>;

    /// Follow multiple branches from initial conditions
    fn trace_all_branches<S>(
        &self,
        system: &mut S,
        initial_states: &[Multivector<P, Q, R>],
        initial_param: f64,
        param_range: (f64, f64),
        config: &BifurcationConfig,
    ) -> Result<Vec<(SolutionBranch<P, Q, R>, Vec<BifurcationPoint<P, Q, R>>)>>
    where
        S: ParametricSystem<P, Q, R, Parameter = f64>,
    {
        let mut results = Vec::new();

        for state in initial_states {
            let result =
                self.continue_branch(system, state.clone(), initial_param, param_range, config)?;
            results.push(result);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_point_stability() {
        let state = Multivector::<2, 0, 0>::zero();

        // Stable: all negative real parts
        let eigenvalues = vec![(-1.0, 0.0), (-2.0, 0.0)];
        let bp = BranchPoint::new(0.0, state.clone(), eigenvalues);
        assert!(bp.is_stable);

        // Unstable: one positive real part
        let eigenvalues2 = vec![(0.1, 0.0), (-2.0, 0.0)];
        let bp2 = BranchPoint::new(0.0, state, eigenvalues2);
        assert!(!bp2.is_stable);
    }

    #[test]
    fn test_solution_branch_operations() {
        let mut branch = SolutionBranch::<2, 0, 0>::new(0);

        assert!(branch.is_empty());

        let state = Multivector::<2, 0, 0>::zero();
        branch.push(BranchPoint::new(0.0, state.clone(), vec![(-1.0, 0.0)]));
        branch.push(BranchPoint::new(1.0, state.clone(), vec![(0.1, 0.0)]));

        assert_eq!(branch.len(), 2);
        assert_eq!(branch.param_range, (0.0, 1.0));

        // Should find stability change
        let changes = branch.stability_changes();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0], (0.0, 1.0));
    }

    #[test]
    fn test_branch_interpolation() {
        let mut branch = SolutionBranch::<1, 0, 0>::new(0);

        let mut s0 = Multivector::<1, 0, 0>::zero();
        s0.set(0, 0.0);
        let mut s1 = Multivector::<1, 0, 0>::zero();
        s1.set(0, 1.0);

        branch.push(BranchPoint::new(0.0, s0, vec![(-1.0, 0.0)]));
        branch.push(BranchPoint::new(1.0, s1, vec![(-1.0, 0.0)]));
        branch.sort_by_parameter();

        // Midpoint should interpolate
        let mid = branch.interpolate(0.5).unwrap();
        assert!((mid.get(0) - 0.5).abs() < 1e-10);
    }
}
