//! Bifurcation diagram generation
//!
//! This module provides tools for generating and visualizing bifurcation diagrams,
//! which show how system behavior changes across parameter space.
//!
//! # Overview
//!
//! A bifurcation diagram plots a state variable (or characteristic of the attractor)
//! against a bifurcation parameter. Key features include:
//!
//! - Branches of fixed points (stable/unstable)
//! - Bifurcation points where branches meet or stability changes
//! - Periodic orbit amplitudes
//! - Chaotic bands
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::bifurcation::{BifurcationDiagram, DiagramConfig};
//!
//! let config = DiagramConfig {
//!     param_range: (0.0, 4.0),
//!     num_param_points: 1000,
//!     ..Default::default()
//! };
//!
//! let diagram = BifurcationDiagram::compute(&mut system, &config)?;
//! for point in diagram.points() {
//!     println!("μ = {:.3}, x = {:.3}, stable = {}", point.param, point.value, point.stable);
//! }
//! ```

use amari_core::Multivector;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::Result;
use crate::flow::ParametricSystem;

use super::continuation::NaturalContinuation;
use super::traits::{ParameterContinuation, SolutionBranch};
use super::types::{BifurcationConfig, BifurcationPoint};

/// A point in a bifurcation diagram
#[derive(Debug, Clone, Copy)]
pub struct DiagramPoint {
    /// Parameter value
    pub param: f64,

    /// State variable value (projection to single dimension)
    pub value: f64,

    /// Whether this point is stable
    pub stable: bool,

    /// Branch index (for distinguishing multiple solution branches)
    pub branch: usize,
}

/// Configuration for bifurcation diagram computation
#[derive(Debug, Clone)]
pub struct DiagramConfig {
    /// Parameter range (start, end)
    pub param_range: (f64, f64),

    /// Number of parameter values to sample
    pub num_param_points: usize,

    /// Index of state component to plot (default: 0)
    pub state_index: usize,

    /// Configuration for bifurcation detection
    pub bifurcation_config: BifurcationConfig,

    /// Number of initial conditions to try at each parameter
    pub num_initial_conditions: usize,

    /// Range for generating random initial conditions
    pub initial_condition_range: f64,

    /// Number of transient iterations before sampling (for maps)
    pub transient_iterations: usize,

    /// Number of iterations to sample (for maps)
    pub sample_iterations: usize,
}

impl Default for DiagramConfig {
    fn default() -> Self {
        Self {
            param_range: (0.0, 1.0),
            num_param_points: 500,
            state_index: 0,
            bifurcation_config: BifurcationConfig::default(),
            num_initial_conditions: 5,
            initial_condition_range: 1.0,
            transient_iterations: 1000,
            sample_iterations: 100,
        }
    }
}

impl DiagramConfig {
    /// Create configuration for detailed analysis
    pub fn detailed() -> Self {
        Self {
            num_param_points: 2000,
            num_initial_conditions: 10,
            bifurcation_config: BifurcationConfig::fine(),
            ..Default::default()
        }
    }

    /// Create configuration for quick overview
    pub fn quick() -> Self {
        Self {
            num_param_points: 200,
            num_initial_conditions: 3,
            bifurcation_config: BifurcationConfig::coarse(),
            ..Default::default()
        }
    }
}

/// A complete bifurcation diagram
#[derive(Debug, Clone)]
pub struct BifurcationDiagram<const P: usize, const Q: usize, const R: usize> {
    /// All points in the diagram
    points: Vec<DiagramPoint>,

    /// Solution branches
    branches: Vec<SolutionBranch<P, Q, R>>,

    /// Detected bifurcation points
    bifurcations: Vec<BifurcationPoint<P, Q, R>>,

    /// Parameter range
    param_range: (f64, f64),

    /// State component index that was plotted
    #[allow(dead_code)]
    state_index: usize,
}

impl<const P: usize, const Q: usize, const R: usize> BifurcationDiagram<P, Q, R> {
    /// Create an empty diagram
    pub fn new(param_range: (f64, f64), state_index: usize) -> Self {
        Self {
            points: Vec::new(),
            branches: Vec::new(),
            bifurcations: Vec::new(),
            param_range,
            state_index,
        }
    }

    /// Compute a bifurcation diagram for continuous-time systems
    ///
    /// This traces solution branches starting from a grid of initial conditions,
    /// detecting bifurcations along the way.
    pub fn compute_continuous<S>(system: &mut S, config: &DiagramConfig) -> Result<Self>
    where
        S: ParametricSystem<P, Q, R, Parameter = f64>,
    {
        let mut diagram = Self::new(config.param_range, config.state_index);

        // Generate initial conditions
        let initial_states = Self::generate_initial_conditions(
            config.num_initial_conditions,
            config.initial_condition_range,
        );

        // Use simple continuation to trace branches
        let continuation = NaturalContinuation::new();

        for (branch_id, initial_state) in initial_states.iter().enumerate() {
            // Continue from both ends of parameter range
            for &start_param in &[config.param_range.0, config.param_range.1] {
                match continuation.continue_branch(
                    system,
                    initial_state.clone(),
                    start_param,
                    config.param_range,
                    &config.bifurcation_config,
                ) {
                    Ok((mut branch, bifurcations)) => {
                        branch.branch_id = branch_id;

                        // Extract diagram points
                        for point in &branch.points {
                            diagram.points.push(DiagramPoint {
                                param: point.parameter,
                                value: point.state.get(config.state_index),
                                stable: point.is_stable,
                                branch: branch_id,
                            });
                        }

                        diagram.branches.push(branch);
                        diagram.bifurcations.extend(bifurcations);
                    }
                    Err(_) => continue,
                }
            }
        }

        // Sort points by parameter
        diagram.sort_points();

        // Remove duplicate bifurcations
        diagram.deduplicate_bifurcations(&config.bifurcation_config);

        Ok(diagram)
    }

    /// Compute a bifurcation diagram using parallel parameter sweep
    ///
    /// This is faster for systems where each parameter value can be
    /// computed independently.
    #[cfg(feature = "parallel")]
    pub fn compute_parallel<S, F>(system_factory: F, config: &DiagramConfig) -> Result<Self>
    where
        S: ParametricSystem<P, Q, R, Parameter = f64> + Send + Sync,
        F: Fn(f64) -> S + Send + Sync,
    {
        let params: Vec<f64> = (0..config.num_param_points)
            .map(|i| {
                let t = i as f64 / (config.num_param_points - 1) as f64;
                config.param_range.0 + t * (config.param_range.1 - config.param_range.0)
            })
            .collect();

        let initial_states = Self::generate_initial_conditions(
            config.num_initial_conditions,
            config.initial_condition_range,
        );

        let fp_config = crate::stability::FixedPointConfig::default();
        let diff_config = crate::stability::DifferentiationConfig::default();

        let points: Vec<DiagramPoint> = params
            .par_iter()
            .flat_map(|&param| {
                let system = system_factory(param);
                let mut local_points = Vec::new();

                for (branch_id, initial) in initial_states.iter().enumerate() {
                    if let Ok(fp_result) =
                        crate::stability::find_fixed_point(&system, initial, &fp_config)
                    {
                        if fp_result.converged {
                            let is_stable = if let Ok(jac) = crate::stability::compute_jacobian(
                                &system,
                                &fp_result.point,
                                &diff_config,
                            ) {
                                if let Ok(eigenvalues) = crate::stability::compute_eigenvalues(&jac)
                                {
                                    eigenvalues.iter().all(|(re, _)| *re < 0.0)
                                } else {
                                    false
                                }
                            } else {
                                false
                            };

                            local_points.push(DiagramPoint {
                                param,
                                value: fp_result.point.get(config.state_index),
                                stable: is_stable,
                                branch: branch_id,
                            });
                        }
                    }
                }

                local_points
            })
            .collect();

        let mut diagram = Self::new(config.param_range, config.state_index);
        diagram.points = points;
        diagram.sort_points();

        Ok(diagram)
    }

    /// Generate random initial conditions
    fn generate_initial_conditions(num: usize, range: f64) -> Vec<Multivector<P, Q, R>> {
        let dim = 1 << (P + Q + R);
        let mut states = Vec::with_capacity(num);

        // Include origin
        states.push(Multivector::zero());

        // Add some standard points
        if num > 1 {
            let mut positive = Multivector::zero();
            positive.set(0, range);
            states.push(positive);
        }

        if num > 2 {
            let mut negative = Multivector::zero();
            negative.set(0, -range);
            states.push(negative);
        }

        // Add random-ish points (deterministic for reproducibility)
        for i in 3..num {
            let mut state = Multivector::zero();
            let seed = (i as f64) / (num as f64);

            for j in 0..dim.min(3) {
                // Only vary first few components
                let val = range * (2.0 * ((seed * (j + 1) as f64 * 7.123).sin()) - 1.0);
                state.set(j, val);
            }

            states.push(state);
        }

        states
    }

    /// Sort points by parameter value
    fn sort_points(&mut self) {
        self.points
            .sort_by(|a, b| a.param.partial_cmp(&b.param).unwrap());
    }

    /// Remove duplicate bifurcation points
    fn deduplicate_bifurcations(&mut self, config: &BifurcationConfig) {
        self.bifurcations
            .sort_by(|a, b| a.parameter_value.partial_cmp(&b.parameter_value).unwrap());

        let tolerance = config.criticality.zero_tolerance * 10.0;
        let mut unique = Vec::new();

        for bif in &self.bifurcations {
            let dominated = unique.iter().any(|existing: &BifurcationPoint<P, Q, R>| {
                (existing.parameter_value - bif.parameter_value).abs() < tolerance
            });

            if !dominated {
                unique.push(bif.clone());
            }
        }

        self.bifurcations = unique;
    }

    /// Get all points in the diagram
    pub fn points(&self) -> &[DiagramPoint] {
        &self.points
    }

    /// Get stable points only
    pub fn stable_points(&self) -> Vec<&DiagramPoint> {
        self.points.iter().filter(|p| p.stable).collect()
    }

    /// Get unstable points only
    pub fn unstable_points(&self) -> Vec<&DiagramPoint> {
        self.points.iter().filter(|p| !p.stable).collect()
    }

    /// Get all solution branches
    pub fn branches(&self) -> &[SolutionBranch<P, Q, R>] {
        &self.branches
    }

    /// Get all detected bifurcations
    pub fn bifurcations(&self) -> &[BifurcationPoint<P, Q, R>] {
        &self.bifurcations
    }

    /// Get the parameter range
    pub fn param_range(&self) -> (f64, f64) {
        self.param_range
    }

    /// Get the value range (min, max) of the plotted variable
    pub fn value_range(&self) -> (f64, f64) {
        if self.points.is_empty() {
            return (0.0, 0.0);
        }

        let min = self
            .points
            .iter()
            .map(|p| p.value)
            .fold(f64::INFINITY, f64::min);
        let max = self
            .points
            .iter()
            .map(|p| p.value)
            .fold(f64::NEG_INFINITY, f64::max);

        (min, max)
    }

    /// Get number of points in the diagram
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if diagram is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Export diagram data as (param, value, stable) tuples
    pub fn export(&self) -> Vec<(f64, f64, bool)> {
        self.points
            .iter()
            .map(|p| (p.param, p.value, p.stable))
            .collect()
    }
}

/// Builder for creating bifurcation diagrams with custom settings
#[derive(Debug, Clone)]
pub struct DiagramBuilder<const P: usize, const Q: usize, const R: usize> {
    config: DiagramConfig,
    _phantom: std::marker::PhantomData<Multivector<P, Q, R>>,
}

impl<const P: usize, const Q: usize, const R: usize> Default for DiagramBuilder<P, Q, R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const P: usize, const Q: usize, const R: usize> DiagramBuilder<P, Q, R> {
    /// Create a new diagram builder
    pub fn new() -> Self {
        Self {
            config: DiagramConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the parameter range
    pub fn param_range(mut self, start: f64, end: f64) -> Self {
        self.config.param_range = (start, end);
        self
    }

    /// Set the number of parameter points
    pub fn num_points(mut self, n: usize) -> Self {
        self.config.num_param_points = n;
        self
    }

    /// Set which state component to plot
    pub fn state_index(mut self, idx: usize) -> Self {
        self.config.state_index = idx;
        self
    }

    /// Set the number of initial conditions to try
    pub fn num_initial_conditions(mut self, n: usize) -> Self {
        self.config.num_initial_conditions = n;
        self
    }

    /// Build and compute the diagram
    pub fn build<S>(self, system: &mut S) -> Result<BifurcationDiagram<P, Q, R>>
    where
        S: ParametricSystem<P, Q, R, Parameter = f64>,
    {
        BifurcationDiagram::compute_continuous(system, &self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::DynamicalSystem;

    // Pitchfork normal form: dx/dt = μx - x³
    struct PitchforkNormalForm {
        mu: f64,
    }

    impl DynamicalSystem<1, 0, 0> for PitchforkNormalForm {
        fn vector_field(&self, state: &Multivector<1, 0, 0>) -> Result<Multivector<1, 0, 0>> {
            let x = state.get(0);
            let mut result = Multivector::zero();
            result.set(0, self.mu * x - x * x * x);
            Ok(result)
        }
    }

    impl ParametricSystem<1, 0, 0> for PitchforkNormalForm {
        type Parameter = f64;

        fn parameter(&self) -> &f64 {
            &self.mu
        }

        fn set_parameter(&mut self, param: f64) {
            self.mu = param;
        }

        fn vector_field_with_param(
            &self,
            state: &Multivector<1, 0, 0>,
            param: &f64,
        ) -> Result<Multivector<1, 0, 0>> {
            let x = state.get(0);
            let mut result = Multivector::zero();
            result.set(0, param * x - x * x * x);
            Ok(result)
        }
    }

    #[test]
    fn test_diagram_config() {
        let default = DiagramConfig::default();
        let detailed = DiagramConfig::detailed();
        let quick = DiagramConfig::quick();

        assert!(detailed.num_param_points > default.num_param_points);
        assert!(quick.num_param_points < default.num_param_points);
    }

    #[test]
    fn test_diagram_creation() {
        let diagram = BifurcationDiagram::<1, 0, 0>::new((0.0, 1.0), 0);
        assert!(diagram.is_empty());
        assert_eq!(diagram.param_range(), (0.0, 1.0));
    }

    #[test]
    fn test_diagram_builder() {
        let mut system = PitchforkNormalForm { mu: 0.0 };

        let config = DiagramConfig {
            param_range: (-1.0, 1.0),
            num_param_points: 50,
            num_initial_conditions: 3,
            bifurcation_config: BifurcationConfig::coarse(),
            ..Default::default()
        };

        let result = BifurcationDiagram::compute_continuous(&mut system, &config);
        assert!(result.is_ok());

        let diagram = result.unwrap();
        assert!(!diagram.is_empty());
    }

    #[test]
    fn test_generate_initial_conditions() {
        let states = BifurcationDiagram::<2, 0, 0>::generate_initial_conditions(5, 1.0);
        assert_eq!(states.len(), 5);

        // First should be origin
        let origin = &states[0];
        assert!((origin.get(0)).abs() < 1e-10);
    }

    #[test]
    fn test_diagram_export() {
        let mut diagram = BifurcationDiagram::<1, 0, 0>::new((0.0, 1.0), 0);
        diagram.points.push(DiagramPoint {
            param: 0.5,
            value: 1.0,
            stable: true,
            branch: 0,
        });

        let exported = diagram.export();
        assert_eq!(exported.len(), 1);
        assert_eq!(exported[0], (0.5, 1.0, true));
    }

    #[test]
    fn test_value_range() {
        let mut diagram = BifurcationDiagram::<1, 0, 0>::new((0.0, 1.0), 0);
        diagram.points.push(DiagramPoint {
            param: 0.0,
            value: -1.0,
            stable: true,
            branch: 0,
        });
        diagram.points.push(DiagramPoint {
            param: 1.0,
            value: 2.0,
            stable: true,
            branch: 0,
        });

        let (min, max) = diagram.value_range();
        assert_eq!(min, -1.0);
        assert_eq!(max, 2.0);
    }
}
