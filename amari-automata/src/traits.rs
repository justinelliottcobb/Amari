//! Core traits for automata operations

use crate::AutomataResult;

/// Trait for systems that can evolve over time
///
/// Represents automata and other systems that have discrete time evolution,
/// maintaining a generation counter and supporting stepwise evolution.
pub trait Evolvable {
    /// Perform one evolution step
    ///
    /// Advances the system by one time step according to its evolution rules.
    /// Updates the internal state and increments the generation counter.
    ///
    /// # Errors
    ///
    /// Returns an error if the evolution step fails due to invalid state,
    /// rule application errors, or other constraints.
    fn step(&mut self) -> AutomataResult<()>;

    /// Get the current generation number
    ///
    /// Returns the number of evolution steps that have been performed
    /// since the system was created or last reset.
    fn generation(&self) -> usize;

    /// Reset the system to its initial state
    ///
    /// Clears all state and resets the generation counter to zero.
    /// The system returns to a clean initial configuration.
    fn reset(&mut self);
}

/// Trait for systems that can self-assemble from components
///
/// Represents geometric self-assembly where components with affinities
/// automatically arrange themselves into stable configurations based on
/// geometric algebra operations.
pub trait SelfAssembling {
    /// The type of individual components to be assembled
    type Component;

    /// The type of the assembled structure
    type Assembly;

    /// Assemble components into a configuration
    ///
    /// Takes a collection of components and assembles them into a stable
    /// configuration based on their affinities and geometric properties.
    ///
    /// # Arguments
    ///
    /// * `components` - The components to assemble
    ///
    /// # Errors
    ///
    /// Returns an error if assembly fails due to:
    /// - Invalid component specifications
    /// - Conflicting constraints
    /// - Energy threshold violations
    /// - Connection errors
    fn assemble(&self, components: &[Self::Component]) -> AutomataResult<Self::Assembly>;

    /// Check if an assembly is stable
    ///
    /// Determines whether an assembled configuration meets stability
    /// criteria such as energy thresholds and structural constraints.
    ///
    /// # Arguments
    ///
    /// * `assembly` - The assembly to check
    ///
    /// # Returns
    ///
    /// `true` if the assembly is stable, `false` otherwise
    fn is_stable(&self, assembly: &Self::Assembly) -> bool;

    /// Compute affinity between two components
    ///
    /// Calculates the geometric affinity between two components,
    /// which determines their tendency to connect during assembly.
    /// Higher values indicate stronger affinity.
    ///
    /// # Arguments
    ///
    /// * `a` - First component
    /// * `b` - Second component
    ///
    /// # Returns
    ///
    /// The affinity value (typically in range [0, 1])
    fn affinity(&self, a: &Self::Component, b: &Self::Component) -> f64;
}

/// Trait for inverse design of automata configurations
///
/// Enables finding initial configurations (seeds) that evolve to produce
/// desired target patterns. Uses automatic differentiation through dual
/// numbers for gradient-based optimization.
pub trait InverseDesignable {
    /// The type representing the desired target pattern
    type Target;

    /// The type representing an automaton configuration
    type Configuration;

    /// Find a seed configuration that evolves to the target
    ///
    /// Performs optimization to discover an initial configuration that,
    /// when evolved, produces a pattern matching the target.
    ///
    /// # Arguments
    ///
    /// * `target` - The desired target pattern
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No suitable configuration is found within iteration limits
    /// - The target pattern is invalid or unreachable
    /// - Optimization fails to converge
    fn find_seed(&self, target: &Self::Target) -> AutomataResult<Self::Configuration>;

    /// Compute fitness of a configuration relative to a target
    ///
    /// Evaluates how well a configuration, when evolved, matches the
    /// target pattern. Lower values indicate better matches.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to evaluate
    /// * `target` - The target pattern to match
    ///
    /// # Returns
    ///
    /// A fitness score, where 0 is a perfect match and higher values
    /// indicate greater distance from the target
    fn fitness(&self, config: &Self::Configuration, target: &Self::Target) -> f64;
}
