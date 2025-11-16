//! Pushforward and pullback of measures
//!
//! This module implements measure transformations under measurable functions,
//! enabling change of variables and measure transport.
//!
//! # Core Concepts
//!
//! ## Pushforward (Push-forward)
//!
//! Given a measurable function f: (X, Σ) → (Y, Γ) and a measure μ on X,
//! the **pushforward** f₊μ is a measure on Y defined by:
//!
//! (f₊μ)(B) = μ(f⁻¹(B)) for all measurable sets B ∈ Γ
//!
//! The pushforward "pushes" the measure forward along the function f.
//!
//! ### Properties
//!
//! - **Measure preservation**: If f is bijective and measure-preserving, f₊μ = ν
//! - **Composition**: (g ∘ f)₊μ = g₊(f₊μ)
//! - **Integration**: ∫_Y φ d(f₊μ) = ∫_X (φ ∘ f) dμ
//! - **Additivity**: f₊(μ₁ + μ₂) = f₊μ₁ + f₊μ₂
//!
//! ## Pullback (Pull-back)
//!
//! The **pullback** f*ν of a measure ν on Y to X requires additional structure.
//! For general measurable functions, pullback may not exist. Common cases:
//!
//! - **Diffeomorphisms**: f*ν(A) = ∫_A |det Df| dν for smooth bijective f
//! - **Covering maps**: f*ν exists with appropriate multiplicity factors
//! - **Absolutely continuous**: When f has density with respect to Lebesgue measure
//!
//! ### Change of Variables Formula
//!
//! For a diffeomorphism f: ℝⁿ → ℝⁿ and Lebesgue measure λ:
//!
//! ∫_f(A) g(y) dλ(y) = ∫_A g(f(x)) |det Df(x)| dλ(x)
//!
//! This relates pushforward and pullback: f₊λ = (|det Df|)⁻¹ · f*λ
//!
//! # Examples
//!
//! ```
//! // TODO: Add examples once integration with Measure trait is complete
//! ```

use crate::error::{MeasureError, Result};
use crate::measure::Measure;
use crate::phantom::*;
use crate::sigma_algebra::SigmaAlgebra;
use core::marker::PhantomData;

/// Measurable function between σ-algebras
///
/// A function f: X → Y is measurable if the preimage of every measurable set
/// in Y is measurable in X:
///
/// B ∈ Γ ⟹ f⁻¹(B) ∈ Σ
///
/// # Type Parameters
///
/// - `S1`: Source σ-algebra
/// - `S2`: Target σ-algebra
///
/// # Mathematical Foundation
///
/// Measurability is the key property that ensures the pushforward is well-defined.
/// Without measurability, f⁻¹(B) might not be measurable, preventing us from
/// computing μ(f⁻¹(B)).
///
/// # Implementation Note
///
/// This is a foundational trait establishing the structure. Full implementations
/// with concrete function types will be added when integrating with amari-core's
/// geometric types.
pub trait MeasurableFunction<S1, S2>
where
    S1: SigmaAlgebra,
    S2: SigmaAlgebra,
{
    /// Evaluate the function at a point
    ///
    /// # Arguments
    ///
    /// * `x` - Point in the source space
    ///
    /// # Returns
    ///
    /// The image of x under the function
    fn apply(&self, x: &S1::Set) -> Result<S2::Set>;

    /// Compute the preimage f⁻¹(B) of a set B
    ///
    /// # Arguments
    ///
    /// * `b` - Measurable set in the target space
    ///
    /// # Returns
    ///
    /// The preimage f⁻¹(B) = {x ∈ X : f(x) ∈ B}
    fn preimage(&self, b: &S2::Set) -> Result<S1::Set>;

    /// Check if the function is measurable
    ///
    /// Verifies that f⁻¹(B) ∈ S1 for all B ∈ S2.
    ///
    /// # Arguments
    ///
    /// * `source` - Source σ-algebra
    /// * `target` - Target σ-algebra
    fn is_measurable(&self, source: &S1, target: &S2) -> Result<bool>;
}

/// Pushforward of a measure under a measurable function
///
/// For a measurable function f: (X, Σ) → (Y, Γ) and measure μ on X,
/// the pushforward f₊μ is defined by:
///
/// (f₊μ)(B) = μ(f⁻¹(B)) for all B ∈ Γ
///
/// # Type Parameters
///
/// - `S1`: Source σ-algebra
/// - `S2`: Target σ-algebra
/// - `M`: Source measure type
/// - `F`: Finiteness property
/// - `S`: Sign property
/// - `C`: Completeness property
///
/// # Mathematical Properties
///
/// - **Well-defined**: Measurability of f ensures f⁻¹(B) ∈ Σ
/// - **Measure axioms**: f₊μ satisfies all measure axioms
/// - **Composition**: (g ∘ f)₊μ = g₊(f₊μ)
/// - **Integration**: ∫_Y φ d(f₊μ) = ∫_X (φ ∘ f) dμ
///
/// # Examples
///
/// use amari_measure::{Pushforward, LebesgueMeasure};
///
/// // Create a linear transformation f(x) = 2x on ℝ
/// // let f = LinearMap::scaling(2.0);
/// // let mu = LebesgueMeasure::new(1);
/// // let f_push_mu = f.pushforward(&mu);
///
/// // For scaling by factor a:
/// // (f₊λ)([0, 1]) = λ(f⁻¹([0, 1])) = λ([0, 1/2]) = 1/2
#[derive(Clone)]
pub struct Pushforward<S1, S2, M, F, S, C>
where
    S1: SigmaAlgebra,
    S2: SigmaAlgebra,
    M: Measure<S1, F, S, C>,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    /// Source measure
    source_measure: M,

    /// Target σ-algebra
    target_sigma: S2,

    /// Phantom type markers
    _phantom: PhantomData<(S1, F, S, C)>,
}

impl<S1, S2, M, F, S, C> Pushforward<S1, S2, M, F, S, C>
where
    S1: SigmaAlgebra,
    S2: SigmaAlgebra,
    M: Measure<S1, F, S, C>,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    /// Create a pushforward measure
    ///
    /// # Arguments
    ///
    /// * `source_measure` - The measure to push forward
    /// * `target_sigma` - The target σ-algebra
    ///
    /// # Examples
    ///
    /// use amari_measure::{Pushforward, LebesgueMeasure, LebesgueSigma};
    /// use amari_measure::{SigmaFinite, Unsigned, Complete};
    ///
    /// let source = LebesgueMeasure::new(1);
    /// let target_sigma = LebesgueSigma::new(1);
    /// let push: Pushforward<_, _, _, SigmaFinite, Unsigned, Complete> =
    ///     Pushforward::new(source, target_sigma);
    /// // Note: This example will compile once Measure trait is fully implemented
    pub fn new(source_measure: M, target_sigma: S2) -> Self {
        Self {
            source_measure,
            target_sigma,
            _phantom: PhantomData,
        }
    }

    /// Get a reference to the source measure
    pub fn source_measure(&self) -> &M {
        &self.source_measure
    }

    /// Get a reference to the target σ-algebra
    pub fn target_sigma(&self) -> &S2 {
        &self.target_sigma
    }

    /// Compute the pushforward measure of a set
    ///
    /// For a set B in the target space, computes (f₊μ)(B) = μ(f⁻¹(B)).
    ///
    /// # Arguments
    ///
    /// * `function` - The measurable function f
    /// * `set` - The set B in the target space
    ///
    /// # Returns
    ///
    /// The measure (f₊μ)(B)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The function is not measurable
    /// - The set is not measurable
    /// - The preimage cannot be computed
    pub fn measure<Fn>(&self, _function: &Fn, _set: &S2::Set) -> Result<f64>
    where
        Fn: MeasurableFunction<S1, S2>,
    {
        Err(MeasureError::computation(
            "Pushforward measure computation not yet implemented",
        ))
    }

    /// Check if the pushforward preserves the measure
    ///
    /// A function f is **measure-preserving** if (f₊μ)(B) = ν(B) for all
    /// measurable sets B, where ν is some reference measure.
    ///
    /// # Examples of measure-preserving maps
    ///
    /// - Translations on ℝⁿ preserve Lebesgue measure
    /// - Rotations on ℝⁿ preserve Lebesgue measure
    /// - Isometries preserve Hausdorff measure
    pub fn is_measure_preserving(&self) -> Result<bool> {
        Err(MeasureError::computation(
            "Measure preservation check not yet implemented",
        ))
    }
}

/// Pullback of a measure under a function
///
/// The pullback f*ν of a measure ν on Y to X is more subtle than the pushforward.
/// It requires additional structure on the function f.
///
/// # Common Cases
///
/// ## Diffeomorphisms
///
/// For a diffeomorphism f: ℝⁿ → ℝⁿ and measure ν on Y:
///
/// (f*ν)(A) = ∫_A |det Df(x)|⁻¹ dν(f(x))
///
/// The Jacobian determinant accounts for how f distorts volumes.
///
/// ## Absolutely Continuous Functions
///
/// For functions with density h = dν/dλ, the pullback can be computed via:
///
/// (f*ν)(A) = ∫_A h(f(x)) |det Df(x)| dλ(x)
///
/// # Type Parameters
///
/// - `S1`: Source σ-algebra
/// - `S2`: Target σ-algebra
/// - `M`: Target measure type
/// - `F`: Finiteness property
/// - `S`: Sign property
/// - `C`: Completeness property
///
/// # Implementation Note
///
/// This is a foundational structure. Full implementations will be added when
/// integrating with amari-core's differential geometry capabilities.
#[derive(Clone)]
pub struct Pullback<S1, S2, M, F, S, C>
where
    S1: SigmaAlgebra,
    S2: SigmaAlgebra,
    M: Measure<S2, F, S, C>,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    /// Target measure
    target_measure: M,

    /// Source σ-algebra
    source_sigma: S1,

    /// Phantom type markers
    _phantom: PhantomData<(S2, F, S, C)>,
}

impl<S1, S2, M, F, S, C> Pullback<S1, S2, M, F, S, C>
where
    S1: SigmaAlgebra,
    S2: SigmaAlgebra,
    M: Measure<S2, F, S, C>,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    /// Create a pullback measure
    ///
    /// # Arguments
    ///
    /// * `target_measure` - The measure to pull back
    /// * `source_sigma` - The source σ-algebra
    ///
    /// # Examples
    ///
    /// use amari_measure::{Pullback, LebesgueMeasure, LebesgueSigma};
    /// use amari_measure::{SigmaFinite, Unsigned, Complete};
    ///
    /// let target = LebesgueMeasure::new(1);
    /// let source_sigma = LebesgueSigma::new(1);
    /// let pull: Pullback<_, _, _, SigmaFinite, Unsigned, Complete> =
    ///     Pullback::new(target, source_sigma);
    /// // Note: This example will compile once Measure trait is fully implemented
    pub fn new(target_measure: M, source_sigma: S1) -> Self {
        Self {
            target_measure,
            source_sigma,
            _phantom: PhantomData,
        }
    }

    /// Get a reference to the target measure
    pub fn target_measure(&self) -> &M {
        &self.target_measure
    }

    /// Get a reference to the source σ-algebra
    pub fn source_sigma(&self) -> &S1 {
        &self.source_sigma
    }

    /// Compute the pullback measure of a set
    ///
    /// For a diffeomorphism f and set A in the source space:
    /// (f*ν)(A) = ∫_A |det Df(x)|⁻¹ dν(f(x))
    ///
    /// # Arguments
    ///
    /// * `function` - The diffeomorphism f
    /// * `set` - The set A in the source space
    ///
    /// # Returns
    ///
    /// The measure (f*ν)(A)
    pub fn measure<Fn>(&self, _function: &Fn, _set: &S1::Set) -> Result<f64>
    where
        Fn: MeasurableFunction<S1, S2>,
    {
        Err(MeasureError::computation(
            "Pullback measure computation not yet implemented",
        ))
    }

    /// Compute the Jacobian determinant at a point
    ///
    /// For a diffeomorphism f: ℝⁿ → ℝⁿ, the Jacobian determinant det Df(x)
    /// measures how f distorts volumes near x.
    ///
    /// # Properties
    ///
    /// - |det Df(x)| = 1 for isometries (distance-preserving maps)
    /// - |det Df(x)| = aⁿ for uniform scaling by factor a in dimension n
    /// - Sign of det Df(x) indicates orientation preservation/reversal
    pub fn jacobian_determinant<Fn>(&self, _function: &Fn, _x: &S1::Set) -> Result<f64>
    where
        Fn: MeasurableFunction<S1, S2>,
    {
        Err(MeasureError::computation(
            "Jacobian determinant computation not yet implemented",
        ))
    }
}

/// Change of variables formula
///
/// For a diffeomorphism f: ℝⁿ → ℝⁿ, integrable function g, and Lebesgue measure λ:
///
/// ∫_f(A) g(y) dλ(y) = ∫_A g(f(x)) |det Df(x)| dλ(x)
///
/// This fundamental formula relates integration in the image and preimage.
///
/// # Arguments
///
/// * `function` - The diffeomorphism f
/// * `g` - The function to integrate
/// * `set` - The integration domain A
///
/// # Returns
///
/// The integral ∫_A g(f(x)) |det Df(x)| dλ(x)
///
/// # Mathematical Foundation
///
/// This formula is the measure-theoretic generalization of the calculus
/// change of variables formula:
///
/// ∫ₐᵇ g(y) dy = ∫_c^d g(f(x)) |f'(x)| dx
///
/// where f: [c,d] → [a,b] is a diffeomorphism.
///
/// # Examples
///
/// ## Polar Coordinates (2D)
///
/// f(r,θ) = (r cos θ, r sin θ)
/// |det Df| = r
///
/// ∫∫_f(R) g(x,y) dx dy = ∫∫_R g(r cos θ, r sin θ) r dr dθ
///
/// ## Spherical Coordinates (3D)
///
/// f(ρ,θ,φ) = (ρ sin φ cos θ, ρ sin φ sin θ, ρ cos φ)
/// |det Df| = ρ² sin φ
///
/// ∫∫∫_f(R) g(x,y,z) dx dy dz = ∫∫∫_R g(...) ρ² sin φ dρ dθ dφ
pub fn change_of_variables<S1, S2, Fn>(
    _function: &Fn,
    _g: &dyn core::ops::Fn(&S2::Set) -> Result<f64>,
    _set: &S1::Set,
) -> Result<f64>
where
    S1: SigmaAlgebra,
    S2: SigmaAlgebra,
    Fn: MeasurableFunction<S1, S2>,
{
    Err(MeasureError::computation(
        "Change of variables integration not yet implemented",
    ))
}

/// Compute the pushforward of a measure
///
/// Convenience function for creating a pushforward measure.
///
/// # Arguments
///
/// * `measure` - The measure to push forward
/// * `target_sigma` - The target σ-algebra
///
/// # Returns
///
/// A pushforward measure f₊μ
///
/// # Examples
///
/// use amari_measure::{pushforward, LebesgueMeasure, LebesgueSigma};
///
/// let mu = LebesgueMeasure::new(1);
/// let target = LebesgueSigma::new(1);
/// let f_push_mu = pushforward(mu, target);
pub fn pushforward<S1, S2, M, F, S, C>(
    measure: M,
    target_sigma: S2,
) -> Pushforward<S1, S2, M, F, S, C>
where
    S1: SigmaAlgebra,
    S2: SigmaAlgebra,
    M: Measure<S1, F, S, C>,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    Pushforward::new(measure, target_sigma)
}

/// Compute the pullback of a measure
///
/// Convenience function for creating a pullback measure.
///
/// # Arguments
///
/// * `measure` - The measure to pull back
/// * `source_sigma` - The source σ-algebra
///
/// # Returns
///
/// A pullback measure f*ν
///
/// # Examples
///
/// use amari_measure::{pullback, LebesgueMeasure, LebesgueSigma};
///
/// let nu = LebesgueMeasure::new(1);
/// let source = LebesgueSigma::new(1);
/// let f_pull_nu = pullback(nu, source);
pub fn pullback<S1, S2, M, F, S, C>(measure: M, source_sigma: S1) -> Pullback<S1, S2, M, F, S, C>
where
    S1: SigmaAlgebra,
    S2: SigmaAlgebra,
    M: Measure<S2, F, S, C>,
    F: MeasureProperty + 'static,
    S: MeasureProperty + 'static,
    C: MeasureProperty + 'static,
{
    Pullback::new(measure, source_sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sigma_algebra::LebesgueSigma;

    #[test]
    fn test_measurable_function_trait_exists() {
        // Verify the trait compiles
        fn _check_trait<S1: SigmaAlgebra, S2: SigmaAlgebra, F: MeasurableFunction<S1, S2>>() {}
    }

    #[test]
    fn test_pushforward_type_exists() {
        // Verify the type compiles with different sigma algebras
        type _PushTest<S1, S2, M> = Pushforward<S1, S2, M, SigmaFinite, Unsigned, Complete>;
    }

    #[test]
    fn test_pullback_type_exists() {
        // Verify the type compiles with different sigma algebras
        type _PullTest<S1, S2, M> = Pullback<S1, S2, M, SigmaFinite, Unsigned, Complete>;
    }

    #[test]
    fn test_pushforward_with_lebesgue_sigma() {
        let _source_sigma = LebesgueSigma::new(1);
        let _target_sigma = LebesgueSigma::new(2);

        // Structure is valid - full implementation requires Measure trait completion
    }

    #[test]
    fn test_pullback_with_lebesgue_sigma() {
        let _source_sigma = LebesgueSigma::new(1);
        let _target_sigma = LebesgueSigma::new(2);

        // Structure is valid - full implementation requires Measure trait completion
    }
}
