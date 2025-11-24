# Amari-Calculus v0.11.0 Design Document

## Overview

**amari-calculus** provides geometric calculus - a unified framework for differential and integral calculus using geometric algebra. It extends vector calculus, differential forms, and tensor calculus into a coherent geometric framework.

**Version:** 0.11.0
**Status:** In Development
**Dependencies:** amari-core, amari-measure

## Mathematical Foundation

### Geometric Calculus

Geometric calculus unifies differential and integral calculus through the **vector derivative operator**:

```
∇ = e^i ∂_i  (sum over basis vectors)
```

This operator combines:
- **Dot product** → divergence (∇·F)
- **Wedge product** → curl (∇∧F)
- **Full geometric product** → complete derivative (∇F = ∇·F + ∇∧F)

### Key Theorems

1. **Fundamental Theorem of Geometric Calculus:**
   ```
   ∫_V (∇F) dV = ∮_∂V F dS
   ```
   Unifies divergence theorem, Stokes' theorem, and fundamental theorem of calculus.

2. **Geometric Derivative:**
   ```
   ∇f = ∂f/∂x^i e_i  (gradient for scalars)
   ∇F = ∇·F + ∇∧F    (full derivative for multivectors)
   ```

3. **Directional Derivative:**
   ```
   D_a F = (a·∇)F  (derivative in direction a)
   ```

4. **Covariant Derivative** (on manifolds):
   ```
   ∇_X Y = X^i (∂Y^j/∂x^i + Γ^j_{ik}Y^k) e_j
   ```

5. **Lie Derivative:**
   ```
   L_X Y = [X, Y]  (Lie bracket for vector fields)
   ```

## Architecture

### Module Structure

```
amari-calculus/
├── src/
│   ├── lib.rs                  # Public API, prelude, re-exports
│   ├── derivative.rs           # Vector derivative operator (∇)
│   ├── geometric_derivative.rs # Geometric derivative (∇F decomposition)
│   ├── directional.rs          # Directional derivatives
│   ├── covariant.rs            # Covariant derivatives on manifolds
│   ├── lie.rs                  # Lie derivatives and Lie bracket
│   ├── integration.rs          # Integration on manifolds (uses amari-measure)
│   ├── differential_forms.rs   # Bridge to differential forms notation
│   ├── operators/              # Common differential operators
│   │   ├── mod.rs
│   │   ├── gradient.rs         # ∇f (scalar → vector)
│   │   ├── divergence.rs       # ∇·F (vector → scalar)
│   │   ├── curl.rs             # ∇∧F (vector → bivector)
│   │   ├── laplacian.rs        # ∇²f (scalar → scalar)
│   │   └── dalembertian.rs     # □ = ∂²/∂t² - ∇² (wave operator)
│   ├── fields/                 # Vector and multivector fields
│   │   ├── mod.rs
│   │   ├── scalar_field.rs     # Scalar fields f: M → ℝ
│   │   ├── vector_field.rs     # Vector fields F: M → V
│   │   └── multivector_field.rs # General multivector fields
│   ├── manifold/               # Manifold calculus
│   │   ├── mod.rs
│   │   ├── connection.rs       # Connection and Christoffel symbols
│   │   ├── metric.rs           # Riemannian/pseudo-Riemannian metrics
│   │   ├── geodesic.rs         # Geodesic equations
│   │   └── curvature.rs        # Curvature tensors (future)
│   └── examples/               # Classical applications
│       ├── maxwell.rs          # Maxwell's equations
│       ├── navier_stokes.rs    # Fluid dynamics (basic)
│       └── heat_equation.rs    # Diffusion equations
├── tests/
│   ├── derivative_tests.rs     # Vector derivative tests
│   ├── operator_tests.rs       # grad, div, curl tests
│   ├── manifold_tests.rs       # Covariant derivative tests
│   └── integration_tests.rs    # Fundamental theorem tests
└── benches/
    └── calculus_bench.rs       # Performance benchmarks
```

## Core Types

### 1. Vector Derivative Operator

```rust
/// Vector derivative operator ∇ = e^i ∂_i
pub struct VectorDerivative<const P: usize, const Q: usize, const R: usize> {
    /// Coordinate system (Cartesian, spherical, cylindrical, etc.)
    coordinates: CoordinateSystem,
    /// Metric tensor (for covariant derivatives)
    metric: Option<MetricTensor<P, Q, R>>,
}

impl<const P: usize, const Q: usize, const R: usize> VectorDerivative<P, Q, R> {
    /// Apply to scalar field: ∇f (gradient)
    pub fn gradient(&self, f: &ScalarField<P, Q, R>) -> VectorField<P, Q, R>;

    /// Apply to vector field: ∇·F (divergence)
    pub fn divergence(&self, f: &VectorField<P, Q, R>) -> ScalarField<P, Q, R>;

    /// Apply to vector field: ∇∧F (curl)
    pub fn curl(&self, f: &VectorField<P, Q, R>) -> BivectorField<P, Q, R>;

    /// Full geometric derivative: ∇F = ∇·F + ∇∧F
    pub fn apply(&self, f: &MultivectorField<P, Q, R>) -> MultivectorField<P, Q, R>;
}
```

### 2. Scalar and Vector Fields

```rust
/// Scalar field f: ℝⁿ → ℝ
pub struct ScalarField<const P: usize, const Q: usize, const R: usize> {
    /// Function defining the field
    function: Box<dyn Fn(&[f64]) -> f64>,
}

/// Vector field F: ℝⁿ → Cl(p,q,r)_1 (grade-1 multivectors)
pub struct VectorField<const P: usize, const Q: usize, const R: usize> {
    /// Function defining the field
    function: Box<dyn Fn(&[f64]) -> Multivector<P, Q, R>>,
}

/// General multivector field F: ℝⁿ → Cl(p,q,r)
pub struct MultivectorField<const P: usize, const Q: usize, const R: usize> {
    /// Function defining the field
    function: Box<dyn Fn(&[f64]) -> Multivector<P, Q, R>>,
    /// Domain dimension
    dim: usize,
}
```

### 3. Covariant Derivative

```rust
/// Covariant derivative ∇_X on manifolds
pub struct CovariantDerivative<const P: usize, const Q: usize, const R: usize> {
    /// Connection (Christoffel symbols)
    connection: Connection<P, Q, R>,
    /// Metric tensor
    metric: MetricTensor<P, Q, R>,
}

impl<const P: usize, const Q: usize, const R: usize> CovariantDerivative<P, Q, R> {
    /// Covariant derivative of vector field Y in direction X
    pub fn apply(
        &self,
        x: &VectorField<P, Q, R>,
        y: &VectorField<P, Q, R>,
    ) -> VectorField<P, Q, R>;

    /// Parallel transport along curve
    pub fn parallel_transport(
        &self,
        v: &Multivector<P, Q, R>,
        curve: &Curve<P, Q, R>,
    ) -> Multivector<P, Q, R>;
}
```

### 4. Lie Derivative

```rust
/// Lie derivative L_X (measures how tensor field changes along flow)
pub struct LieDerivative<const P: usize, const Q: usize, const R: usize> {
    /// Vector field generating the flow
    flow: VectorField<P, Q, R>,
}

impl<const P: usize, const Q: usize, const R: usize> LieDerivative<P, Q, R> {
    /// Lie bracket [X, Y] of vector fields
    pub fn lie_bracket(
        &self,
        x: &VectorField<P, Q, R>,
        y: &VectorField<P, Q, R>,
    ) -> VectorField<P, Q, R>;

    /// Lie derivative of scalar field
    pub fn scalar(&self, f: &ScalarField<P, Q, R>) -> ScalarField<P, Q, R>;

    /// Lie derivative of vector field
    pub fn vector(&self, f: &VectorField<P, Q, R>) -> VectorField<P, Q, R>;
}
```

### 5. Integration on Manifolds

```rust
/// Integration of multivector fields over manifolds
pub struct ManifoldIntegrator<const P: usize, const Q: usize, const R: usize> {
    /// Measure from amari-measure
    measure: GeometricMeasure<P, Q, R>,
    /// Manifold parameterization
    manifold: Manifold<P, Q, R>,
}

impl<const P: usize, const Q: usize, const R: usize> ManifoldIntegrator<P, Q, R> {
    /// Integrate scalar field over manifold
    pub fn integrate_scalar(
        &self,
        f: &ScalarField<P, Q, R>,
    ) -> Result<f64, IntegrationError>;

    /// Integrate multivector field over manifold
    pub fn integrate_multivector(
        &self,
        f: &MultivectorField<P, Q, R>,
    ) -> Result<Multivector<P, Q, R>, IntegrationError>;

    /// Verify fundamental theorem: ∫_V (∇F) dV = ∮_∂V F dS
    pub fn verify_fundamental_theorem(
        &self,
        f: &MultivectorField<P, Q, R>,
    ) -> Result<bool, IntegrationError>;
}
```

## Implementation Strategy

### Phase 1: Core Operators (Week 1-2)

**Goal:** Basic differential operators working in Euclidean space

1. **VectorDerivative** basic implementation
   - Cartesian coordinates only
   - Numerical differentiation (finite differences)
   - Gradient, divergence, curl

2. **ScalarField and VectorField**
   - Function wrapper types
   - Evaluation at points
   - Composition operations

3. **Tests:**
   - Gradient of scalar fields
   - Divergence and curl of vector fields
   - Product rules (∇(fg), ∇(f·g))
   - Identity verification (∇∧∇f = 0, ∇·(∇∧F) = 0)

### Phase 2: Geometric Integration (Week 2-3)

**Goal:** Connect to amari-measure for rigorous integration

1. **ManifoldIntegrator** implementation
   - Use LebesgueIntegrator from amari-measure
   - Volume integrals
   - Surface integrals

2. **Fundamental Theorem verification**
   - Implement boundary extraction
   - Verify ∫_V (∇F) dV = ∮_∂V F dS

3. **Tests:**
   - Volume integrals of divergence
   - Surface integrals over boundaries
   - Fundamental theorem on simple domains (cube, sphere)

### Phase 3: Manifold Calculus (Week 3-4)

**Goal:** Covariant derivatives on curved spaces

1. **Connection** and **MetricTensor**
   - Christoffel symbol computation
   - Levi-Civita connection (torsion-free, metric-compatible)

2. **CovariantDerivative**
   - Covariant derivative of vector fields
   - Parallel transport

3. **Tests:**
   - Covariant derivative on sphere
   - Parallel transport around closed loops
   - Compatibility with metric (∇_X g = 0)

### Phase 4: Advanced Features (Week 4-5)

**Goal:** Lie derivatives and classical applications

1. **LieDerivative**
   - Lie bracket implementation
   - Lie derivative of fields

2. **Classical equations:**
   - Maxwell's equations in GA form
   - Heat equation
   - Basic fluid dynamics

3. **Tests:**
   - Jacobi identity for Lie bracket
   - Maxwell equation solutions
   - Heat equation numerical solutions

### Phase 5: Documentation and Examples (Week 5-6)

1. **Comprehensive documentation**
   - Mathematical background
   - API documentation
   - Tutorials

2. **Example applications**
   - Electromagnetism (Maxwell)
   - Fluid flow (Navier-Stokes basics)
   - Heat diffusion
   - Geodesics on surfaces

## Integration with Existing Crates

### amari-core
- Uses `Multivector<P, Q, R>` as fundamental type
- Geometric product for derivative operations
- Basis blade operations

### amari-measure
- Uses `GeometricMeasure` for integration
- Uses `LebesgueIntegrator` for volume integrals
- Provides rigorous measure-theoretic foundation

### amari-info-geom
- Provides Fisher metric for statistical manifolds
- Uses covariant derivatives for natural gradients
- Geodesics for optimization

### amari-relativistic
- Uses Minkowski metric (signature (1,3))
- Covariant derivatives for spacetime
- Will integrate with existing geodesic code

## Performance Considerations

### Numerical Differentiation
- Use Richardson extrapolation for improved accuracy
- Adaptive step sizes
- Cache function evaluations

### Finite Element Methods (Future)
- Basis functions for field representation
- Sparse matrix assembly
- Iterative solvers

### GPU Acceleration (Future)
- Batch evaluation of fields
- Parallel integration using amari-gpu
- WASM compatibility

## Testing Strategy

### Unit Tests
1. **Operator identities:**
   - ∇∧(∇f) = 0 (curl of gradient is zero)
   - ∇·(∇∧F) = 0 (divergence of curl is zero)
   - ∇²f = ∇·(∇f) (Laplacian definition)

2. **Product rules:**
   - ∇(fg) = f∇g + g∇f
   - ∇·(fF) = (∇f)·F + f∇·F
   - ∇∧(fF) = (∇f)∧F + f∇∧F

3. **Coordinate transformations:**
   - Spherical coordinates
   - Cylindrical coordinates
   - Polar coordinates (2D)

### Integration Tests
1. **Fundamental theorem verification**
2. **Maxwell's equations**
3. **Heat equation solutions**
4. **Geodesic computation**

### Property-Based Tests
1. **Linearity:** ∇(af + bg) = a∇f + b∇g
2. **Leibniz rule:** ∇(fg) satisfies product rule
3. **Commutation:** [∇_X, ∇_Y] - ∇_{[X,Y]} = R(X,Y) (curvature)

## Success Criteria

### Functionality
- ✅ All basic operators (grad, div, curl, Laplacian) working
- ✅ Fundamental theorem verified on test cases
- ✅ Covariant derivatives on curved manifolds
- ✅ Lie derivatives and bracket computation
- ✅ Integration with amari-measure

### Quality
- ✅ 90%+ test coverage
- ✅ All property tests passing
- ✅ Documentation complete with examples
- ✅ Benchmarks showing reasonable performance

### Integration
- ✅ Works seamlessly with amari-core
- ✅ Uses amari-measure for integration
- ✅ Ready for amari-info-geom geodesics
- ✅ Compatible with amari-relativistic

## Future Enhancements (Post v0.11.0)

1. **Automatic Differentiation:** Exact symbolic derivatives
2. **Finite Element Methods:** Weak solutions to PDEs
3. **Spectral Methods:** Fourier-based PDE solvers
4. **GPU Acceleration:** Parallel field evaluation
5. **Curvature Tensors:** Riemann, Ricci, Weyl
6. **Clifford-Fourier Transform:** Frequency domain analysis

## References

1. **Geometric Calculus:**
   - Hestenes, D. (1984). "Clifford Algebra to Geometric Calculus"
   - Doran, C., & Lasenby, A. (2003). "Geometric Algebra for Physicists"

2. **Differential Geometry:**
   - Lee, J. M. (2018). "Introduction to Riemannian Manifolds"
   - do Carmo, M. P. (1992). "Riemannian Geometry"

3. **Clifford Analysis:**
   - Brackx, F., Delanghe, R., & Sommen, F. (1982). "Clifford Analysis"
   - Gürlebeck, K., & Sprößig, W. (1997). "Quaternionic and Clifford Calculus"

## Timeline

- **Week 1-2:** Core operators and fields
- **Week 2-3:** Integration and fundamental theorem
- **Week 3-4:** Manifold calculus and covariant derivatives
- **Week 4-5:** Lie derivatives and classical applications
- **Week 5-6:** Documentation, examples, and polish
- **Week 6:** Release v0.11.0
