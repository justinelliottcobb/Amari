# Amari Project Context for Claude Code

This document consolidates all project context, guidelines, and technical documentation for AI assistants working on the Amari mathematical computing library.

## Project Overview

Amari is a high-performance mathematical computing library focusing on:
- Geometric Algebra (Clifford Algebra)
- Tropical Algebra (max-plus algebra)
- Dual Numbers and Automatic Differentiation
- Information Geometry
- Enumerative Geometry
- Fusion Systems
- Cellular Automata

## Core Design Principles

### 1. Mathematical Rigor
- All operations must preserve mathematical properties
- Formal verification through property-based testing
- Compile-time type safety via phantom types
- No silent failures or undefined behavior

### 2. Performance
- Zero-cost abstractions where possible
- SIMD optimization for vectorized operations
- GPU acceleration via WebGPU
- Cache-friendly memory layouts

### 3. Type Safety
- Phantom types encode mathematical constraints
- Compile-time dimension checking
- Type-level encoding of algebraic structures
- Safe cross-language bindings

## API Design Conventions

### Naming Convention
- **Geometric operations**: `geometric_product`, `inner_product`, `outer_product`
- **Algebraic operations**: `add`, `mul`, `inverse`
- **Transformations**: `apply_rotor`, `reflect`, `project`
- **Constructors**: `from_*`, `new`, `zero`, `identity`
- **Conversions**: `to_*`, `into_*`, `as_*`

### Method Organization
```rust
// 1. Constructors
impl<const N: usize, const S: usize, const P: usize> Multivector<N, S, P> {
    pub fn new(components: [f64; 1 << N]) -> Self { ... }
    pub fn zero() -> Self { ... }
}

// 2. Core operations
impl<const N: usize, const S: usize, const P: usize> Multivector<N, S, P> {
    pub fn geometric_product(&self, other: &Self) -> Self { ... }
    pub fn grade_projection<const K: usize>(&self) -> Blade<N, K, S, P> { ... }
}

// 3. Trait implementations
impl<const N: usize, const S: usize, const P: usize> Add for Multivector<N, S, P> { ... }
```

## Error Handling Strategy

### Principles
1. **Compile-time over runtime**: Catch errors at compile time when possible
2. **Explicit over implicit**: No silent failures
3. **Recoverable errors**: Use `Result<T, E>` for operations that can fail
4. **Type safety**: Use phantom types to prevent invalid operations

### Error Types
```rust
#[derive(Debug, thiserror::Error)]
pub enum AlgebraError {
    #[error("Division by zero")]
    DivisionByZero,

    #[error("Non-invertible element")]
    NonInvertible,

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}
```

## Phantom Types Methodology

### Type Parameters
- `N`: Number of basis vectors (dimension)
- `S`: Positive signature dimensions
- `P`: Negative signature dimensions
- `Q`: Zero signature dimensions (degenerate)

### Metric Signatures
```rust
// Euclidean 3D: (3, 0, 0)
type Euclidean3D = Multivector<3, 3, 0>;

// Minkowski spacetime: (3, 1, 0)
type Spacetime = Multivector<4, 3, 1>;

// Projective geometry: (3, 0, 1)
type Projective3D = Multivector<4, 3, 0>;
```

## Testing Strategy

### Test Categories
1. **Unit tests**: Core mathematical operations
2. **Property tests**: Mathematical invariants
3. **Integration tests**: Cross-crate interactions
4. **Formal verification**: Contract-based testing
5. **Performance benchmarks**: Regression prevention

### Test Enforcement
```rust
#[test]
fn test_geometric_product_associativity() {
    let a = Multivector::random();
    let b = Multivector::random();
    let c = Multivector::random();

    assert_eq!(
        (a.geometric_product(&b)).geometric_product(&c),
        a.geometric_product(&b.geometric_product(&c))
    );
}
```

## CI/CD Pipeline

### Workflow Structure
1. **Validation**: Format, lint, test
2. **Build**: All features, all targets
3. **Publish**: crates.io and npm
4. **Release**: GitHub releases

### Quality Gates
- All tests must pass
- No clippy warnings
- Code formatted with rustfmt
- Documentation builds without warnings
- Formal verification tests pass

## Publishing Process

### Version Management
Use `scripts/bump-version.sh` to update versions:
```bash
./scripts/bump-version.sh 0.6.2
```

### Publishing Order
1. Core crates (amari-core, etc.)
2. Dependent crates (amari-fusion, etc.)
3. Main crate (amari)
4. WASM/npm packages

### Release Workflow
```bash
# Automated release with version bump
gh workflow run publish.yml --field version=0.6.2
```

## Development Phases

### Completed Phases
- **Phase 1**: Core geometric algebra
- **Phase 2**: Advanced algebras (tropical, dual)
- **Phase 3**: Cross-language bindings
- **Phase 4**: Formal verification
- **Phase 5**: Performance optimization
- **Phase 6**: Example suite expansion

### Current Focus
- Documentation consolidation
- npm package publishing
- GPU acceleration improvements

## Code Quality Standards

### Documentation
- All public APIs must be documented
- Include mathematical formulas where relevant
- Provide usage examples
- Link to relevant papers/resources

### Performance
- Benchmark critical operations
- Profile memory usage
- Optimize hot paths
- Use SIMD where beneficial

### Safety
- No unsafe code without justification
- Proper error handling
- Memory safety guaranteed
- Thread safety where applicable

## Project Structure

```
amari/
├── amari-core/          # Core geometric algebra
├── amari-tropical/      # Tropical algebra
├── amari-dual/          # Dual numbers
├── amari-info-geom/     # Information geometry
├── amari-enumerative/   # Enumerative geometry
├── amari-fusion/        # Fusion systems
├── amari-automata/      # Cellular automata
├── amari-gpu/           # GPU acceleration
├── amari-wasm/          # WebAssembly bindings
├── examples/            # Usage examples
├── scripts/             # Build and release scripts
└── docs/               # Documentation
    ├── claude-code/    # AI assistant context
    └── technical/      # Technical specifications
```

## Guidelines for AI Assistants

### Do's
- Maintain mathematical correctness
- Use existing patterns and conventions
- Write comprehensive tests
- Document complex algorithms
- Consider performance implications

### Don'ts
- Don't break mathematical invariants
- Don't use unsafe without justification
- Don't skip error handling
- Don't ignore compiler warnings
- Don't duplicate existing functionality

## Quick References

### Common Commands
```bash
# Run all tests
cargo test --workspace --all-features

# Check formatting
cargo fmt --all -- --check

# Run clippy
cargo clippy --workspace --all-features -- -D warnings

# Build documentation
cargo doc --workspace --no-deps

# Bump version
./scripts/bump-version.sh X.Y.Z

# Publish
gh workflow run publish.yml --field version=X.Y.Z
```

### Key Files
- `Cargo.toml`: Workspace configuration
- `rust-toolchain.toml`: Rust version specification
- `.github/workflows/`: CI/CD pipelines
- `scripts/bump-version.sh`: Version management
- `CHANGELOG.md`: Release notes

## Contact and Resources

- Repository: https://github.com/justinelliottcobb/Amari
- Documentation: https://docs.rs/amari
- Crates.io: https://crates.io/crates/amari
- Issues: https://github.com/justinelliottcobb/Amari/issues