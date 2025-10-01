# Amari Error Handling Design v0.3.0

## Overview

This document outlines the unified error handling architecture for the Amari mathematical computing library, establishing consistent error patterns across all crates.

## Design Principles

1. **Consistency**: All crates use the same error handling patterns
2. **Clarity**: Error messages clearly describe what went wrong
3. **Context**: Errors preserve context for debugging
4. **Composability**: Errors can be composed across crate boundaries
5. **Performance**: Zero-cost abstractions where possible

## Error Categories

### 1. Validation Errors
Errors that occur when input validation fails:
- `InvalidDimension`: Mismatched or invalid dimensions
- `InvalidCoordinates`: Out of bounds coordinates
- `ParameterOutOfRange`: Parameter values outside acceptable range
- `InvalidConfiguration`: Invalid configuration or setup

### 2. Computation Errors
Errors during mathematical computations:
- `NumericalInstability`: Numerical precision or stability issues
- `SolverConvergenceFailure`: Iterative solver failed to converge
- `ComputationError`: General computation failure
- `DivisionByZero`: Division by zero encountered
- `Overflow`: Numerical overflow detected
- `Underflow`: Numerical underflow detected

### 3. Resource Errors
Errors related to system resources:
- `InitializationError`: Failed to initialize resource
- `BufferError`: Buffer allocation or access error
- `ShaderError`: GPU shader compilation/execution error
- `OutOfMemory`: Insufficient memory available

### 4. Algorithmic Errors
Errors in algorithm execution:
- `MaxIterationsExceeded`: Algorithm exceeded iteration limit
- `ConstraintViolation`: Algorithm constraint violated
- `NotImplemented`: Feature not yet implemented
- `UnsupportedOperation`: Operation not supported for given type

## Implementation Strategy

### Phase 1: Infrastructure (v0.3.0-alpha)
1. Add `thiserror` dependency to workspace
2. Create base error types in each crate
3. Establish error conversion patterns

### Phase 2: Migration (v0.3.0-beta)
1. Convert manual error implementations to `thiserror`
2. Replace panics with Result types where appropriate
3. Add error context and chaining

### Phase 3: Integration (v0.3.0)
1. Create unified `AmariError` in root crate
2. Implement cross-crate error conversion
3. Add comprehensive error documentation

## Error Type Hierarchy

```rust
// Root error type in amari crate
#[derive(Error, Debug)]
pub enum AmariError {
    #[error(transparent)]
    Core(#[from] amari_core::CoreError),

    #[error(transparent)]
    Automata(#[from] amari_automata::AutomataError),

    #[error(transparent)]
    Enumerative(#[from] amari_enumerative::EnumerativeError),

    #[error(transparent)]
    Gpu(#[from] amari_gpu::GpuError),

    #[error(transparent)]
    InfoGeom(#[from] amari_info_geom::InfoGeomError),

    #[error(transparent)]
    Fusion(#[from] amari_fusion::FusionError),

    #[error(transparent)]
    Tropical(#[from] amari_tropical::TropicalError),

    #[error(transparent)]
    Dual(#[from] amari_dual::DualError),
}

pub type AmariResult<T> = Result<T, AmariError>;
```

## Crate-Specific Error Types

### amari-core
```rust
#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Numerical instability detected")]
    NumericalInstability,
}
```

### amari-automata
```rust
#[derive(Error, Debug)]
pub enum AutomataError {
    #[error("Invalid coordinates: ({0}, {1})")]
    InvalidCoordinates(usize, usize),

    #[error("Configuration not found during inverse design")]
    ConfigurationNotFound,

    #[error("Assembly constraint violation: {0}")]
    AssemblyConstraintViolation(String),
}
```

### amari-enumerative
```rust
#[derive(Error, Debug)]
pub enum EnumerativeError {
    #[error("Invalid dimension for {context}: {message}")]
    InvalidDimension { context: String, message: String },

    #[error("Intersection computation failed: {0}")]
    IntersectionError(String),

    #[error("Schubert calculus error: {0}")]
    SchubertError(String),
}
```

## Migration Guidelines

### Converting Panics to Results

**Before:**
```rust
pub fn divide(a: f64, b: f64) -> f64 {
    assert!(b != 0.0, "Division by zero");
    a / b
}
```

**After:**
```rust
pub fn divide(a: f64, b: f64) -> CoreResult<f64> {
    if b == 0.0 {
        return Err(CoreError::DivisionByZero);
    }
    Ok(a / b)
}
```

### Adding Error Context

```rust
use thiserror::Error;

operation()
    .map_err(|e| CoreError::ComputationError {
        context: "matrix inversion".to_string(),
        source: Box::new(e),
    })?;
```

## Testing Strategy

1. **Unit Tests**: Test each error condition
2. **Integration Tests**: Test error propagation across crates
3. **Property Tests**: Ensure error handling preserves invariants
4. **Documentation Tests**: Ensure examples handle errors correctly

## Backward Compatibility

To maintain backward compatibility:
1. Keep panic-based APIs as deprecated wrappers
2. Add new Result-based APIs alongside existing ones
3. Document migration path in CHANGELOG
4. Provide conversion utilities where needed

## Performance Considerations

- Use `#[inline]` for small error creation functions
- Avoid allocations in hot paths
- Use `&'static str` for error messages where possible
- Consider using error codes for frequently occurring errors

## Documentation Standards

Each error variant should document:
1. When the error occurs
2. Common causes
3. How to handle or prevent it
4. Example code demonstrating the error

## Version Plan

- **v0.3.0-alpha**: Basic error infrastructure
- **v0.3.0-beta**: Complete migration of existing errors
- **v0.3.0**: Full unified error handling with documentation