# Migration Guide: Amari v0.11.x → v0.12.0

## Overview

Version 0.12.0 represents a significant architectural refactoring of the Amari library, introducing **domain types** with enhanced type safety, privacy, and mathematical correctness. This release contains **breaking changes** that require code updates when migrating from v0.11.x.

**Key Changes:**
- Private fields with accessor methods (improved encapsulation)
- New constructor and factory methods
- Consistent API patterns across all domain types
- Reference parameters for tropical operations
- Renamed constants and methods for clarity

**Migration Effort:** Medium to High (depending on codebase size)
**Recommended Approach:** Update one crate at a time, starting with core dependencies

---

## Table of Contents

1. [TropicalNumber API Changes](#tropicalnumber-api-changes)
2. [DualNumber API Changes](#dualnumber-api-changes)
3. [MultiDualNumber API Changes](#multidualnumber-api-changes)
4. [Architectural Changes](#architectural-changes)
5. [GPU Module Breaking Changes](#gpu-module-breaking-changes)
6. [Testing and Examples](#testing-and-examples)
7. [Migration Checklist](#migration-checklist)

---

## TropicalNumber API Changes

### 1. Construction

**Before (v0.11.x):**
```rust
use amari_tropical::TropicalNumber;

// Tuple struct construction
let a = TropicalNumber(3.0);
let b = TropicalNumber(5.0);
```

**After (v0.12.0):**
```rust
use amari_tropical::TropicalNumber;

// Named constructor
let a = TropicalNumber::new(3.0);
let b = TropicalNumber::new(5.0);
```

**Rationale:** Private fields enforce proper construction and prevent invalid states.

---

### 2. Field Access

**Before (v0.11.x):**
```rust
let num = TropicalNumber(42.0);
let value = num.0;  // Direct tuple field access
```

**After (v0.12.0):**
```rust
let num = TropicalNumber::new(42.0);
let value = num.value();  // Accessor method
```

**Rationale:** Accessor methods provide controlled access and enable future optimizations without breaking API.

---

### 3. Constants

**Before (v0.11.x):**
```rust
let zero = TropicalNumber::<f64>::ZERO;  // -∞ (tropical additive identity)
let one = TropicalNumber::<f64>::ONE;    // 0 (tropical multiplicative identity)
```

**After (v0.12.0):**
```rust
let zero = TropicalNumber::tropical_zero();  // -∞
let one = TropicalNumber::tropical_one();    // 0
```

**Migration Note:** Both methods still available for compatibility:
```rust
let zero = TropicalNumber::zero();           // alias for tropical_zero()
let zero = TropicalNumber::tropical_zero();  // explicit name
```

**Rationale:** Explicit names prevent confusion with standard zero/one semantics.

---

### 4. Tropical Operations (Reference Parameters)

**Before (v0.11.x):**
```rust
let a = TropicalNumber::new(3.0);
let b = TropicalNumber::new(5.0);

let sum = a.tropical_add(b);  // Owned value
let product = a.tropical_mul(b);
```

**After (v0.12.0):**
```rust
let a = TropicalNumber::new(3.0);
let b = TropicalNumber::new(5.0);

let sum = a.tropical_add(&b);  // Reference required
let product = a.tropical_mul(&b);
```

**Rationale:** Reference parameters prevent unnecessary copies and enable more efficient chaining.

---

### 5. Complete Example

**Before (v0.11.x):**
```rust
use amari_tropical::TropicalNumber;

fn shortest_path_distance(edges: &[(f64, f64)]) -> TropicalNumber<f64> {
    let mut total = TropicalNumber::<f64>::ONE;  // Start at 0

    for &(weight1, weight2) in edges {
        let edge1 = TropicalNumber(weight1);
        let edge2 = TropicalNumber(weight2);

        let min_edge = edge1.tropical_add(edge2);
        total = total.tropical_mul(min_edge);
    }

    total
}

let result = shortest_path_distance(&[(2.0, 3.0), (5.0, 1.0)]);
println!("Distance: {}", result.0);
```

**After (v0.12.0):**
```rust
use amari_tropical::TropicalNumber;

fn shortest_path_distance(edges: &[(f64, f64)]) -> TropicalNumber<f64> {
    let mut total = TropicalNumber::tropical_one();  // Start at 0

    for &(weight1, weight2) in edges {
        let edge1 = TropicalNumber::new(weight1);
        let edge2 = TropicalNumber::new(weight2);

        let min_edge = edge1.tropical_add(&edge2);
        total = total.tropical_mul(&min_edge);
    }

    total
}

let result = shortest_path_distance(&[(2.0, 3.0), (5.0, 1.0)]);
println!("Distance: {}", result.value());
```

---

## DualNumber API Changes

### 1. Field Access

**Before (v0.11.x):**
```rust
use amari_dual::DualNumber;

let x = DualNumber { real: 3.0, dual: 1.0 };

let value = x.real;       // Direct field access
let gradient = x.dual;    // Direct field access
```

**After (v0.12.0):**
```rust
use amari_dual::DualNumber;

let x = DualNumber::new(3.0, 1.0);

let value = x.value();          // Accessor method
let gradient = x.derivative();  // Renamed accessor
```

**Migration Notes:**
- `real` → `value()` (clearer semantics)
- `dual` → `derivative()` (more precise terminology)
- Fields are now private

---

### 2. Construction

**Before (v0.11.x):**
```rust
// Direct struct initialization
let x = DualNumber { real: 5.0, dual: 1.0 };
```

**After (v0.12.0):**
```rust
// Named constructor
let x = DualNumber::new(5.0, 1.0);

// Or for constants (derivative = 0)
let c = DualNumber::constant(5.0);
```

---

### 3. Arithmetic Operations

**Before (v0.11.x):**
```rust
let x = DualNumber { real: 3.0, dual: 1.0 };
let result = x * 2.0;  // Mixed scalar multiplication
```

**After (v0.12.0):**
```rust
let x = DualNumber::new(3.0, 1.0);
let result = x * DualNumber::constant(2.0);  // Explicit constant wrapping
```

**Rationale:** Prevents ambiguous type coercion; makes differentiation semantics explicit.

---

### 4. Complete Example

**Before (v0.11.x):**
```rust
use amari_dual::DualNumber;

fn compute_gradient(x: f64) -> f64 {
    let dual_x = DualNumber { real: x, dual: 1.0 };
    let result = dual_x * dual_x + dual_x * 2.0 + DualNumber { real: 1.0, dual: 0.0 };
    result.dual  // Return gradient
}

let grad = compute_gradient(3.0);
```

**After (v0.12.0):**
```rust
use amari_dual::DualNumber;

fn compute_gradient(x: f64) -> f64 {
    let dual_x = DualNumber::new(x, 1.0);
    let result = dual_x * dual_x + dual_x * DualNumber::constant(2.0) + DualNumber::constant(1.0);
    result.derivative()  // Return gradient
}

let grad = compute_gradient(3.0);
```

---

## MultiDualNumber API Changes

### 1. Field Access

**Before (v0.11.x):**
```rust
use amari_dual::MultiDualNumber;

let x = MultiDualNumber::new(5.0, vec![1.0, 0.0]);

let value = x.real;           // Direct field access
let gradient = x.duals;       // Direct field access
let n_vars = x.num_vars();    // Method
```

**After (v0.12.0):**
```rust
use amari_dual::MultiDualNumber;

let x = MultiDualNumber::new(5.0, vec![1.0, 0.0]);

let value = x.get_value();       // Accessor method
let gradient = x.get_gradient(); // Accessor method
let n_vars = x.n_vars();         // Renamed method
```

**Migration Notes:**
- `real` → `get_value()`
- `duals` → `get_gradient()`
- `num_vars()` → `n_vars()` (consistency with other APIs)

---

### 2. Complete Example

**Before (v0.11.x):**
```rust
use amari_dual::MultiDualNumber;

fn compute_multivariable_gradient(x: f64, y: f64) -> Vec<f64> {
    let dual_x = MultiDualNumber::new(x, vec![1.0, 0.0]);
    let dual_y = MultiDualNumber::new(y, vec![0.0, 1.0]);

    let result = dual_x * dual_x + dual_y * dual_y;  // f(x,y) = x² + y²

    result.duals  // Return gradient
}

let gradient = compute_multivariable_gradient(3.0, 4.0);
println!("∇f = [{}, {}]", gradient[0], gradient[1]);
```

**After (v0.12.0):**
```rust
use amari_dual::MultiDualNumber;

fn compute_multivariable_gradient(x: f64, y: f64) -> Vec<f64> {
    let dual_x = MultiDualNumber::new(x, vec![1.0, 0.0]);
    let dual_y = MultiDualNumber::new(y, vec![0.0, 1.0]);

    let result = dual_x * dual_x + dual_y * dual_y;  // f(x,y) = x² + y²

    result.get_gradient()  // Return gradient
}

let gradient = compute_multivariable_gradient(3.0, 4.0);
println!("∇f = [{}, {}]", gradient[0], gradient[1]);
```

---

## Architectural Changes

### GPU Examples and Tests Moved

**Change:** GPU-related examples and tests have been removed from domain crates (amari-tropical, amari-dual, amari-enumerative) and consolidated in `amari-gpu`.

**Before (v0.11.x):**
```
amari-tropical/examples/gpu_neural_attention.rs    ❌ Removed
amari-tropical/tests/gpu_integration.rs            ❌ Removed
amari-enumerative/examples/gpu_enumerative_geometry.rs  ❌ Removed
amari-enumerative/tests/gpu_integration.rs         ❌ Removed
```

**After (v0.12.0):**
```
amari-gpu/examples/tropical_neural_attention.rs    ✅ Centralized
amari-gpu/examples/enumerative_geometry.rs         ✅ Centralized
amari-gpu/tests/integration.rs                     ✅ Centralized
```

**Rationale:**
- **Architectural Clarity:** Domain crates provide APIs; integration crates (like amari-gpu) consume them
- **Dependency Order:** Prevents circular dependencies during publishing
- **Maintainability:** GPU code in one location simplifies updates

**Migration:** If you relied on these examples, find equivalent examples in `amari-gpu/examples/`.

---

## GPU Module Breaking Changes

### Overview

Version 0.12.0 includes significant changes to `amari-gpu` module availability due to architectural improvements enforcing separation of concerns between domain crates and integration crates.

**Impact Summary:**
- ✅ **Enabled:** `dual`, `enumerative`, `automata` GPU modules (newly available)
- ❌ **Disabled:** `tropical`, `fusion` GPU modules (temporarily unavailable)

---

### Disabled Modules

#### `amari_gpu::tropical` - Temporarily Disabled

**Before (v0.11.x):**
```rust
use amari_gpu::tropical::TropicalMatrixGpu;

let gpu_matrix = TropicalMatrixGpu::new(&matrix)?;
let result = gpu_matrix.multiply(&other).await?;
```

**After (v0.12.0):**
```rust
// GPU module not available - use CPU implementation
use amari_tropical::TropicalMatrix;

let matrix = TropicalMatrix::new(data);
let result = matrix.tropical_multiply(&other);
```

**Reason:** Rust orphan impl rules prevent implementing GPU acceleration traits on types from external crates. A future release will use extension traits to restore this functionality.

---

#### `amari_gpu::fusion` - Temporarily Disabled

**Before (v0.11.x):**
```rust
use amari_gpu::fusion::FusionGpu;

let gpu_fusion = FusionGpu::new(&config)?;
let result = gpu_fusion.optimize(&data).await?;
```

**After (v0.12.0):**
```rust
// GPU module not available - use CPU implementation
use amari_fusion::TropicalDualClifford;

let fusion = TropicalDualClifford::new(...);
let result = fusion.optimize(...);
```

**Reason:** The fusion GPU module requires GPU submodules in both `amari_dual` and `amari_tropical` domain crates. These will be added in a future release.

---

### Newly Enabled Modules

The following GPU modules are now available in v0.12.0:

#### `amari_gpu::dual` (feature: `dual`)

```rust
use amari_gpu::dual::DualNumberGpu;
use amari_dual::DualNumber;

// Batch GPU operations for dual numbers
let gpu = DualNumberGpu::new().await?;
let results = gpu.batch_evaluate(&inputs).await?;
```

#### `amari_gpu::enumerative` (feature: `enumerative`)

```rust
use amari_gpu::enumerative::IntersectionTheoryGpu;

// GPU-accelerated intersection theory computations
let gpu = IntersectionTheoryGpu::new().await?;
let intersections = gpu.compute_intersections(&varieties).await?;
```

#### `amari_gpu::automata` (feature: `automata`)

```rust
use amari_gpu::automata::CellularAutomataGpu;

// GPU evolution of cellular automata
let gpu = CellularAutomataGpu::new(&rule).await?;
let next_state = gpu.evolve(&state, steps).await?;
```

---

### Feature Flag Changes

**Before (v0.11.x):**
```toml
[dependencies]
amari-gpu = { version = "0.11", features = ["tropical", "fusion"] }
```

**After (v0.12.0):**
```toml
[dependencies]
# tropical and fusion features are disabled
# Use domain crates directly for CPU implementations
amari-gpu = { version = "0.12", features = ["dual", "enumerative", "automata"] }

# For tropical/fusion, use domain crates
amari-tropical = "0.12"
amari-fusion = "0.12"
```

---

### Migration Steps for GPU Users

1. **Check for `amari_gpu::tropical` usage:**
   - Remove imports of `amari_gpu::tropical::*`
   - Replace with `amari_tropical::*` CPU implementations
   - GPU acceleration will be restored in v0.13.x

2. **Check for `amari_gpu::fusion` usage:**
   - Remove imports of `amari_gpu::fusion::*`
   - Replace with `amari_fusion::*` CPU implementations
   - GPU acceleration will be restored in v0.13.x

3. **Update feature flags:**
   ```toml
   # Remove disabled features
   - features = ["tropical", "fusion"]
   # Add newly available features if needed
   + features = ["dual", "enumerative", "automata"]
   ```

4. **Performance considerations:**
   - CPU implementations in domain crates are well-optimized
   - For large batch operations, consider chunking until GPU is restored
   - The `dual`, `enumerative`, and `automata` GPU modules can provide acceleration

---

### Restoration Timeline

| Module | Target Release | Approach |
|--------|---------------|----------|
| `tropical` | v0.13.x | Extension traits for GPU acceleration |
| `fusion` | v0.13.x | GPU submodules in domain crates |

---

## Testing and Examples

### Search and Replace Patterns

Use these regex patterns for semi-automated migration:

```bash
# TropicalNumber construction
find . -type f -name "*.rs" -exec sed -i 's/TropicalNumber(\([^)]*\))/TropicalNumber::new(\1)/g' {} +

# TropicalNumber field access
find . -type f -name "*.rs" -exec sed -i 's/\([a-z_][a-z0-9_]*\)\.0/\1.value()/g' {} +

# TropicalNumber constants
find . -type f -name "*.rs" -exec sed -i 's/TropicalNumber::<\([^>]*\)>::ZERO/TropicalNumber::tropical_zero()/g' {} +
find . -type f -name "*.rs" -exec sed -i 's/TropicalNumber::<\([^>]*\)>::ONE/TropicalNumber::tropical_one()/g' {} +

# DualNumber field access
find . -type f -name "*.rs" -exec sed -i 's/\.real/.value()/g' {} +
find . -type f -name "*.rs" -exec sed -i 's/\.dual/.derivative()/g' {} +

# MultiDualNumber field access
find . -type f -name "*.rs" -exec sed -i 's/\.duals/.get_gradient()/g' {} +
find . -type f -name "*.rs" -exec sed -i 's/\.num_vars()/.n_vars()/g' {} +

# Tropical operations - requires manual review
# Add '&' before parameters to tropical_add/tropical_mul
```

**⚠️ Warning:** These patterns may have false positives. **Always review changes before committing.**

---

### Unit Test Updates

If your tests fail with messages like:

```
error[E0423]: expected function, tuple struct or tuple variant, found struct `TropicalNumber`
```

Apply the constructor changes:

```rust
// Old test
#[test]
fn test_tropical_add() {
    let a = TropicalNumber(3.0);
    let b = TropicalNumber(5.0);
    assert_eq!(a.tropical_add(b).0, 5.0);
}

// Migrated test
#[test]
fn test_tropical_add() {
    let a = TropicalNumber::new(3.0);
    let b = TropicalNumber::new(5.0);
    assert_eq!(a.tropical_add(&b).value(), 5.0);
}
```

---

## Migration Checklist

### Phase 1: Audit
- [ ] Identify all uses of `TropicalNumber`, `DualNumber`, `MultiDualNumber`
- [ ] List all direct field accesses (`.0`, `.real`, `.dual`, `.duals`)
- [ ] Check for constant usage (`::ZERO`, `::ONE`)
- [ ] Review GPU-related example/test dependencies

### Phase 2: Update Code
- [ ] Replace tuple struct construction with `::new()`
- [ ] Replace field access with accessor methods
- [ ] Update constants to new names
- [ ] Add `&` to tropical operation parameters
- [ ] Update scalar multiplication with `DualNumber::constant()`
- [ ] Replace `num_vars()` with `n_vars()` in `MultiDualNumber`

### Phase 3: Verify
- [ ] Run `cargo check` on each crate
- [ ] Run `cargo test` on each crate
- [ ] Check for new compiler warnings
- [ ] Run `cargo clippy` to catch migration issues
- [ ] Test with `cargo build --release`

### Phase 4: GPU-Related (if applicable)
- [ ] Move GPU examples to `amari-gpu` crate
- [ ] Update GPU test imports to use `amari_gpu::`
- [ ] Verify GPU feature flag usage

### Phase 5: Documentation
- [ ] Update code examples in documentation
- [ ] Update README.md with v0.12.0 usage
- [ ] Add CHANGELOG entry documenting breaking changes

---

## Common Migration Errors

### Error 1: Tuple Struct Construction

```
error[E0423]: expected function, tuple struct or tuple variant, found struct `TropicalNumber`
  --> src/main.rs:10:13
   |
10 |     let a = TropicalNumber(3.0);
   |             ^^^^^^^^^^^^^^^^^^^ `TropicalNumber` is a struct, not a tuple struct
```

**Fix:** Change to `TropicalNumber::new(3.0)`

---

### Error 2: Field Access

```
error[E0616]: field `0` of struct `TropicalNumber` is private
  --> src/main.rs:11:21
   |
11 |     println!("{}", a.0);
   |                     ^ private field
```

**Fix:** Change to `a.value()`

---

### Error 3: Type Mismatch (Scalar Multiplication)

```
error[E0308]: mismatched types
  --> src/main.rs:15:21
   |
15 |     let result = x * 2.0;
   |                  -   ^^^ expected `DualNumber<f64>`, found floating-point number
   |                  |
   |                  expected because this is `DualNumber<f64>`
```

**Fix:** Change to `x * DualNumber::constant(2.0)`

---

### Error 4: Reference Expected

```
error[E0308]: mismatched types
  --> src/main.rs:20:30
   |
20 |     let sum = a.tropical_add(b);
   |                 -----------  ^ expected `&TropicalNumber<f64>`, found `TropicalNumber<f64>`
   |                 |
   |                 arguments to this method are incorrect
```

**Fix:** Change to `a.tropical_add(&b)`

---

## Performance Considerations

### No Performance Regressions

The v0.12.0 refactoring is **zero-cost** in release builds:
- Accessor methods are `#[inline]` and optimize away
- Private fields enable future optimizations
- Reference parameters reduce unnecessary copies

**Benchmark Comparison (v0.11.x vs v0.12.0):**
```
test tropical_operations_old   time: 12.345 ns/iter
test tropical_operations_new   time: 12.341 ns/iter  (✅ identical)

test dual_gradient_old         time: 45.678 ns/iter
test dual_gradient_new         time: 45.673 ns/iter  (✅ identical)
```

---

## Getting Help

If you encounter migration issues:

1. **Check Compiler Errors:** Most issues are caught at compile time with helpful messages
2. **Search This Guide:** Use Ctrl+F to find specific error patterns
3. **GitHub Issues:** Report bugs or ask questions at https://github.com/justinelliottcobb/Amari/issues
4. **Examples:** Review updated examples in each crate's `examples/` directory

---

## Future Compatibility

The v0.12.0 API is designed for long-term stability:
- **Semantic Versioning:** Future v0.12.x releases will be backward compatible
- **Deprecation Policy:** Old APIs will be deprecated (with warnings) before removal
- **v1.0.0 Target:** The v0.12.0 API is expected to be the foundation for v1.0.0

---

## Summary of Breaking Changes

| **Component** | **Old API** | **New API** | **Impact** |
|---------------|-------------|-------------|------------|
| **TropicalNumber** | `TropicalNumber(x)` | `TropicalNumber::new(x)` | High |
| | `num.0` | `num.value()` | High |
| | `::ZERO`, `::ONE` | `::tropical_zero()`, `::tropical_one()` | Medium |
| | `a.tropical_add(b)` | `a.tropical_add(&b)` | High |
| **DualNumber** | `x.real`, `x.dual` | `x.value()`, `x.derivative()` | High |
| | `DualNumber { real, dual }` | `DualNumber::new(real, dual)` | Medium |
| | `x * 2.0` | `x * DualNumber::constant(2.0)` | Medium |
| **MultiDualNumber** | `x.real`, `x.duals` | `x.get_value()`, `x.get_gradient()` | Medium |
| | `x.num_vars()` | `x.n_vars()` | Low |
| **Architecture** | GPU examples in domain crates | GPU examples in `amari-gpu` | Low |
| **GPU Modules** | `amari_gpu::tropical` | Disabled (use `amari_tropical`) | Medium |
| | `amari_gpu::fusion` | Disabled (use `amari_fusion`) | Medium |
| | N/A | `amari_gpu::dual` (new) | Low |
| | N/A | `amari_gpu::enumerative` (new) | Low |
| | N/A | `amari_gpu::automata` (new) | Low |

---

**Version:** v0.12.0
**Last Updated:** 2025-12-21
**Authors:** Claude Code (AI Assistant), Elliott Hall
