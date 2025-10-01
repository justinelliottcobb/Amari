# Amari API Naming Convention Guide

## Overview

This document establishes consistent naming conventions across all Amari crates to achieve professional, predictable API design for v1.0.0 release.

## Current Naming Inconsistencies Identified

### 1. **Geometric Product Operations**

**Current State**: ✅ **CONSISTENT**
- All crates use `geometric_product()` consistently
- Found in: amari-core, amari-dual, amari-tropical, amari-wasm, amari-fusion, amari-gpu
- **No changes needed**

### 2. **Magnitude/Norm Operations**

**Current State**: ❌ **INCONSISTENT**
- **Duplicated methods**: Both `magnitude()` and `norm()` exist in same types
- **Semantic confusion**: Unclear when to use which method

**Found inconsistencies**:
```rust
// amari-core::Multivector - BOTH methods exist
pub fn norm(&self) -> f64 { ... }        // Line 365
pub fn magnitude(&self) -> f64 { ... }    // Line 370

// amari-core::Bivector - BOTH methods exist
pub fn magnitude(&self) -> f64 { ... }    // Line 804
pub fn norm(&self) -> f64 { ... }         // Line 870

// amari-dual::DualMultivector - BOTH methods exist
pub fn magnitude(&self) -> T { ... }      // Line 126
pub fn norm(&self) -> DualNumber<T> { ... } // Line 261

// amari-wasm::WasmMultivector - Only norm()
pub fn norm(&self) -> f64 { ... }         // Line 155
```

### 3. **Basis Vector Creation**

**Current State**: ✅ **MOSTLY CONSISTENT**
- Primary method: `basis_vector(index)` (consistent across crates)
- Alternative: `e(index, value)` method in basis.rs (builder pattern - acceptable)
- **No changes needed**

## Established Naming Standards

### ✅ **Confirmed Standards (Keep As-Is)**

#### **Core Mathematical Operations**
```rust
// Geometric algebra operations (CONSISTENT)
geometric_product(other)     // Clifford product a * b
inner_product(other)         // Inner product a · b
outer_product(other)         // Wedge product a ∧ b
scalar_product(other)        // Scalar part of geometric product

// Vector creation (CONSISTENT)
basis_vector(index)          // Create basis vector e_i
zero()                       // Zero element
scalar(value)                // Scalar multivector

// Algebraic operations (CONSISTENT)
reverse()                    // Reverse operation ~a
inverse()                    // Multiplicative inverse a⁻¹
normalize()                  // Unit normalization a/|a|
```

### 🔧 **Standards to Fix**

#### **Magnitude/Norm Standardization**

**Problem**: Duplicated and inconsistent magnitude/norm methods

**Solution**: Standardize on `magnitude()` as primary method, remove `norm()`

**Rationale**:
1. **Mathematical accuracy**: "Magnitude" is the standard term in geometric algebra
2. **Geometric intuition**: "Magnitude" clearly refers to |a| = √(a·ã)
3. **API clarity**: Single method eliminates confusion
4. **Rust conventions**: `magnitude()` aligns with mathematical naming

**Implementation**:
```rust
// STANDARD: Use magnitude() everywhere
pub fn magnitude(&self) -> f64 {
    // Implementation: sqrt(self.scalar_product(&self.reverse()))
}

// DEPRECATED: Remove norm() method
// pub fn norm(&self) -> f64 { ... } // DELETE THIS
```

## Implementation Plan

### **Phase 1: magnitude/norm Unification**

#### **Files to Modify**:

1. **amari-core/src/lib.rs**:
   - Keep: `magnitude()` method (Line 370)
   - Remove: `norm()` method (Line 365)
   - Keep: Bivector `magnitude()` (Line 804)
   - Remove: Bivector `norm()` (Line 870)

2. **amari-dual/src/multivector.rs**:
   - Keep: `magnitude()` method (Line 126)
   - Evaluate: `norm()` method (Line 261) - may have different semantics for dual numbers

3. **amari-wasm/src/lib.rs**:
   - Rename: `norm()` → `magnitude()` (Line 155)

4. **Update all references**:
   - Search for `.norm()` calls and replace with `.magnitude()`
   - Update documentation and examples
   - Update test cases

### **Phase 2: Documentation Updates**

```rust
/// Compute the magnitude (length) of this multivector
///
/// The magnitude is defined as |a| = √(a·ã) where ã is the reverse of a.
/// This provides the natural norm inherited from the underlying vector space.
///
/// # Mathematical Properties
/// - Always non-negative: |a| ≥ 0
/// - Zero iff a = 0: |a| = 0 ⟺ a = 0
/// - Multiplicative: |ab| ≤ |a||b| (sub-multiplicative)
///
/// # Examples
/// ```rust
/// use amari_core::Multivector;
/// let v = Multivector::<3,0,0>::basis_vector(0);
/// assert_eq!(v.magnitude(), 1.0);
/// ```
pub fn magnitude(&self) -> f64 { ... }
```

## Verification Strategy

### **Compatibility Checks**

1. **Compile-time verification**: Ensure all `.norm()` references are updated
2. **Test suite validation**: All existing tests must pass with new naming
3. **Documentation builds**: Ensure all doc examples compile
4. **WASM compatibility**: Verify JavaScript interop still works

### **Migration Guide**

For users upgrading to v0.2.0:

```rust
// BEFORE (v0.1.x)
let length = multivector.norm();

// AFTER (v0.2.0+)
let length = multivector.magnitude();
```

## Success Criteria

✅ **API Consistency**:
- Single method for magnitude computation across all crates
- Consistent naming eliminates developer confusion
- Professional, predictable API surface

✅ **Mathematical Accuracy**:
- Terminology aligns with geometric algebra literature
- Clear semantic meaning for each method

✅ **Developer Experience**:
- No more guessing between `norm()` vs `magnitude()`
- Consistent muscle memory across entire API
- Better IDE autocompletion experience

## Implementation Timeline

| Task | Duration | Deliverable |
|------|----------|-------------|
| Remove duplicate `norm()` methods | 2-3 hours | Clean API surface |
| Update all call sites | 2-3 hours | Consistent usage |
| Update documentation | 1-2 hours | Clear examples |
| Test validation | 1 hour | Passing test suite |
| **Total** | **6-9 hours** | **v0.2.0 ready** |

## Future Considerations

### **Additional Standardization Opportunities**

1. **Error types**: Unify error hierarchies (Phase A2)
2. **Feature flags**: Standardize optional functionality (Phase C1)
3. **Module organization**: Consistent re-export patterns
4. **Generic constraints**: Standardize type bounds

### **Version 1.0 Stability Promise**

Once v1.0.0 is released, these naming conventions become **stable API contracts**:
- Method names will not change without major version bump
- Semantic meaning will remain consistent
- New methods will follow established patterns

---

**Document Status**: ✅ Ready for implementation
**Target Version**: v0.2.0
**Estimated Impact**: Low risk, high developer experience improvement