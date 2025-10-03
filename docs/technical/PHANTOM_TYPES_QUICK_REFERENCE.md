# Phantom Types Quick Reference for Claude Code Sessions

## 🚀 TL;DR Implementation Steps

### 1. **Pre-Flight Check** (5 minutes)
```bash
# Baseline measurement
cargo test -- --list | wc -l
cargo tarpaulin --out Html --output-dir coverage

# Target: 300%+ test increase, then phantom types
```

### 2. **Test Coverage Explosion** (Main effort)
Apply proven patterns from `amari-core` and `amari-tropical`:

**Copy these test module patterns:**
- `constructor_tests` - All creation methods
- `property_tests` - Mathematical axioms (associativity, commutativity, etc.)
- `edge_case_tests` - Infinity, zero, boundaries
- `operation_tests` - All arithmetic operations
- `interoperability_tests` - Cross-type operations

**Success criteria:** 95%+ tests passing before proceeding to phantom types

### 3. **Phantom Types** (After comprehensive tests)
```toml
# Cargo.toml
[features]
formal-verification = ["amari-core/formal-verification"]
contracts = ["formal-verification"]
```

```rust
// src/verified.rs
#[derive(Debug, Clone, Copy)]
pub struct VerifiedType<T: Float + Clone + Copy, P1, P2, ...> {
    inner: OriginalType<T>,
    _phantom: PhantomData<(P1, P2, ...)>,
}
```

### 4. **Formal Verification Contracts**
```rust
// src/verified_contracts.rs
impl<T: Float + Clone + Copy> VerifiedType<T, PhantomParams> {
    /// Contract: Mathematical property description
    pub fn axiom_verification_method(&self, ...) -> Self {
        // Implementation that verifies mathematical properties
    }
}
```

## ⚠️ Critical Bug Prevention

### Identity Element Confusion (Most Common Bug)
```rust
// ❌ WRONG - Will cause test failures
let sum = T::zero(); // Multiplicative identity (0.0)

// ✅ CORRECT - For additive operations
let sum = T::additive_identity(); // Or T::neg_infinity() for tropical
```

### Trait Bound Issues
```rust
// ❌ WRONG
pub struct Verified<T, S> { ... }

// ✅ CORRECT
pub struct Verified<T: Float + Clone + Copy, S> { ... }
```

### Method Resolution Ambiguity
```rust
// ❌ WRONG
VerifiedNumber::method()

// ✅ CORRECT
VerifiedNumber::<T, PhantomType>::method()
```

## 📊 Success Metrics Checklist

- [ ] **Before phantom types**: 95%+ test success rate
- [ ] **Test count increase**: 300%+ from baseline
- [ ] **After phantom types**: 100% test success rate
- [ ] **Performance**: Zero runtime overhead
- [ ] **Integration**: All existing tests work with phantom types

## 🔧 Copy-Paste Cargo.toml Template

```toml
[features]
default = ["std"]
std = []
serialize = ["serde"]
formal-verification = ["amari-core/formal-verification"]
contracts = ["formal-verification"]
```

## 🏗️ File Structure Template

```
src/
├── lib.rs                    # Existing implementation
├── verified.rs              # Phantom type wrappers
└── verified_contracts.rs    # Formal verification contracts
```

## 🎯 Test Pattern Templates

### Constructor Tests
```rust
#[test]
fn test_all_constructors() {
    let zero = Type::zero();
    let one = Type::one();
    let from_value = Type::new(5.0);

    assert!(zero.is_zero());
    assert!(one.is_one());
    assert_eq!(from_value.value(), 5.0);
}
```

### Mathematical Property Tests
```rust
#[test]
fn test_mathematical_properties() {
    let a = Type::new(2.0);
    let b = Type::new(3.0);
    let c = Type::new(1.0);

    // Commutativity
    assert_eq!(a.operation(b), b.operation(a));

    // Associativity
    assert_eq!((a.operation(b)).operation(c), a.operation(b.operation(c)));

    // Identity
    assert_eq!(a.operation(Type::identity()), a);
}
```

### Edge Case Tests
```rust
#[test]
fn test_edge_cases() {
    let inf = Type::new(f64::INFINITY);
    let neg_inf = Type::new(f64::NEG_INFINITY);
    let zero = Type::new(0.0);

    // Test infinity behavior
    assert!(inf.operation(zero).is_infinite());

    // Test boundary conditions
    assert_eq!(neg_inf.operation(zero), zero);
}
```

## 🚦 Implementation Status Tracking

For each crate, track progress:

```markdown
## [Crate Name] Status

- [ ] **Phase 1**: Baseline (X tests) → Target (X*3 tests)
- [ ] **Phase 2**: 95%+ test success achieved
- [ ] **Phase 3**: Phantom types implemented
- [ ] **Phase 4**: 100% test success with phantom types
- [ ] **Phase 5**: Formal verification contracts
- [ ] **Phase 6**: Ready for PR
```

## 🎯 Next Targets (2 crates per PR)

1. **amari-clifford** + **amari-differential**
2. **amari-enumerative** + **amari-synthetic**
3. **amari-tensor** + **amari-topology**

## 💡 Pro Tips

- **Always test first, phantom types second** - The comprehensive test suite catches semantic bugs that would break phantom type integration
- **Identity elements are the #1 bug source** - Double-check additive vs multiplicative identities
- **Use explicit type parameters** - Avoid method resolution ambiguity
- **Copy successful patterns** - The `amari-core` and `amari-tropical` implementations are proven references
- **Validate 100% success** - Don't proceed with broken tests; they indicate semantic issues

## 📖 Reference Implementations

- **amari-core**: [Basic phantom types, comprehensive test patterns]
- **amari-tropical**: [Advanced phantom types, formal verification contracts]
- **This methodology**: [Proven systematic approach]

---
*This guide ensures consistent, successful phantom type implementations across the entire Amari ecosystem.*