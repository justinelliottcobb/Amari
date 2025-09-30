# Test Enforcement for Mathematical Correctness

## Overview

The Amari mathematical library implements strict test enforcement to ensure mathematical correctness across all commits. This is critical for a library that forms the foundation of mathematical computations.

## 🎯 Why Test Enforcement?

Mathematical libraries require absolute correctness because:
- **Cascading Errors**: A single mathematical bug can invalidate entire research results
- **Trust**: Users depend on mathematical libraries to be mathematically sound
- **Regression Prevention**: Changes to one mathematical domain shouldn't break others
- **Formal Verification**: Our phantom types and contracts must always be validated

## 🔧 Setup

Run once after cloning:

```bash
./setup-githooks.sh
```

This configures git to use our custom pre-commit hooks that enforce:

### ✅ Pre-Commit Checks

1. **Code Formatting** (`cargo fmt --check`)
   - Ensures consistent code style across the ecosystem

2. **Clippy Lints** (`cargo clippy --workspace --all-targets --all-features`)
   - Catches common bugs and enforces Rust best practices
   - All warnings must be resolved (treated as errors)

3. **Core Mathematical Tests** (`cargo test --workspace`)
   - All mathematical operations must pass their tests
   - Includes comprehensive test suites with 300%+ coverage
   - Validates mathematical properties and edge cases

4. **Formal Verification Tests** (`cargo test --workspace --features formal-verification`)
   - Phantom type safety validations
   - Mathematical contract verifications
   - Semiring axiom validations

5. **Documentation Build** (`cargo doc --workspace --no-deps`)
   - Ensures all documentation compiles correctly
   - Validates doctests and examples

## 🚦 Test Enforcement Workflow

### Normal Commit Flow
```bash
git add .
git commit -m "feat: implement new mathematical operation"
# Pre-commit hooks automatically run:
# ✅ Code formatting check
# ✅ Clippy lints
# ✅ Core tests (all crates)
# ✅ Formal verification tests
# ✅ Documentation build
# Commit proceeds if all checks pass
```

### Failed Check Workflow
```bash
git commit -m "feat: buggy implementation"
# ❌ Tests failed!
# Fix the issues, then retry
git add .
git commit -m "feat: correct implementation"
# ✅ All checks pass, commit proceeds
```

### Emergency Bypass (Use Sparingly)
```bash
# Only for urgent fixes when tests are temporarily broken
git commit --no-verify -m "emergency: critical security fix"
```

## 📊 Test Coverage Standards

The enforcement system validates our comprehensive test coverage:

### Test Count Targets
- **amari-core**: 140+ tests covering all geometric algebra operations
- **amari-tropical**: 53+ tests covering tropical algebra and formal verification
- **Each crate**: Minimum 300% increase from baseline

### Test Categories Required
- **Constructor Tests**: All creation methods validated
- **Property Tests**: Mathematical axioms verified
- **Edge Case Tests**: Infinity, zero, boundary conditions
- **Operation Tests**: All arithmetic and algebraic operations
- **Integration Tests**: Cross-crate compatibility
- **Formal Verification Tests**: Phantom type and contract validation

## 🔬 Formal Verification Enforcement

The system specifically validates:

### Mathematical Properties
- **Associativity**: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
- **Commutativity**: a ⊕ b = b ⊕ a
- **Identity Elements**: a ⊕ 0 = a
- **Distributivity**: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)

### Type Safety
- **Phantom Type Constraints**: Compile-time safety enforced
- **Dimensional Consistency**: Matrix/tensor operations validated
- **Semiring Constraints**: Mathematical structure preserved

### Contract Verification
- **Semiring Axioms**: Tropical algebra properties validated
- **Geometric Algebra Laws**: Clifford algebra properties checked
- **Information Geometry**: Manifold properties verified

## 📈 Benefits

### Mathematical Correctness
- **Zero Mathematical Regressions**: Changes cannot break existing mathematical properties
- **Continuous Validation**: Every commit validates the entire mathematical framework
- **Early Bug Detection**: Mathematical errors caught before they reach users

### Development Quality
- **Consistent Code Style**: Uniform formatting across all crates
- **Best Practices**: Clippy enforces Rust best practices
- **Documentation Quality**: All docs must build successfully

### Research Reliability
- **Reproducible Results**: Mathematical operations remain consistent
- **Trustworthy Foundation**: Researchers can depend on mathematical correctness
- **Formal Guarantees**: Phantom types provide compile-time mathematical safety

## 🛠️ Troubleshooting

### Common Issues

#### Test Failures
```bash
# See which specific tests failed
cargo test --workspace

# Run specific test for debugging
cargo test test_name -- --nocapture

# Run formal verification tests
cargo test --workspace --features formal-verification
```

#### Formatting Issues
```bash
# Fix formatting automatically
cargo fmt

# Check formatting
cargo fmt --check
```

#### Clippy Warnings
```bash
# See all warnings
cargo clippy --workspace --all-targets --all-features

# Auto-fix some issues
cargo clippy --workspace --all-targets --all-features --fix
```

#### Documentation Errors
```bash
# Build docs to see errors
cargo doc --workspace --no-deps

# Include private items for debugging
cargo doc --workspace --no-deps --document-private-items
```

### Disable Temporarily
If you need to temporarily disable enforcement (use sparingly):

```bash
# Disable for single commit
git commit --no-verify -m "emergency fix"

# Disable hooks entirely (not recommended)
git config core.hooksPath ""

# Re-enable hooks
git config core.hooksPath .githooks
```

## 📋 Maintenance

### Adding New Tests
When adding new mathematical operations:

1. **Implement comprehensive tests first**
2. **Follow the proven test patterns** from existing crates
3. **Include edge cases and mathematical properties**
4. **Add formal verification if applicable**
5. **Ensure all tests pass before committing**

### Updating Enforcement
To modify the pre-commit checks:

1. **Edit `.githooks/pre-commit`**
2. **Test changes thoroughly**
3. **Update this documentation**
4. **Communicate changes to the team**

## 🎯 Success Metrics

The enforcement system ensures:
- **100% test pass rate** on all commits
- **Zero mathematical regressions** across releases
- **Comprehensive coverage** of all mathematical operations
- **Formal verification** of critical mathematical properties
- **Consistent code quality** across all contributors

This creates a robust foundation for mathematical computing that researchers and developers can trust completely.

---

*This enforcement system is critical for maintaining the mathematical integrity that makes Amari a reliable foundation for computational mathematics and research.*