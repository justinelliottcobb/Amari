# Mathematical Rigor Enforcement

## Philosophy

> "A mathematical library lives and dies on its correctness."

This repository prioritizes **mathematical correctness above all else**. Every commit, every pull request, and every release must pass the most rigorous verification possible.

## Verification Layers

### 1. 🛡️ Pre-commit Hook (Local)
**Purpose**: Immediate feedback to prevent bad commits
**Runtime**: ~30 seconds
**Coverage**: Complete verification identical to CI

```bash
# Automatically runs on every commit attempt
.githooks/pre-commit
```

**Blocks commits that fail:**
- Code formatting (rustfmt)
- Linting violations (clippy -D warnings)
- Any failing tests
- Documentation build failures

### 2. 🔬 GitHub Actions CI (Cloud)
**Purpose**: Comprehensive verification in clean environment
**Runtime**: ~10-15 minutes (rigor over speed)
**Triggers**: Every push, every PR

**Verification Steps:**
1. **Code Quality**: Format + Clippy with zero tolerance
2. **Core Mathematical Tests**: All library and binary tests
3. **Formal Verification**: Contracts and phantom type validation
4. **Integration Tests**: Cross-crate interaction verification
5. **Documentation Tests**: All doc examples must work
6. **Geometric Algebra Axioms**: Explicit algebraic property verification
7. **Phantom Type Safety**: Compile-time constraint validation
8. **Mathematical Property Analysis**: Test coverage requirements

### 3. 📊 Continuous Monitoring
**Test Coverage Requirements:**
- Minimum 300+ tests (currently 579+)
- 100% verification test success rate
- All mathematical axioms explicitly tested
- All phantom type constraints validated

## Mathematical Guarantees

### ✅ Geometric Algebra Correctness
- **Associativity**: `(ab)c = a(bc)` verified
- **Distributivity**: `a(b+c) = ab + ac` verified
- **Basis vector properties**: `e₁² = ±1` per signature
- **Multivector grade consistency**: All operations preserve grade

### ✅ Tropical Semiring Axioms
- **Idempotency**: `a ⊕ a = a` (max/min property)
- **Commutativity**: `a ⊕ b = b ⊕ a`
- **Associativity**: `(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)`
- **Distributivity**: `a ⊙ (b ⊕ c) = (a ⊙ b) ⊕ (a ⊙ c)`
- **Identity elements**: Tropical zero/one properties

### ✅ Phantom Type Safety
- **Compile-time dimension checking**: Matrix operations verified at compile time
- **Signature preservation**: `Cl(p,q,r)` constraints enforced
- **Type-safe tropical operations**: MaxPlus/MinPlus separation maintained
- **Verification contracts**: Creusot-style formal verification

### ✅ Information Geometry Properties
- **Fisher Information Matrix**: Positive definiteness verified
- **Amari-Chentsov tensor**: Correct geometric computation
- **Bregman divergences**: Non-negativity guaranteed
- **Dual connections**: ∇ and ∇* relationship verified

## Failure Handling

### Zero Tolerance Policy
- **Any test failure blocks the entire pipeline**
- **No exceptions for "minor" mathematical errors**
- **Documentation examples must be mathematically correct**

### Failure Investigation Protocol
1. **Immediate**: Identify which mathematical property failed
2. **Analysis**: Determine if it's a regression or new edge case
3. **Fix**: Correct the mathematical implementation
4. **Verification**: Ensure fix doesn't break other properties
5. **Enhancement**: Add tests to prevent similar regressions

## Performance vs. Correctness

### Current Choice: **CORRECTNESS FIRST**
- **CI Runtime**: 10-15 minutes (comprehensive verification)
- **Parallel optimization available but disabled**: Prioritizing rigor
- **Cost**: ~15-20 commits fit in free tier per day

### Future Optimization Path
When commit frequency increases:
1. Enable `parallel-verification.yml` for speed
2. Keep `mathematical-correctness.yml` for releases
3. Add matrix testing across Rust versions
4. Implement property-based testing

## Repository Protection

### Required Status Checks
- ✅ Mathematical Correctness workflow must pass
- ✅ All verification tests must pass
- ✅ Documentation must build successfully
- ✅ Zero clippy warnings tolerated

### Branch Protection Rules
```yaml
required_status_checks:
  strict: true
  contexts:
    - "Mathematical Correctness Check"
dismiss_stale_reviews: true
require_code_owner_reviews: true
required_approving_review_count: 1
```

## Mathematical Integrity Pledge

By contributing to this repository, you agree that:

1. **Mathematical correctness is non-negotiable**
2. **All algebraic properties must be explicitly tested**
3. **Type safety is a mathematical requirement, not optimization**
4. **Documentation examples are mathematical claims requiring verification**
5. **Performance optimizations cannot compromise correctness**

## Verification Commands

### Local Development
```bash
# Full verification (matches CI exactly)
./.githooks/pre-commit

# Specific mathematical property checks
cargo test geometric_axioms
cargo test semiring_axioms
cargo test verified_contracts
```

### Manual CI Trigger
```bash
# Trigger comprehensive verification
gh workflow run mathematical-correctness.yml

# Trigger optimization verification (when enabled)
gh workflow run parallel-verification.yml
```

---

**Remember**: In mathematics, there is no "close enough" - there is only correct or incorrect. This CI setup ensures we stay on the correct side of that line.