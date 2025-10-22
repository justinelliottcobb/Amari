# v0.9.8 Release Process: Synchronized Crate Management

## Lessons Learned from v0.9.7-1 Release Issues

The v0.9.7-1 release highlighted critical gaps in our release process that resulted in:
- Version inconsistencies across crates (mix of 0.9.6, 0.9.7, 0.9.7-1)
- Missing crates in publish workflows (`amari-optimization` not in CRATES array)
- Semver pre-release issues preventing main `amari` crate publication
- Dependency resolution failures due to mismatched versions

## v0.9.8 Release Requirements

### New Crate Integration: amari-dynamics
v0.9.8 will introduce the `amari-dynamics` crate. This MUST be integrated systematically.

## Mandatory Pre-Release Checklist

### 1. **Workspace Configuration Synchronization**
- [ ] Add `amari-dynamics` to `Cargo.toml` workspace members
- [ ] Add `amari-dynamics` workspace dependency with correct version
- [ ] Update main `amari` crate dependencies to include `amari-dynamics` (optional)
- [ ] Verify ALL workspace crates use `version.workspace = true`

### 2. **Publish Workflow Integration**
**CRITICAL**: Update `.github/workflows/publish.yml`
- [ ] Add `amari-dynamics` to the CRATES array in correct dependency order
- [ ] Verify dependency chain: core → domain-specific → integration → main
- [ ] Expected order for v0.9.8:
  ```bash
  CRATES=(
    "amari-core"
    "amari-tropical"
    "amari-dual"
    "amari-network"
    "amari-info-geom"
    "amari-relativistic"
    "amari-fusion"
    "amari-automata"
    "amari-enumerative"
    "amari-dynamics"        # NEW CRATE
    "amari-gpu"
    "amari-optimization"
    "amari"
  )
  ```

### 3. **WASM Integration**
- [ ] Create `amari-wasm/src/dynamics.rs` with WASM bindings
- [ ] Add dynamics module to `amari-wasm/src/lib.rs`
- [ ] Update `amari-wasm/Cargo.toml` dependencies
- [ ] Test WASM compilation and npm package generation

### 4. **GPU Integration**
- [ ] Add dynamics-specific GPU acceleration in `amari-gpu`
- [ ] Update GPU workflow examples with dynamics integration
- [ ] Verify GPU fallback patterns work correctly

### 5. **Version Synchronization Protocol**

#### Single Source of Truth
- Use **standard semantic versioning** (e.g., 0.9.8, NOT 0.9.8-1)
- ALL crates MUST be at the same version for each release
- Update `[workspace.package] version = "0.9.8"` as single source

#### Pre-Release Version Verification
```bash
# Verify all crates will be at the same version
grep -r "version.*=" Cargo.toml */Cargo.toml | grep -v workspace
# Should return ONLY external dependencies
```

### 6. **Dependency Order Validation**

#### Dependency Chain for v0.9.8:
1. **Foundation**: `amari-core` (no internal deps)
2. **Domain-Specific**: `amari-tropical`, `amari-dual`, `amari-network`, `amari-info-geom`, `amari-relativistic` (depend on core)
3. **Advanced**: `amari-fusion`, `amari-automata`, `amari-enumerative`, `amari-dynamics` (depend on core + domain)
4. **Acceleration**: `amari-gpu` (depends on multiple)
5. **Integration**: `amari-optimization` (depends on core + domain + gpu)
6. **Umbrella**: `amari` (depends on all)

### 7. **Testing Protocol**

#### Pre-Publication Testing
- [ ] `cargo test --workspace` passes
- [ ] `./run_all_tests.sh` passes
- [ ] WASM compilation succeeds
- [ ] GPU integration tests pass
- [ ] Documentation tests pass

#### Publication Simulation
```bash
# Test publish workflow locally (dry-run)
for crate in "${CRATES[@]}"; do
  echo "Testing $crate..."
  cd "$crate" || continue
  cargo publish --dry-run
  cd ..
done
```

### 8. **Release Execution Protocol**

#### Phase 1: Preparation
1. Create `v0.9.8-development` branch
2. Complete all integration work
3. Run full test suite
4. Update all documentation

#### Phase 2: Version Synchronization
1. Update workspace version to 0.9.8
2. Verify no version inconsistencies
3. Commit all changes as single atomic commit

#### Phase 3: Publication
1. Merge to master
2. Create git tag `v0.9.8`
3. Trigger publish workflow with version 0.9.8
4. Monitor all crates publish successfully

#### Phase 4: Verification
1. Verify all crates available at 0.9.8 on crates.io
2. Test npm package includes new dynamics module
3. Validate main `amari` crate includes all modules

## Automation Improvements for Future Releases

### 1. Pre-Commit Hooks
Add validation for:
- Workspace version consistency
- Publish workflow completeness
- WASM integration coverage

### 2. CI/CD Enhancements
- Automated dependency order validation
- WASM compilation checks on all PRs
- Version synchronization verification

### 3. Release Scripts
Create `scripts/prepare-release.sh`:
```bash
#!/bin/bash
VERSION=$1
echo "Preparing release $VERSION..."
# Update workspace version
# Validate all integrations
# Run comprehensive tests
# Generate release notes
```

## Success Metrics for v0.9.8

- [ ] ALL crates publish at exactly version 0.9.8
- [ ] Main `amari` crate includes `amari-dynamics` functionality
- [ ] WASM package includes dynamics module
- [ ] GPU acceleration works with dynamics
- [ ] Zero version inconsistencies across ecosystem
- [ ] Complete documentation coverage

## Emergency Rollback Plan

If v0.9.8 encounters similar issues:
1. Do NOT use pre-release versions (0.9.8-1)
2. Move directly to v0.9.9 with all fixes
3. Update this process document with lessons learned

---

**Remember**: The cost of fixing version inconsistencies post-release is far higher than ensuring synchronization during development. This systematic approach will prevent the v0.9.7-1 issues from recurring.