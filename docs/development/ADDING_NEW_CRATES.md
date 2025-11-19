# Adding New Crates to the Amari Workspace

This guide explains how to properly add a new crate to the Amari workspace.

## The Problem

Adding a crate to `Cargo.toml` workspace members **does NOT automatically add it to CI/CD workflows**. The publish workflow has a hardcoded list of crates, which means new crates won't be published unless you manually update the workflow files.

This has happened multiple times in the project (e.g., amari-flynn and amari-flynn-macros in v0.9.10).

## Step-by-Step Checklist

When adding a new crate, follow these steps:

### 1. Create the Crate

```bash
cargo new --lib amari-yourcrate
```

### 2. Add to Workspace (`Cargo.toml`)

Add to the `members` array in the root `Cargo.toml`:

```toml
[workspace]
members = [
    # ... existing crates ...
    "amari-yourcrate"
]
```

Add to workspace dependencies:

```toml
[workspace.dependencies]
amari-yourcrate = { path = "amari-yourcrate", version = "0.9.10" }
```

### 3. Update CI/CD Workflows

**CRITICAL**: You must manually update these files:

#### `.github/workflows/publish.yml`

Add your crate to the `CRATES` array in **dependency order** (lines ~96-111):

```bash
CRATES=(
    "amari-core"
    # ... other crates in dependency order ...
    "amari-yourcrate"  # Add here based on dependencies
    "amari"            # Always last
)
```

**Dependency order matters!** If `amari-yourcrate` depends on other crates, it must come after them.

#### `.github/workflows/parallel-verification.yml`

Add to the appropriate matrix strategy (lines ~38 or ~53):

```yaml
strategy:
  matrix:
    crate: [amari-fusion, amari-enumerative, amari-automata, amari-yourcrate]
```

#### `.github/workflows/test-status.yml`

Add to the test counting loop (line ~77):

```bash
for dir in amari-core amari-dual ... amari-yourcrate; do
```

### 4. Run Verification Script

We have a script to catch these issues automatically:

```bash
./scripts/verify-workflow-crates.sh
```

This will check if all workspace crates are properly configured in the publish workflow.

**Expected output:**
```
✅ All workspace crates are properly configured in publish workflow!
```

If you see errors, fix them before proceeding!

### 5. Update Documentation

- Add crate description to main `README.md`
- Add to `docs/README.md` if it has special documentation
- Update `CHANGELOG.md` with the new crate

### 6. Configure Crate Metadata

Each crate should have proper metadata in its `Cargo.toml`:

```toml
[package]
name = "amari-yourcrate"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
authors = ["Your Name <your.email@example.com>"]
repository = "https://github.com/justinelliottcobb/Amari"
description = "Brief description of what this crate does"
keywords = ["mathematics", "your", "keywords"]
categories = ["mathematics", "science"]
```

### 7. Test Before Publishing

```bash
# Run all tests
cargo test --workspace --all-features

# Try building the crate
cargo build -p amari-yourcrate

# Check dependencies
cargo tree -p amari-yourcrate
```

### 8. Publish

When ready to publish, use the automated workflow:

```bash
# Create and push a version tag
git tag v0.9.11
git push origin v0.9.11
```

Or manually trigger the workflow:

```bash
gh workflow run publish.yml --field version=0.9.11
```

## Common Mistakes to Avoid

### ❌ Only updating `Cargo.toml`

**Problem:** Crate won't be published to crates.io

**Solution:** Update all workflow files listed above

### ❌ Wrong dependency order

**Problem:** Publishing fails because dependencies aren't available yet

**Solution:** Place crate after all its dependencies in the CRATES array

### ❌ Forgetting `-macros` crates

**Problem:** Procedural macro crates often forgotten

**Solution:** Macro crates MUST be published before the crates that use them

### ❌ Not testing the verification script

**Problem:** Silent failures that only appear during publishing

**Solution:** Always run `./scripts/verify-workflow-crates.sh` before committing

## Excluded Crates

Some crates are intentionally NOT published to crates.io:

- **amari-wasm**: Published to npm instead of crates.io
- **amari-measure**: In development, not ready for publishing

These are listed in `scripts/verify-workflow-crates.sh` under `EXCLUDED_CRATES`.

## Quick Reference

**Files to Update:**
1. ✅ `Cargo.toml` - workspace members
2. ✅ `Cargo.toml` - workspace dependencies
3. ✅ `.github/workflows/publish.yml` - CRATES array
4. ✅ `.github/workflows/parallel-verification.yml` - matrix strategy
5. ✅ `.github/workflows/test-status.yml` - test counting loop
6. ✅ `README.md` - documentation
7. ✅ `CHANGELOG.md` - release notes

**Verification:**
```bash
./scripts/verify-workflow-crates.sh
```

## Historical Issues

- **v0.9.10**: amari-flynn and amari-flynn-macros were added to workspace but not to publish.yml, causing them to be skipped during publishing
- **v0.9.9**: Similar issue with initial deterministic physics crate

This documentation was created to prevent these issues from happening again.
