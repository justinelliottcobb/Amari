# Version Management System

This document describes the version management system for the Amari project, which ensures all packages stay synchronized across the Rust workspace and WebAssembly packages.

## Overview

The Amari project consists of multiple components that need to stay version-synchronized:

- **Rust Workspace**: Main workspace defined in `Cargo.toml`
- **Individual Rust Crates**: All amari-* crates inherit from workspace version
- **WebAssembly Package**: `amari-wasm/package.json` (npm package)

## Automated Version Synchronization

### Version Sync Script

The `scripts/version-sync.sh` script provides automated version management:

```bash
# Show current version status
./scripts/version-sync.sh status

# Set all versions to a specific version
./scripts/version-sync.sh set 1.2.3

# Bump version automatically
./scripts/version-sync.sh bump patch   # 1.2.3 -> 1.2.4
./scripts/version-sync.sh bump minor   # 1.2.3 -> 1.3.0
./scripts/version-sync.sh bump major   # 1.2.3 -> 2.0.0

# Verify all versions match expected version
./scripts/version-sync.sh verify 1.2.3
```

### What Gets Updated

The script synchronizes versions across:

1. **Cargo.toml workspace.package.version** - Main workspace version
2. **Cargo.toml workspace.dependencies** - All internal crate references
3. **amari-wasm/package.json version** - npm package version

## Release Process

### For New Releases

1. **Decide on version number** following [Semantic Versioning](https://semver.org/):
   - `MAJOR.MINOR.PATCH` (e.g., 0.9.3)
   - Breaking changes → bump MAJOR
   - New features → bump MINOR
   - Bug fixes → bump PATCH

2. **Update all versions**:
   ```bash
   # Option 1: Set specific version
   ./scripts/version-sync.sh set 0.9.3

   # Option 2: Auto-bump version
   ./scripts/version-sync.sh bump minor
   ```

3. **Verify synchronization**:
   ```bash
   ./scripts/version-sync.sh verify 0.9.3
   ```

4. **Commit version changes**:
   ```bash
   git add Cargo.toml amari-wasm/package.json
   git commit -m "Bump version to 0.9.3"
   ```

5. **Continue with normal release process** (PR, testing, merge, tag)

### For Development

Always check version status before starting new features:

```bash
./scripts/version-sync.sh status
```

If versions are out of sync, fix them before proceeding:

```bash
./scripts/version-sync.sh set $(./scripts/version-sync.sh status | grep "Cargo.toml workspace" | cut -d: -f2 | tr -d ' ')
```

## Manual Version Management

If you need to manually update versions (not recommended), ensure you update:

### Rust Workspace (Cargo.toml)

```toml
[workspace.package]
version = "0.9.3"

[workspace.dependencies]
amari = { path = ".", version = "0.9.3" }
amari-core = { path = "amari-core", version = "0.9.3" }
# ... all other amari-* crates
```

### WebAssembly Package (amari-wasm/package.json)

```json
{
  "name": "@justinelliottcobb/amari-wasm",
  "version": "0.9.3",
  ...
}
```

## Version Validation

### Pre-commit Hooks

Consider adding this to your pre-commit hooks:

```bash
# .githooks/pre-commit
./scripts/version-sync.sh verify $(grep -E "^version = " Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
if [ $? -ne 0 ]; then
    echo "❌ Version synchronization check failed!"
    echo "Run: ./scripts/version-sync.sh status"
    exit 1
fi
```

### CI/CD Integration

Add version sync verification to CI:

```yaml
# .github/workflows/ci.yml
- name: Verify version synchronization
  run: |
    WORKSPACE_VERSION=$(grep -E "^version = " Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
    ./scripts/version-sync.sh verify $WORKSPACE_VERSION
```

## Troubleshooting

### Common Issues

1. **Versions out of sync**:
   ```bash
   ./scripts/version-sync.sh status
   ./scripts/version-sync.sh set <correct-version>
   ```

2. **Script reports mismatch but versions look correct**:
   - Check for hidden characters or formatting issues
   - Run `cat -A Cargo.toml` and `cat -A amari-wasm/package.json` to see hidden chars

3. **Script fails to update versions**:
   - Ensure you have write permissions
   - Check that sed is available and supports -i flag
   - Verify file paths are correct

### Version Format Requirements

- Must follow semantic versioning: `MAJOR.MINOR.PATCH`
- Examples: `1.0.0`, `0.9.3`, `2.1.5`
- Pre-release suffixes supported: `1.0.0-alpha`, `1.0.0-beta.1`
- Build metadata supported: `1.0.0+build.1`

## Best Practices

1. **Always use the script** for version updates
2. **Verify synchronization** before committing
3. **Update versions atomically** (all at once)
4. **Follow semantic versioning** guidelines
5. **Document breaking changes** in release notes
6. **Test after version updates** to ensure everything still works

## Integration with Release Automation

The version sync script can be integrated into release automation:

```bash
#!/bin/bash
# Release automation script

# Bump version
./scripts/version-sync.sh bump minor

# Get new version
NEW_VERSION=$(grep -E "^version = " Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')

# Commit version bump
git add Cargo.toml amari-wasm/package.json
git commit -m "Release v$NEW_VERSION"

# Create tag
git tag "v$NEW_VERSION"

# Push changes
git push origin main --tags
```

This ensures consistent, automated version management across the entire project.