# Release Process

This document describes the release process for the Amari mathematical computing library.

## Version Management

The Amari workspace uses synchronized versioning across all crates and packages. All crates share the same version number, which is managed at the workspace level.

### Version Bump Script

Use the `scripts/bump-version.sh` script to update versions across the entire workspace:

```bash
./scripts/bump-version.sh <new_version>
```

Example:
```bash
./scripts/bump-version.sh 0.6.2
```

The script will:
- Update the workspace version in root `Cargo.toml`
- Update all inter-crate dependency versions (using major.minor format)
- Update all `package.json` files in TypeScript packages
- Update version badges in README.md
- Add an entry to CHANGELOG.md with the current date
- Validate the version format (X.Y.Z or X.Y.Z-suffix)

### Manual Release Process

1. **Bump the version:**
   ```bash
   ./scripts/bump-version.sh 0.6.2
   ```

2. **Review the changes:**
   ```bash
   git diff
   ```

3. **Commit the version bump:**
   ```bash
   git commit -am "chore: bump version to 0.6.2"
   ```

4. **Tag the release:**
   ```bash
   git tag v0.6.2
   ```

5. **Push changes and tags:**
   ```bash
   git push origin master
   git push origin v0.6.2
   ```

6. **Trigger the publish workflow:**
   ```bash
   gh workflow run publish.yml
   ```

### Automated Release via GitHub Actions

The publish workflow can automatically bump versions when triggered with a version parameter:

```bash
gh workflow run publish.yml --field version=0.6.2
```

This will:
1. Automatically run the version bump script
2. Run all tests and validations
3. Publish packages to crates.io (in dependency order)
4. Build and publish WASM packages to npm
5. Create a GitHub release with generated notes

### Publishing Order

Crates are published in dependency order to ensure all dependencies are available:

1. `amari-core` - Core geometric algebra structures
2. `amari-tropical` - Tropical algebra (depends on core)
3. `amari-dual` - Dual numbers (depends on core)
4. `amari-info-geom` - Information geometry (depends on core)
5. `amari-enumerative` - Enumerative geometry (depends on core, tropical)
6. `amari-fusion` - Fusion systems (depends on core, tropical, dual)
7. `amari-automata` - Cellular automata (depends on core, tropical, dual)
8. `amari-gpu` - GPU acceleration (depends on core, info-geom)
9. `amari` - Main crate (depends on all above)

### Version Format

- **Release versions:** `X.Y.Z` (e.g., `0.6.1`)
- **Pre-release versions:** `X.Y.Z-suffix` (e.g., `0.7.0-beta.1`)
- **Dependency versions:** Use `X.Y` format in Cargo.toml (e.g., `0.6` matches `0.6.0`, `0.6.1`, etc.)

### Troubleshooting

#### Publishing Fails with "Email Not Verified"
- Visit https://crates.io/settings/profile to verify your email address
- Ensure the `CARGO_REGISTRY_TOKEN` secret is set in GitHub repository settings

#### Version Already Published
- Crates.io does not allow republishing the same version
- Bump to a new version number and try again

#### Dependencies Not Found
- Ensure crates are published in the correct order
- The workflow includes 30-second delays between publishes for indexing
- If a dependency fails to publish, subsequent crates will also fail

### GitHub Secrets Required

- `CARGO_REGISTRY_TOKEN` - API token for crates.io publishing
- `NPM_TOKEN` - API token for npm publishing (if publishing TypeScript packages)

## Release Checklist

- [ ] All tests passing locally
- [ ] CHANGELOG.md updated with release notes
- [ ] Version bumped using the script
- [ ] Changes committed and pushed
- [ ] Release tagged with `v` prefix
- [ ] Publishing workflow successful
- [ ] Packages visible on crates.io
- [ ] npm packages updated (if applicable)
- [ ] GitHub release created