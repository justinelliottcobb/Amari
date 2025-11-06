# Amari Deployment Strategy

## Overview

This document outlines the deployment strategy for the Amari mathematical computing library across multiple package registries (crates.io and npm).

## Current State (v0.3.0)

### Crates.io Publications
All Rust crates are published to crates.io:
- ✅ `amari-core` - Core geometric algebra
- ✅ `amari-tropical` - Tropical algebra
- ✅ `amari-dual` - Dual numbers/automatic differentiation
- ✅ `amari-info-geom` - Information geometry
- ✅ `amari-enumerative` - Enumerative geometry
- ✅ `amari-fusion` - Fusion systems
- ✅ `amari-automata` - Cellular automata
- ✅ `amari-gpu` - GPU acceleration
- ✅ `amari` - Root crate re-exporting all functionality

### NPM Publications
Currently only one package:
- ✅ `@justinelliottcobb/amari-core` - WASM bindings (from `amari-wasm`)

## Proposed Strategy (v0.4.0+)

### Phase 1: Fix Current Issues (v0.3.1)
1. ✅ Add `amari-enumerative` to crates.io publish workflow
2. ✅ Sync npm package version with Rust crates
3. ⏳ Consider renaming npm package to `@justinelliottcobb/amari`

### Phase 2: Expand WASM Coverage (v0.4.0)

#### Primary WASM Package
**`@justinelliottcobb/amari`** (renamed from amari-core)
- Complete WASM bindings for all core functionality
- Includes all crates that work in WASM environment
- Single package for ease of use
- Tree-shakeable for optimal bundle size

#### Specialized WASM Packages (Future)
Consider separate packages for specialized use cases:
- `@justinelliottcobb/amari-gpu` - WebGPU acceleration
- `@justinelliottcobb/amari-automata` - Cellular automata (no_std compatible)

### Phase 3: Documentation & Examples (v0.4.0)
- TypeScript definitions for all exposed APIs
- JavaScript/TypeScript examples
- Integration guides for popular frameworks (React, Vue, etc.)

## Technical Considerations

### WASM Compatibility Matrix

| Crate | std/no_std | WASM Compatible | Notes |
|-------|------------|-----------------|-------|
| amari-core | std | ✅ Yes | Core functionality |
| amari-tropical | std (cfg no_std) | ✅ Yes | Can be made no_std |
| amari-dual | std (cfg no_std) | ✅ Yes | Can be made no_std |
| amari-info-geom | std | ⚠️ Maybe | Depends on dependencies |
| amari-enumerative | std | ⚠️ Maybe | Complex computations |
| amari-fusion | std (cfg no_std) | ✅ Yes | Can be made no_std |
| amari-automata | no_std | ✅ Yes | Already no_std |
| amari-gpu | std | ❌ No | Requires native GPU |

### Publishing Order

Due to dependency relationships, crates must be published in this order:

1. **Tier 1** (no dependencies)
   - amari-core

2. **Tier 2** (depends on core)
   - amari-tropical
   - amari-dual
   - amari-info-geom

3. **Tier 3** (depends on tier 2)
   - amari-enumerative (depends on core, tropical)
   - amari-fusion (depends on core, tropical, dual)
   - amari-automata (depends on core, tropical, dual)

4. **Tier 4** (depends on multiple)
   - amari-gpu (depends on core, info-geom)

5. **Tier 5** (root)
   - amari (depends on all)

## Version Synchronization

All crates maintain synchronized versions:
- Major/Minor versions always match (e.g., all at 0.3.x)
- Patch versions may vary for crate-specific fixes
- WASM package version matches Rust crates

## CI/CD Workflow

### Automated Publishing Triggers
- **Tag Push** (`v*`): Publishes to both crates.io and npm
- **Manual Workflow**: Allows selective publishing

### Publishing Steps
1. **Validation**: Format, clippy, tests
2. **Crates.io**: Publish in dependency order with delays
3. **WASM Build**: Build for web, Node.js, and bundler targets
4. **NPM Publish**: Publish WASM packages
5. **Documentation**: Update docs.rs and npm README

## Security Considerations

### Token Management
- `CARGO_REGISTRY_TOKEN`: Stored as GitHub secret
- `NPM_TOKEN`: Stored as GitHub secret
- Tokens have minimal required permissions
- Regular rotation schedule (quarterly)

### Package Integrity
- All packages signed with GPG keys
- SHA checksums verified
- Dependency audit before each release
- SBOM (Software Bill of Materials) generated

## Rollback Strategy

### Crates.io
- Cannot unpublish (by design)
- Yank broken versions: `cargo yank --vers x.y.z`
- Publish patch version with fixes

### NPM
- Deprecate broken versions: `npm deprecate @justinelliottcobb/amari@x.y.z "Critical bug"`
- Unpublish within 72 hours if necessary
- Publish patch version with fixes

## Monitoring & Analytics

### Package Metrics
- Download counts from crates.io API
- NPM download statistics
- GitHub stars and issue tracking
- Community feedback channels

### Quality Gates
- Minimum test coverage: 80%
- Zero clippy warnings
- Documentation coverage: 100% public APIs
- Example code for all major features

## Future Considerations

### WebAssembly Component Model
- Prepare for component model adoption
- Interface types for better JS interop
- Shared memory and threading support

### Alternative Registries
- GitHub Package Registry
- Self-hosted registry for pre-release versions
- Mirror to Chinese registries (npmjs.com, cnpm)

### Platform-Specific Packages
- Native Node.js bindings (N-API)
- Deno support
- Cloudflare Workers optimization

## Release Checklist

- [ ] Version bump in Cargo.toml files
- [ ] Version bump in package.json
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create git tag
- [ ] Push tag to trigger CI/CD
- [ ] Verify crates.io publication
- [ ] Verify npm publication
- [ ] Update example repositories
- [ ] Announce release

## Support Matrix

| Platform | Minimum Version | Tested Version |
|----------|-----------------|----------------|
| Rust | 1.75 | Latest stable |
| Node.js | 14.x | 18.x, 20.x |
| Browser | ES2015 | Chrome, Firefox, Safari latest |
| WASM | MVP | Latest |
| WebGPU | Experimental | Chrome 113+ |