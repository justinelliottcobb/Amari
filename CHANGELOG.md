# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2024-01-10

### Fixed
- Fix npm publish workflow to use wasm-pack generated package.json from pkg/ directory
- Remove redundant manual package.json from amari-wasm (wasm-pack generates its own)
- Correct package name references in CI/CD to use @justinelliottcobb/amari-wasm

### Changed
- CI/CD now publishes from amari-wasm/pkg/ instead of amari-wasm/
- Package version management now fully handled by wasm-pack from Cargo.toml

## [0.3.1] - 2024-01-10

### Fixed
- Add missing `amari-enumerative` crate to crates.io publish workflow
- Sync npm package version with Rust crates (was stuck at 0.1.1)

### Added
- Comprehensive deployment strategy documentation (`DEPLOYMENT_STRATEGY.md`)
- NPM publishing roadmap for phased WASM rollout (`NPM_PUBLISHING_ROADMAP.md`)

### Changed
- No breaking changes - patch release with deployment fixes only

## [0.3.0] - 2024-01-10

### Added
- Unified error handling system with `AmariError` type
- Error types for all crates: `CoreError`, `DualError`, `TropicalError`, `FusionError`
- Comprehensive error handling design documentation
- `thiserror` integration for consistent error patterns

### Changed
- All crates now use `Result` types instead of panics for recoverable errors
- Version bump from 0.2.0 to 0.3.0 across all workspace members

## [0.2.0] - 2024-01-09

### Added
- API naming convention guide (`API_NAMING_CONVENTION.md`)
- `magnitude()` method as primary API for computing vector/multivector length

### Changed
- Standardized method naming across all crates
- `norm()` methods maintained as backward-compatible aliases
- Version bump from 0.1.1 to 0.2.0

### Deprecated
- `norm()` methods (use `magnitude()` in new code)

## [0.1.1] - 2024-01-08

### Initial Release
- Core geometric algebra operations (`amari-core`)
- Tropical (max-plus) algebra (`amari-tropical`)
- Dual numbers for automatic differentiation (`amari-dual`)
- Information geometry (`amari-info-geom`)
- Fusion systems combining algebraic structures (`amari-fusion`)
- Cellular automata with geometric algebra (`amari-automata`)
- Enumerative geometry (`amari-enumerative`)
- GPU acceleration (`amari-gpu`)
- WebAssembly bindings (`amari-wasm`)

[0.3.1]: https://github.com/justinelliottcobb/Amari/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/justinelliottcobb/Amari/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/justinelliottcobb/Amari/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/justinelliottcobb/Amari/releases/tag/v0.1.1