# NPM Publishing Roadmap

## Executive Summary

Publishing all Amari crates to npm requires significant WASM binding work. Rather than rushing this into a patch release, we propose a phased approach across multiple minor versions.

## Recommended Release Path

### 🔧 **v0.3.1** - Deployment Pipeline Fix (Immediate)
**Timeline**: 1 day
**Scope**: Current PR only
- ✅ Fix missing amari-enumerative in crates.io publish
- ✅ Sync npm package version to 0.3.0 → 0.3.1
- ✅ Add deployment strategy documentation
- ❌ No WASM changes yet

**Why**: Critical fix that unblocks current deployment issues without introducing risk.

---

### 📦 **v0.4.0** - WASM Infrastructure (1-2 weeks)
**Timeline**: 1-2 weeks
**Scope**: Core WASM expansion
- Expand amari-wasm to include all core mathematical crates
- Add WASM bindings for:
  - `amari-tropical`
  - `amari-dual`
  - `amari-fusion`
  - `amari-info-geom`
- Create unified WASM module with feature flags
- Single npm package: `@justinelliottcobb/amari`

**Deliverables**:
```
@justinelliottcobb/amari (complete WASM bindings)
├── core (geometric algebra)
├── tropical (max-plus algebra)
├── dual (automatic differentiation)
├── fusion (combined systems)
└── info-geom (information geometry)
```

**Why**: Establishes foundation with proven, stable crates first.

---

### 🚀 **v0.5.0** - Advanced WASM Features (2-3 weeks)
**Timeline**: 2-3 weeks
**Scope**: Complex crates requiring special handling
- Add WASM support for:
  - `amari-automata` (no_std, already compatible)
  - `amari-enumerative` (has partial WASM support)
- Create specialized builds:
  - `@justinelliottcobb/amari-gpu` (WebGPU features)
- Implement tree-shaking optimizations
- Add TypeScript type definitions

**Deliverables**:
```
@justinelliottcobb/amari (enhanced)
├── automata (cellular automata)
└── enumerative (intersection theory)

@justinelliottcobb/amari-gpu (separate package)
└── WebGPU acceleration
```

**Why**: More complex crates need careful testing and optimization.

---

### 🎯 **v0.6.0** - Multi-Package Architecture (3-4 weeks)
**Timeline**: 3-4 weeks
**Scope**: Individual npm packages per module
- Split monolithic WASM into feature-specific packages
- Create TypeScript wrapper packages:
  - `@justinelliottcobb/amari-core`
  - `@justinelliottcobb/amari-tropical`
  - `@justinelliottcobb/amari-dual`
  - `@justinelliottcobb/amari-fusion`
  - `@justinelliottcobb/amari-automata`
  - `@justinelliottcobb/amari-enumerative`
- Optimize bundle sizes per package
- Add package-specific examples and docs

**Why**: Users can import only what they need, reducing bundle size.

---

### ✨ **v0.7.0** - Developer Experience (2 weeks)
**Timeline**: 2 weeks
**Scope**: Polish and tooling
- Framework integrations (React, Vue, Svelte)
- Build tool plugins (Vite, Webpack, Rollup)
- Interactive documentation site
- Performance benchmarks
- Migration guides

---

### 🏁 **v1.0.0** - Production Ready (4 weeks)
**Timeline**: 4 weeks
**Scope**: Final stabilization
- API stability guarantee
- Performance optimizations
- Comprehensive test coverage
- Security audit
- LTS support commitment

## Decision Matrix

### Option A: Fast Track (Not Recommended)
**Release**: v0.3.1
**Timeline**: 2-3 weeks rushed
**Pros**: Everything available immediately
**Cons**: High risk, poor quality, breaking changes likely

### Option B: Phased Rollout (Recommended) ✅
**Releases**: v0.3.1 → v0.4.0 → v0.5.0 → v0.6.0
**Timeline**: 8-10 weeks total
**Pros**: Low risk, high quality, progressive enhancement
**Cons**: Longer timeline

### Option C: Single Big Release
**Release**: v0.4.0
**Timeline**: 6-8 weeks
**Pros**: Single migration for users
**Cons**: Long wait, high complexity, harder to test

## Implementation Priority

### High Priority (v0.4.0)
1. **amari-core**: Foundation for everything
2. **amari-dual**: Critical for ML/AI use cases
3. **amari-tropical**: Unique capability

### Medium Priority (v0.5.0)
4. **amari-fusion**: Combined systems
5. **amari-automata**: Specialized use cases
6. **amari-info-geom**: Statistical applications

### Lower Priority (v0.6.0)
7. **amari-enumerative**: Academic/research focus
8. **amari-gpu**: Performance optimization

## Risk Assessment

### Technical Risks
- **WASM Size**: Monolithic build could be 5-10MB
  - *Mitigation*: Feature flags, code splitting
- **Browser Compatibility**: WebAssembly support varies
  - *Mitigation*: Polyfills, fallbacks
- **Performance**: WASM overhead for small operations
  - *Mitigation*: Batch operations, Web Workers

### Project Risks
- **Scope Creep**: Trying to do too much at once
  - *Mitigation*: Strict phase boundaries
- **Breaking Changes**: API instability during development
  - *Mitigation*: Experimental flags until v1.0
- **Maintenance Burden**: Multiple packages to maintain
  - *Mitigation*: Automated tooling, monorepo structure

## Success Criteria

### v0.4.0 Success Metrics
- [ ] All core crates compile to WASM
- [ ] Single npm package published
- [ ] Bundle size < 2MB gzipped
- [ ] Basic TypeScript types
- [ ] 10+ working examples

### v0.5.0 Success Metrics
- [ ] GPU features working in Chrome
- [ ] All crates available via WASM
- [ ] Bundle size < 3MB gzipped
- [ ] Complete TypeScript types
- [ ] 50+ working examples

### v0.6.0 Success Metrics
- [ ] Individual packages < 500KB each
- [ ] Tree-shaking reduces size by 60%+
- [ ] Framework integrations working
- [ ] Documentation website live
- [ ] 100+ working examples

## Recommendation

**Start with v0.3.1 for the deployment fix, then proceed with v0.4.0 for core WASM expansion.**

This provides:
1. Immediate fix for deployment issues
2. Solid foundation for npm ecosystem
3. Time to properly design and test
4. Progressive enhancement path
5. Clear communication to users

## Next Steps

1. **Merge current PR as v0.3.1**
2. **Create feature branch for v0.4.0 WASM work**
3. **Prototype amari-tropical WASM bindings first**
4. **Benchmark bundle sizes and performance**
5. **Get community feedback on API design**

---

This phased approach balances speed with quality, ensuring we deliver robust npm packages without compromising the stability users expect from Amari.