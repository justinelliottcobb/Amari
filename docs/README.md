# Amari Documentation

This directory contains comprehensive documentation for the Amari mathematical computing library.

## Documentation Structure

### 📋 Quick Links

- **[Main README](../README.md)** - Project overview and quick start
- **[Changelog](../CHANGELOG.md)** - Version history and changes
- **[Version Management](../VERSION_MANAGEMENT.md)** - Version synchronization procedures

### 📦 Releases

Release documentation, processes, and summaries:

- **[Release Process](releases/RELEASE_PROCESS.md)** - Standard release procedures
- **[v0.9.8 Release Process](releases/RELEASE_PROCESS_v0.9.8.md)** - Lessons learned and improvements
- **[v0.9.6 Release Summary](releases/v0.9.6_RELEASE_SUMMARY.md)** - Multi-GPU infrastructure release
- **[v0.9.5 Release Summary](releases/v0.9.5_RELEASE_SUMMARY.md)** - Complete GPU coverage achievement
- **[v0.9.5 GPU Optimization Achievement](releases/GPU_OPTIMIZATION_ACHIEVEMENT_v0.9.5.md)**

### 🏗️ Architecture

Technical architecture and design documentation:

- **[Technical Overview](architecture/technical-overview.md)** - Comprehensive technical documentation
- **[Integration Status](architecture/INTEGRATION_STATUS.md)** - WASM and GPU integration matrix
- **[Multi-GPU Design](architecture/v0.9.6-multi-gpu-design.md)** - Multi-GPU architecture details

### 🔧 Development

Development guides and procedures:

- Coming soon...

### 📚 Examples

Example code and learning resources:

- **[Examples README](../examples/README.md)** - Example code overview
- **[Learning Paths](../examples/LEARNING_PATHS.md)** - Structured learning paths
- **[Examples Documentation](../examples/DOCUMENTATION.md)** - Detailed example documentation

### 📦 Archive

Historical and outdated documentation (kept for reference):

- **[DUAL GPU Integration Plan](archive/DUAL_GPU_INTEGRATION_PLAN.md)** - Historical planning document
- **[GPU Integration Plan v0.9.4](archive/GPU_INTEGRATION_PLAN_v0.9.4.md)** - Historical planning document
- **[Tropical GPU Integration Plan](archive/TROPICAL_GPU_INTEGRATION_PLAN.md)** - Historical planning document
- **[WASM/GPU Implementation Status](archive/WASM_GPU_IMPLEMENTATION_STATUS.md)** - Historical status document
- **[Roadmap v0.9.5](archive/ROADMAP_v0.9.5.md)** - Historical roadmap

## Getting Started

1. **New Users**: Start with the [Main README](../README.md)
2. **Contributors**: Check [Technical Overview](architecture/technical-overview.md) and [Integration Status](architecture/INTEGRATION_STATUS.md)
3. **Developers**: Review the [Release Process](releases/RELEASE_PROCESS.md) before making releases
4. **Learners**: Follow the [Learning Paths](../examples/LEARNING_PATHS.md) for structured tutorials

## Crate-Specific Documentation

Individual crates have their own documentation:

- **[amari-network README](../amari-network/README.md)** - Geometric network analysis
- **[amari-network Mathematical Foundations](../amari-network/MATHEMATICAL_FOUNDATIONS.md)** - Mathematical background
- **[amari-optimization README](../amari-optimization/README.md)** - Optimization algorithms
- **[amari-wasm README](../amari-wasm/README.md)** - WebAssembly bindings
- **[amari-automata CONTEXT](../amari-automata/CONTEXT.md)** - Cellular automata context

## Contributing to Documentation

When adding new documentation:

1. **Place it appropriately**:
   - Release docs → `docs/releases/`
   - Architecture docs → `docs/architecture/`
   - Development guides → `docs/development/`
   - Outdated docs → `docs/archive/`

2. **Update this index** to include the new document

3. **Follow naming conventions**:
   - Use lowercase with hyphens: `multi-gpu-design.md`
   - Include version numbers where relevant: `v0.9.6-multi-gpu-design.md`
   - Use descriptive names: `RELEASE_PROCESS.md` not `process.md`

## Documentation Maintenance

- **Review quarterly** to identify outdated content
- **Archive** completed plans and historical documents
- **Update** integration status after major releases
- **Keep** this index current with all documentation files
