# TARGET Amari Examples Suite

Welcome to the comprehensive Amari examples collection! This suite demonstrates the power and elegance of geometric algebra and dual numbers across multiple domains.

## FEATURED What's Included

### SCIENTIFIC Physics Simulations (`rust/physics-simulation/`)
Real-world physics demonstrations using geometric algebra's natural representations:

- **🌀 Rigid Body Dynamics**: Singularity-free rotations using rotors
- **ELECTROMAGNETIC Electromagnetic Fields**: Unified E+B field treatment with Maxwell's equations
- **🌊 Fluid Dynamics**: Vorticity as bivectors, circulation theorems
- **QUANTUM Quantum Mechanics**: Pauli matrices, spin states, Bell's inequality

### GRAPHICS Computer Graphics (`rust/computer-graphics/`)
Modern graphics applications showcasing GA's geometric intuition:

- **🎭 3D Transformations**: Gimbal lock-free rotations and interpolation
- **📷 Camera Systems**: Perspective projection and orbital controls
- **🔺 Mesh Operations**: Normal calculations and geometric queries
- **FEATURED Ray Tracing**: Natural ray representation and lighting

### 🧠 Machine Learning (`rust/machine-learning/`)
Verified ML algorithms using dual number automatic differentiation:

- **COMPUTATION Automatic Differentiation**: Exact gradients without approximation errors
- **🤖 Neural Networks**: Verified backpropagation and training
- **METRICS Optimization**: Gradient descent, Adam, Newton's method
- **VERIFIED Verified Learning**: Mathematical guarantees and error analysis

### INTERACTIVE Interactive Demos (`web/interactive-demos/`)
Real-time visualizations for hands-on learning:

- **🎮 3D Rotor Manipulator**: Interactive rotation exploration
- **ELECTROMAGNETIC EM Field Visualizer**: Dynamic electromagnetic field lines
- **📊 AutoDiff Grapher**: Real-time derivative computation
- **TARGET Optimization Tracer**: Watch algorithms converge live

## 🚀 Quick Start

### Prerequisites
- Rust 1.70+ with Cargo
- Node.js 16+ (for web demos)
- Basic linear algebra knowledge

### Running Examples

```bash
# Clone the repository
git clone https://github.com/amari-project/amari.git
cd amari

# Physics simulations
cargo run --bin rigid_body_dynamics
cargo run --bin electromagnetic_fields
cargo run --bin fluid_dynamics
cargo run --bin quantum_mechanics

# Computer graphics
cargo run --bin 3d_transformations
cargo run --bin camera_projection
cargo run --bin mesh_operations
cargo run --bin ray_tracing

# Machine learning
cargo run --bin automatic_differentiation
cargo run --bin neural_networks
cargo run --bin optimization_algorithms
cargo run --bin verified_learning

# Interactive web demos
cd examples/web/interactive-demos
npm install
npm run dev
# Open http://localhost:3000
```

## 📚 Educational Pathways

We've designed structured learning paths for different backgrounds:

### 🌱 [Beginner Track](LEARNING_PATHS.md#beginner-track-foundations)
- Start with basic GA concepts
- Learn dual number fundamentals
- Build first applications
- *Duration: 2-3 weeks*

### 🚀 [Intermediate Track](LEARNING_PATHS.md#intermediate-track-applications)
- Apply GA to real problems
- Master advanced techniques
- Create integration projects
- *Duration: 3-4 weeks*

### TARGET [Advanced Track](LEARNING_PATHS.md#advanced-track-mastery)
- Research-level applications
- Mathematical verification
- Contribute to the field
- *Duration: 4-6 weeks*

### SCIENTIFIC [Research Track](LEARNING_PATHS.md#research-track-innovation)
- Push theoretical boundaries
- Develop novel applications
- Lead community efforts
- *Duration: Ongoing*

[📖 **View Complete Learning Paths**](LEARNING_PATHS.md)

## TARGET Example Highlights

### Physics: Electromagnetic Unity
```rust
// Electric and magnetic fields as unified multivector
let em_field = ElectromagneticField::new(
    [1000.0, 0.0, 0.0],  // Electric field
    [0.0, 0.001, 0.0]    // Magnetic field
);

// Maxwell's equations become: ∇F = J
let field_multivector = em_field.field_multivector(); // F = E + I·B
```

### Graphics: Gimbal Lock-Free Rotations
```rust
// Smooth rotation interpolation without singularities
let start_rotation = Transform3D::rotate(Vector::e1(), PI/2.0);
let end_rotation = Transform3D::rotate(Vector::e3(), PI/2.0);
let interpolated = start_rotation.interpolate(&end_rotation, 0.5);
```

### Machine Learning: Exact Gradients
```rust
// Automatic differentiation with machine precision
let x = Dual::variable(2.0);
let y = x.cube().subtract(&x.square().scale(2.0)); // f(x) = x³ - 2x²
let derivative = y.dual(); // f'(2) = 12 - 8 = 4, exactly
```

## 🛠️ Technical Details

### Architecture
- **Core Library**: `amari-core` - Fundamental GA operations
- **Dual Numbers**: `amari-dual` - Automatic differentiation
- **WASM Bindings**: `amari-wasm` - Web integration
- **Examples**: Comprehensive demonstrations

### Performance Features
- SIMD optimization for geometric products
- Memory-efficient multivector storage
- Parallel computation support
- WebAssembly compilation

### Verification
- Exact arithmetic where possible
- Numerical stability analysis
- Mathematical property verification
- Comprehensive test coverage

## 📊 Comparison with Traditional Methods

| Aspect | Traditional | Amari GA/Dual |
|--------|------------|---------------|
| **3D Rotations** | Euler angles (gimbal lock) | Rotors (singularity-free) |
| **Derivatives** | Finite differences | Exact computation |
| **EM Fields** | Separate E, B vectors | Unified F multivector |
| **Optimization** | Approximate gradients | Machine-precision gradients |
| **Quantum States** | Complex matrices | Natural bivector representation |

## 🎓 Educational Value

### Mathematical Insights
- **Geometric Clarity**: Operations have clear geometric meaning
- **Unified Framework**: Single language for diverse applications
- **Exact Computation**: Eliminate approximation errors
- **Verified Algorithms**: Mathematical guarantees

### Practical Benefits
- **Reduced Bugs**: Type safety prevents geometric errors
- **Better Performance**: Optimized operations
- **Easier Debugging**: Clear mathematical structure
- **Educational**: Learn through interactive exploration

## INTERACTIVE Community & Support

### Getting Help
- 📚 **Documentation**: Comprehensive guides and API docs
- 💬 **Discussions**: GitHub issues and community forums
- 🎥 **Tutorials**: Video walkthroughs (coming soon)
- 👥 **Community**: Join our Discord/Slack channels

### Contributing
We welcome contributions! Areas where you can help:
- 🐛 **Bug Reports**: Find and report issues
- 📝 **Documentation**: Improve explanations
- 🚀 **Examples**: Add new demonstrations
- SCIENTIFIC **Research**: Contribute novel applications

### Citing Amari
If you use Amari in research or education:
```bibtex
@software{amari2024,
  title={Amari: Geometric Algebra and Dual Numbers for Verified Computing},
  author={Amari Project Contributors},
  year={2024},
  url={https://github.com/amari-project/amari}
}
```

## 🗺️ Roadmap

### Short Term (v0.7-0.8)
- [ ] Additional physics simulations
- [ ] Advanced optimization algorithms
- [ ] Performance benchmarks
- [ ] Mobile-friendly web demos

### Medium Term (v0.9-1.0)
- [ ] GPU acceleration
- [ ] Distributed computing support
- [ ] Advanced visualization tools
- [ ] Educational platform integration

### Long Term (v1.0+)
- [ ] Production-ready applications
- [ ] Industry partnerships
- [ ] Academic course integration
- [ ] Standard library inclusion

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Ready to explore?** Start with our [🌱 Beginner Track](LEARNING_PATHS.md#beginner-track-foundations) or dive into [INTERACTIVE Interactive Demos](web/interactive-demos/index.html)!

*Amari: Where mathematics meets intuition, and computation becomes art.* ✨