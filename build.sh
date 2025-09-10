#!/bin/bash

# Amari Build Script
# Builds the entire Rust workspace and TypeScript package

set -e  # Exit on any error

echo "🚀 Building Amari Geometric Algebra Library"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo not found. Please install Rust: https://rustup.rs/"
        exit 1
    fi
    
    if ! command -v wasm-pack &> /dev/null; then
        print_warning "wasm-pack not found. Installing..."
        curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
    fi
    
    if ! command -v node &> /dev/null; then
        print_warning "Node.js not found. TypeScript build will be skipped."
        SKIP_TS=1
    fi
    
    print_status "Dependencies check complete ✓"
}

# Build Rust workspace
build_rust() {
    print_status "Building Rust workspace..."
    
    # Build all crates in release mode
    cargo build --workspace --release
    
    # Run tests
    print_status "Running Rust tests..."
    cargo test --workspace
    
    print_status "Rust build complete ✓"
}

# Build WASM package
build_wasm() {
    print_status "Building WASM package..."
    
    cd amari-wasm
    
    # Build for web target
    wasm-pack build --target web --out-dir ../typescript/pkg --release
    
    # Build for Node.js target
    wasm-pack build --target nodejs --out-dir ../typescript/pkg-node --release
    
    cd ..
    
    print_status "WASM build complete ✓"
}

# Build TypeScript package
build_typescript() {
    if [[ "$SKIP_TS" == "1" ]]; then
        print_warning "Skipping TypeScript build (Node.js not found)"
        return
    fi
    
    print_status "Building TypeScript package..."
    
    cd typescript
    
    # Install dependencies if needed
    if [[ ! -d "node_modules" ]]; then
        print_status "Installing npm dependencies..."
        npm install
    fi
    
    # Build TypeScript
    npm run build:ts
    
    cd ..
    
    print_status "TypeScript build complete ✓"
}

# Generate documentation
generate_docs() {
    print_status "Generating documentation..."
    
    # Generate Rust docs
    cargo doc --workspace --no-deps --release
    
    print_status "Documentation generated ✓"
}

# Run benchmarks
run_benchmarks() {
    if [[ "$1" == "--bench" ]]; then
        print_status "Running benchmarks..."
        
        # Run core benchmarks
        cd amari-core
        cargo bench
        cd ..
        
        # Run GPU benchmarks if available
        if [[ -f "amari-gpu/Cargo.toml" ]]; then
            cd amari-gpu
            cargo bench
            cd ..
        fi
        
        print_status "Benchmarks complete ✓"
    fi
}

# Run examples
run_examples() {
    if [[ "$1" == "--examples" ]]; then
        print_status "Running examples..."
        
        # Run Rust example
        cargo run --example basic --release
        
        # Run TypeScript examples if available
        if [[ "$SKIP_TS" != "1" ]] && [[ -f "typescript/dist/examples.js" ]]; then
            cd typescript
            node dist/examples.js
            cd ..
        fi
        
        print_status "Examples complete ✓"
    fi
}

# Clean build artifacts
clean() {
    if [[ "$1" == "--clean" ]]; then
        print_status "Cleaning build artifacts..."
        
        cargo clean
        rm -rf typescript/dist
        rm -rf typescript/pkg
        rm -rf typescript/pkg-node
        rm -rf typescript/node_modules
        
        print_status "Clean complete ✓"
    fi
}

# Main build process
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean)
                clean "$1"
                exit 0
                ;;
            --bench)
                BENCH=1
                shift
                ;;
            --examples)
                EXAMPLES=1
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --clean     Clean all build artifacts"
                echo "  --bench     Run benchmarks after building"
                echo "  --examples  Run examples after building"
                echo "  --help      Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Start timer
    start_time=$(date +%s)
    
    # Run build steps
    check_dependencies
    build_rust
    build_wasm
    build_typescript
    generate_docs
    
    # Optional steps
    if [[ "$BENCH" == "1" ]]; then
        run_benchmarks "--bench"
    fi
    
    if [[ "$EXAMPLES" == "1" ]]; then
        run_examples "--examples"
    fi
    
    # Calculate build time
    end_time=$(date +%s)
    build_time=$((end_time - start_time))
    
    echo ""
    print_status "🎉 Build completed successfully in ${build_time}s!"
    echo ""
    echo "📁 Output directories:"
    echo "   • target/release/          - Rust binaries and libraries"
    echo "   • target/doc/              - Rust documentation"
    echo "   • typescript/pkg/          - WASM package for web"
    echo "   • typescript/pkg-node/     - WASM package for Node.js"
    echo "   • typescript/dist/         - TypeScript compiled output"
    echo ""
    echo "🧪 To run tests:    cargo test --workspace"
    echo "📊 To run benchmarks: ./build.sh --bench"
    echo "🔍 To view docs:    cargo doc --open"
    echo "🏃 To run examples: ./build.sh --examples"
}

# Run main function
main "$@"