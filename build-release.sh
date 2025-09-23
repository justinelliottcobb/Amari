#!/bin/bash

# Build Release Script for Amari Mathematical Computing Library
# This script prepares the entire library for release

set -e

echo "🚀 Building Amari Mathematical Computing Library for Release"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ Cargo not found. Please install Rust.${NC}"
    exit 1
fi

if ! command -v wasm-pack &> /dev/null; then
    echo -e "${YELLOW}wasm-pack not found. Installing...${NC}"
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Add wasm target if not present
rustup target add wasm32-unknown-unknown

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
cargo clean

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
cargo test --workspace --quiet || {
    echo -e "${RED}❌ Tests failed${NC}"
    exit 1
}
echo -e "${GREEN}✅ All tests passed${NC}"

# Run clippy
echo -e "${YELLOW}Running clippy...${NC}"
cargo clippy --workspace -- -D warnings 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Clippy warnings detected (non-blocking)${NC}"
}

# Check formatting
echo -e "${YELLOW}Checking formatting...${NC}"
cargo fmt --all -- --check || {
    echo -e "${YELLOW}⚠️  Format issues detected. Run 'cargo fmt --all' to fix${NC}"
}

# Build all crates
echo -e "${YELLOW}Building all crates...${NC}"
cargo build --workspace --release
echo -e "${GREEN}✅ All crates built successfully${NC}"

# Build WASM packages
echo -e "${YELLOW}Building WASM packages...${NC}"
cd amari-wasm

# Build for different targets
echo "  Building for web..."
wasm-pack build --target web --out-dir pkg --scope amari

echo "  Building for Node.js..."
wasm-pack build --target nodejs --out-dir pkg-node --scope amari

echo "  Building for bundlers..."
wasm-pack build --target bundler --out-dir pkg-bundler --scope amari

echo -e "${GREEN}✅ WASM packages built successfully${NC}"

# Package size report
echo -e "${YELLOW}Package sizes:${NC}"
if [ -f "pkg/amari_wasm_bg.wasm" ]; then
    wasm_size=$(du -h pkg/amari_wasm_bg.wasm | cut -f1)
    echo "  WASM binary: $wasm_size"
fi

cd ..

# Generate documentation
echo -e "${YELLOW}Generating documentation...${NC}"
cargo doc --workspace --no-deps

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✅ Build completed successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Update version numbers in Cargo.toml and package.json"
echo "  2. Commit changes: git add . && git commit -m 'chore: prepare release'"
echo "  3. Create version tag: git tag v0.1.0"
echo "  4. Push tag to trigger release: git push origin v0.1.0"
echo ""
echo "Or trigger manual release from GitHub Actions."