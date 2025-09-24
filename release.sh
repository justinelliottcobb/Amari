#!/bin/bash

# Release script for publishing Amari to both npm and crates.io

set -e

VERSION=${1:-}
PUBLISH_NPM=${2:-true}
PUBLISH_CRATES=${3:-true}

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version> [publish_npm] [publish_crates]"
    echo "Example: $0 0.1.0 true true"
    echo "Example: $0 0.1.1 true false  # Only publish to npm"
    exit 1
fi

echo "🚀 Preparing release $VERSION"

# Validate version format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "❌ Invalid version format. Use semantic versioning (e.g., 0.1.0)"
    exit 1
fi

# Update workspace version
echo "📝 Updating workspace version to $VERSION"
sed -i "0,/^version = /s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml

# Update amari-wasm package.json version
echo "📝 Updating amari-wasm package.json version to $VERSION"
cd amari-wasm
npm version $VERSION --no-git-tag-version
cd ..

# Run tests
echo "🧪 Running tests..."
if ! cargo test --all-features --workspace; then
    echo "❌ Tests failed"
    exit 1
fi

# Check if everything can build
echo "🔨 Building all crates..."
if ! cargo build --all-features --workspace; then
    echo "❌ Build failed"
    exit 1
fi

# Build WASM to ensure it works
echo "🌐 Building WASM..."
cd amari-wasm
if ! npm run build; then
    echo "❌ WASM build failed"
    exit 1
fi
cd ..

echo "✅ All checks passed!"

# Commit version changes
echo "📝 Committing version changes..."
git add -A
git commit -m "🔖 Release version $VERSION

- Update workspace version to $VERSION
- Update amari-wasm package.json version
- Ready for publishing to npm and crates.io

🤖 Generated with release script"

# Create tag
echo "🏷️  Creating tag v$VERSION"
git tag "v$VERSION"

# Push changes and tag
echo "📤 Pushing changes and tag..."
git push origin HEAD
git push origin "v$VERSION"

echo "🎉 Release $VERSION is ready!"
echo ""
echo "Publishing will happen automatically via GitHub Actions:"
echo "  📦 npm: $PUBLISH_NPM"
echo "  📦 crates.io: $PUBLISH_CRATES"
echo ""
echo "Monitor the workflow at: https://github.com/justinelliottcobb/Amari/actions"
echo ""
echo "After publishing, the examples suite will be automatically updated to use the published @amari/core package."