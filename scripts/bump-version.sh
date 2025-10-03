#!/bin/bash
set -e

# Script to bump version numbers across the entire Amari workspace
# Usage: ./scripts/bump-version.sh <new_version>
# Example: ./scripts/bump-version.sh 0.6.1

if [ $# -ne 1 ]; then
    echo "Usage: $0 <new_version>"
    echo "Example: $0 0.6.1"
    exit 1
fi

NEW_VERSION="$1"

# Validate version format (basic check)
if ! echo "$NEW_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-.*)?$'; then
    echo "Error: Invalid version format. Expected format: X.Y.Z or X.Y.Z-suffix"
    exit 1
fi

echo "Bumping version to $NEW_VERSION across the workspace..."

# Extract major.minor for dependency specifications (e.g., 0.6 from 0.6.1)
MAJOR_MINOR=$(echo "$NEW_VERSION" | sed 's/\([0-9]*\.[0-9]*\).*/\1/')

# Update root workspace version
echo "Updating root Cargo.toml..."
sed -i "s/^version = \"[^\"]*\"/version = \"$NEW_VERSION\"/" Cargo.toml

# Update all workspace member Cargo.toml files
for crate_dir in amari-*; do
    if [ -d "$crate_dir" ] && [ -f "$crate_dir/Cargo.toml" ]; then
        echo "Updating $crate_dir/Cargo.toml..."

        # Update inter-crate dependencies to use the new major.minor version
        sed -i "s/\(amari-[a-z-]* = {.*version = \"\)[^\"]*/\1$MAJOR_MINOR/" "$crate_dir/Cargo.toml"
    fi
done

# Update root Cargo.toml dependencies
echo "Updating root dependencies..."
sed -i "s/\(amari-[a-z-]* = {.*version = \"\)[^\"]*/\1$MAJOR_MINOR/" Cargo.toml

# Update package.json files if they exist
if [ -f "typescript/package.json" ]; then
    echo "Updating typescript/package.json..."
    sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"$NEW_VERSION\"/" typescript/package.json
fi

if [ -f "amari-wasm/pkg/package.json" ]; then
    echo "Updating amari-wasm/pkg/package.json..."
    sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"$NEW_VERSION\"/" amari-wasm/pkg/package.json
fi

if [ -f "examples/typescript/package.json" ]; then
    echo "Updating examples/typescript/package.json..."
    sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"$NEW_VERSION\"/" examples/typescript/package.json
fi

if [ -f "examples-suite/package.json" ]; then
    echo "Updating examples-suite/package.json..."
    sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"$NEW_VERSION\"/" examples-suite/package.json
fi

# Update README badges if they contain version numbers
if [ -f "README.md" ]; then
    echo "Updating README.md badges..."
    sed -i "s/\(crates\.io-v\)[0-9.]*\(-\)/\1$NEW_VERSION\2/g" README.md
    sed -i "s/\(npm-v\)[0-9.]*\(-\)/\1$NEW_VERSION\2/g" README.md
fi

# Update CHANGELOG if it exists
if [ -f "CHANGELOG.md" ]; then
    echo "Adding entry to CHANGELOG.md..."
    # Add a new unreleased section if this is a new version
    if ! grep -q "## \[$NEW_VERSION\]" CHANGELOG.md; then
        DATE=$(date +%Y-%m-%d)
        sed -i "s/## \[Unreleased\]/## [Unreleased]\n\n## [$NEW_VERSION] - $DATE/" CHANGELOG.md
    fi
fi

echo ""
echo "✅ Version bumped to $NEW_VERSION successfully!"
echo ""
echo "Summary of changes:"
echo "  - Workspace version: $NEW_VERSION"
echo "  - Dependency versions: $MAJOR_MINOR"
echo ""
echo "Next steps:"
echo "  1. Review the changes: git diff"
echo "  2. Commit the changes: git commit -am \"chore: bump version to $NEW_VERSION\""
echo "  3. Tag the release: git tag v$NEW_VERSION"
echo "  4. Push changes: git push && git push --tags"