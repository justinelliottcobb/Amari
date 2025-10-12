#!/bin/bash

# Version Synchronization Script for Amari
# This script ensures all packages (Rust workspace + WASM package) stay synchronized

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default files to update
CARGO_TOML="Cargo.toml"
PACKAGE_JSON="amari-wasm/package.json"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to validate semantic version format
validate_version() {
    local version=$1
    if [[ ! $version =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9-]+)?(\+[a-zA-Z0-9-]+)?$ ]]; then
        print_error "Invalid version format: $version"
        print_error "Expected format: MAJOR.MINOR.PATCH (e.g., 1.2.3, 1.2.3-alpha, 1.2.3+build)"
        return 1
    fi
    return 0
}

# Function to get current workspace version from Cargo.toml
get_current_version() {
    if [[ ! -f "$CARGO_TOML" ]]; then
        print_error "Cargo.toml not found in current directory"
        return 1
    fi

    grep -E "^version = " "$CARGO_TOML" | head -1 | sed 's/version = "\(.*\)"/\1/'
}

# Function to get current package.json version
get_package_json_version() {
    if [[ ! -f "$PACKAGE_JSON" ]]; then
        print_error "package.json not found at $PACKAGE_JSON"
        return 1
    fi

    # Look for the main version field (not in scripts section)
    grep -E '^\s*"version":' "$PACKAGE_JSON" | head -1 | sed 's/.*"version": "\(.*\)".*/\1/'
}

# Function to update workspace version in Cargo.toml
update_cargo_version() {
    local new_version=$1

    print_status "Updating Cargo.toml workspace version to $new_version..."

    # Update workspace.package version
    sed -i.bak "s/^version = \".*\"/version = \"$new_version\"/" "$CARGO_TOML"

    # Update all workspace.dependencies versions
    sed -i.bak "s/version = \"[0-9]\+\.[0-9]\+\.[0-9]\+\"/version = \"$new_version\"/g" "$CARGO_TOML"

    # Remove backup file
    rm -f "$CARGO_TOML.bak"

    print_success "Updated Cargo.toml versions"
}

# Function to update package.json version
update_package_json_version() {
    local new_version=$1

    print_status "Updating package.json version to $new_version..."

    # Only update the main version field (not in scripts section)
    sed -i.bak "/^\s*\"version\":/s/\"version\": \".*\"/\"version\": \"$new_version\"/" "$PACKAGE_JSON"

    # Remove backup file
    rm -f "$PACKAGE_JSON.bak"

    print_success "Updated package.json version"
}

# Function to verify all versions are synchronized
verify_versions() {
    local expected_version=$1

    print_status "Verifying version synchronization..."

    # Check Cargo.toml workspace version
    local cargo_version=$(grep -E "^version = " "$CARGO_TOML" | head -1 | sed 's/version = "\(.*\)"/\1/')
    if [[ "$cargo_version" != "$expected_version" ]]; then
        print_error "Cargo.toml workspace version mismatch: expected $expected_version, found $cargo_version"
        return 1
    fi

    # Check package.json version
    local package_version=$(get_package_json_version)
    if [[ "$package_version" != "$expected_version" ]]; then
        print_error "package.json version mismatch: expected $expected_version, found $package_version"
        return 1
    fi

    # Check that all workspace dependencies have the correct version
    local workspace_deps_count=$(grep -c "version = \"$expected_version\"" "$CARGO_TOML" || true)
    if [[ $workspace_deps_count -lt 10 ]]; then  # We have 11+ internal crates
        print_warning "Some workspace dependencies may not be updated (found $workspace_deps_count matches)"
    fi

    print_success "All versions synchronized to $expected_version"
    return 0
}

# Function to show current version status
show_status() {
    print_status "Current version status:"

    if [[ -f "$CARGO_TOML" ]]; then
        local cargo_version=$(grep -E "^version = " "$CARGO_TOML" | head -1 | sed 's/version = "\(.*\)"/\1/' || echo "NOT_FOUND")
        echo "  Cargo.toml workspace: $cargo_version"
    else
        echo "  Cargo.toml: NOT_FOUND"
    fi

    if [[ -f "$PACKAGE_JSON" ]]; then
        local package_version=$(get_package_json_version || echo "NOT_FOUND")
        echo "  package.json: $package_version"
    else
        echo "  package.json: NOT_FOUND"
    fi

    # Show workspace dependencies versions
    echo "  Workspace dependencies:"
    grep -E "version = \"[0-9]" "$CARGO_TOML" | head -5 | sed 's/^/    /'
    if [[ $(grep -c "version = \"[0-9]" "$CARGO_TOML") -gt 5 ]]; then
        echo "    ... ($(grep -c "version = \"[0-9]" "$CARGO_TOML") total)"
    fi
}

# Function to bump version automatically
bump_version() {
    local bump_type=$1
    local current_version=$(get_current_version)

    if [[ -z "$current_version" ]]; then
        print_error "Could not determine current version"
        return 1
    fi

    # Parse current version
    IFS='.' read -r major minor patch <<< "$current_version"

    case $bump_type in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        *)
            print_error "Invalid bump type: $bump_type (use: major, minor, patch)"
            return 1
            ;;
    esac

    local new_version="$major.$minor.$patch"
    echo "$new_version"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
    status                  Show current version status
    set <version>          Set specific version (e.g., 1.0.0)
    bump <major|minor|patch>  Bump version automatically
    verify <version>       Verify all versions match expected version

Examples:
    $0 status              # Show current versions
    $0 set 1.0.0          # Set all versions to 1.0.0
    $0 bump minor         # Bump minor version (1.0.0 -> 1.1.0)
    $0 verify 1.0.0       # Check if all versions are 1.0.0

This script synchronizes versions between:
- Cargo.toml workspace.package.version
- Cargo.toml workspace.dependencies versions
- amari-wasm/package.json version
EOF
}

# Main script logic
main() {
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 1
    fi

    local command=$1

    case $command in
        status)
            show_status
            ;;
        set)
            if [[ $# -ne 2 ]]; then
                print_error "Usage: $0 set <version>"
                exit 1
            fi
            local new_version=$2
            if ! validate_version "$new_version"; then
                exit 1
            fi

            print_status "Setting all versions to $new_version"
            update_cargo_version "$new_version"
            update_package_json_version "$new_version"
            verify_versions "$new_version"
            ;;
        bump)
            if [[ $# -ne 2 ]]; then
                print_error "Usage: $0 bump <major|minor|patch>"
                exit 1
            fi
            local bump_type=$2
            local new_version=$(bump_version "$bump_type")
            if [[ $? -ne 0 ]]; then
                exit 1
            fi

            print_status "Bumping $bump_type version to $new_version"
            update_cargo_version "$new_version"
            update_package_json_version "$new_version"
            verify_versions "$new_version"
            ;;
        verify)
            if [[ $# -ne 2 ]]; then
                print_error "Usage: $0 verify <version>"
                exit 1
            fi
            local expected_version=$2
            if ! validate_version "$expected_version"; then
                exit 1
            fi
            verify_versions "$expected_version"
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"