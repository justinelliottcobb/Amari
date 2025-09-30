#!/bin/bash

# Setup script for Amari git hooks
# Run this once after cloning the repository to enable test enforcement

set -e

echo "🔧 Setting up Amari git hooks for mathematical correctness..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Configure git to use our custom hooks directory
git config core.hooksPath .githooks

# Make sure hooks are executable
chmod +x .githooks/pre-commit

echo "✅ Git hooks configured successfully!"
echo ""
echo "📋 What this enables:"
echo "  • 🧪 All tests must pass before commits"
echo "  • 🔬 Formal verification tests enforced"
echo "  • 📏 Code formatting validation"
echo "  • 🔍 Clippy lints must pass"
echo "  • 📚 Documentation must build"
echo ""
echo "🎯 This ensures mathematical correctness in all commits"
echo "💡 To temporarily bypass (emergency only): git commit --no-verify"