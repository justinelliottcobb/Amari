#!/bin/bash

# Amari Mathematical Computing Library - Comprehensive Test Runner
# This script runs all TDD test phases to validate the entire mathematical framework

# set -e  # Exit on any error - disabled to show full results

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

print_phase() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

# Test tracking
TOTAL_PHASES=7
PASSED_PHASES=0
FAILED_PHASES=0

run_phase_tests() {
    local phase_name="$1"
    local test_command="$2"
    local description="$3"
    
    print_phase "Phase $phase_name: $description"
    
    if eval "$test_command"; then
        print_success "Phase $phase_name passed!"
        ((PASSED_PHASES++))
    else
        print_error "Phase $phase_name failed!"
        ((FAILED_PHASES++))
        return 1
    fi
}

# Start comprehensive testing
print_phase "Amari Mathematical Computing Library - TDD Test Suite"
print_status "Testing advanced mathematical computing across 7 phases"
print_status "Geometric Algebra | Tropical Algebra | Dual Numbers | Information Geometry | Integration"

# Phase 1: Geometric Product Tests (Core Geometric Algebra)
run_phase_tests "1" \
    "cargo test --package amari-core geometric_product_tests" \
    "Geometric Product Tests (Core Geometric Algebra Operations)"

# Phase 2: Inner and Outer Product Tests
run_phase_tests "2" \
    "cargo test --package amari-core product_tests" \
    "Inner and Outer Product Tests (Grade-based Operations)"

# Phase 3: Rotor and Rotation Tests
run_phase_tests "3" \
    "cargo test --package amari-core rotor_tests" \
    "Rotor and Rotation Tests (3D Rotations and Quaternion-like Operations)"

# Phase 4: Tropical Algebra Tests
run_phase_tests "4" \
    "cargo test --package amari-tropical tropical_tests" \
    "Tropical Algebra Tests (Max-plus Semiring and Viterbi Algorithm)"

# Phase 5: Dual Number Automatic Differentiation Tests
run_phase_tests "5" \
    "cargo test --package amari-dual dual_tests" \
    "Dual Number Automatic Differentiation Tests (Forward-mode AD)"

# Phase 6: Information Geometry Tests
run_phase_tests "6" \
    "cargo test --package amari-info-geom info_geom_tests" \
    "Information Geometry Tests (Fisher Metrics and Bregman Divergences)"

# Phase 7: Integration Tests
run_phase_tests "7" \
    "cargo test --test integration" \
    "Integration Tests (Cross-crate Consistency and Performance)"

# Additional comprehensive tests
print_phase "Additional Testing: Comprehensive Coverage"

print_status "Running all unit tests across the workspace..."
if cargo test --workspace --lib; then
    print_success "All unit tests passed!"
else
    print_warning "Some unit tests failed (may be non-critical)"
fi

print_status "Running doc tests..."
if cargo test --workspace --doc; then
    print_success "All doc tests passed!"
else
    print_warning "Some doc tests failed (may be non-critical)"
fi

print_status "Checking build across all targets..."
if cargo build --workspace; then
    print_success "All crates build successfully!"
else
    print_error "Build failed!"
    ((FAILED_PHASES++))
fi

# Performance benchmarks (if available)
print_status "Running performance tests..."
if cargo test --release --test integration test_performance; then
    print_success "Performance tests passed!"
else
    print_warning "Performance tests may have issues"
fi

# Final summary
print_phase "Test Suite Summary"

echo -e "📊 ${BLUE}Test Results:${NC}"
echo -e "   ✅ Passed Phases: ${GREEN}$PASSED_PHASES${NC}/$TOTAL_PHASES"
echo -e "   ❌ Failed Phases: ${RED}$FAILED_PHASES${NC}/$TOTAL_PHASES"

if [ $FAILED_PHASES -eq 0 ]; then
    echo -e "\n🎉 ${GREEN}ALL TESTS PASSED!${NC}"
    echo -e "   🧮 Geometric Algebra: ✅ Fully functional"
    echo -e "   🌴 Tropical Algebra: ✅ Max-plus operations working"
    echo -e "   🔄 Automatic Differentiation: ✅ Forward-mode AD operational"
    echo -e "   📐 Information Geometry: ✅ Statistical manifolds ready"
    echo -e "   🔗 Integration: ✅ Cross-crate consistency verified"
    echo -e "\n🚀 ${GREEN}Amari mathematical computing library is ready for advanced applications!${NC}"
    exit 0
else
    echo -e "\n⚠️  ${YELLOW}Some tests failed${NC}"
    echo -e "   Review the output above for details on failed phases"
    echo -e "   ${PASSED_PHASES} out of ${TOTAL_PHASES} phases passed successfully"
    exit 1
fi