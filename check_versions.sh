#!/bin/bash

# Check published versions on crates.io for all Amari crates
# Order matches publishing order in .github/workflows/publish.yml

echo "Checking published versions on crates.io..."
echo "============================================="
echo ""

for crate in \
  amari-core \
  amari-tropical \
  amari-dual \
  amari-network \
  amari-info-geom \
  amari-relativistic \
  amari-fusion \
  amari-automata \
  amari-enumerative \
  amari-optimization \
  amari-flynn-macros \
  amari-flynn \
  amari-measure \
  amari-wasm \
  amari-gpu \
  amari; do
  version=$(curl -s "https://crates.io/api/v1/crates/$crate" | jq -r '.crate.max_version // "NOT FOUND"')
  echo "$crate: $version"
done

echo ""
echo "============================================="
echo "Note: amari-measure and amari-wasm are new in v0.10.x"
echo "Note: amari-flynn is at v0.1.0 (different versioning)"
