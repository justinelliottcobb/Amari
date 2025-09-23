#!/bin/bash

# Add publishing metadata to all crates

add_metadata() {
    local crate_dir=$1
    local description=$2
    local keywords=$3

    if [ -f "$crate_dir/Cargo.toml" ]; then
        echo "Adding metadata to $crate_dir..."

        # Check if already has description
        if ! grep -q "description" "$crate_dir/Cargo.toml"; then
            # Find the license line and add metadata after it
            sed -i "/license.workspace = true/a\\
description = \"$description\"\\
repository = \"https://github.com/justinelliottcobb/Amari\"\\
homepage = \"https://github.com/justinelliottcobb/Amari\"\\
keywords = $keywords\\
categories = [\"mathematics\", \"science\", \"algorithms\"]" "$crate_dir/Cargo.toml"
        fi
    fi
}

# Add metadata to each crate
add_metadata "amari-tropical" "Tropical (max-plus) algebra implementation" "[\"tropical-algebra\", \"max-plus\", \"semiring\", \"mathematics\", \"optimization\"]"

add_metadata "amari-dual" "Dual number automatic differentiation" "[\"automatic-differentiation\", \"dual-numbers\", \"calculus\", \"mathematics\", \"derivatives\"]"

add_metadata "amari-info-geom" "Information geometry and statistical manifolds" "[\"information-geometry\", \"statistics\", \"manifolds\", \"fisher-metric\", \"mathematics\"]"

add_metadata "amari-fusion" "Fusion system for combining algebraic structures" "[\"fusion-system\", \"algebraic-structures\", \"mathematics\", \"composition\", \"category-theory\"]"

add_metadata "amari-automata" "Cellular automata with geometric constraints" "[\"cellular-automata\", \"geometric-constraints\", \"computation\", \"complexity\", \"mathematics\"]"

add_metadata "amari-gpu" "GPU acceleration for mathematical computations" "[\"gpu-computing\", \"webgpu\", \"parallel\", \"performance\", \"mathematics\"]"

echo "✅ Metadata added to all crates"