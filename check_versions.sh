#!/bin/bash

for crate in amari amari-core amari-tropical amari-dual amari-fusion amari-info-geom amari-automata amari-enumerative amari-relativistic amari-gpu amari-network amari-optimization; do
  version=$(curl -s "https://crates.io/api/v1/crates/$crate" | jq -r '.crate.max_version // "NOT FOUND"')
  echo "$crate: $version"
done
