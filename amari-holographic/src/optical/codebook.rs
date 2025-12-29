//! Codebook for mapping symbols to optical rotor fields.
//!
//! Codebooks are deterministically generated from seeds, enabling
//! regeneration without storing full fields.

use super::rotor_field::OpticalRotorField;
use alloc::collections::BTreeMap;
use alloc::string::String;
use core::hash::{Hash, Hasher};

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Unique identifier for a symbol in the VSA.
///
/// Symbols are named entities that can be bound, bundled, and compared
/// in the vector symbolic architecture.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct SymbolId(pub String);

impl SymbolId {
    /// Create a new symbol ID from a string.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the symbol name as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for SymbolId {
    fn from(s: &str) -> Self {
        Self(s.into())
    }
}

impl From<String> for SymbolId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl core::fmt::Display for SymbolId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Deterministic codebook generation parameters.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct CodebookConfig {
    /// Grid dimensions for all fields.
    pub dimensions: (usize, usize),
    /// Base seed for deterministic generation.
    ///
    /// Symbol-specific seeds are derived from this base seed
    /// combined with the symbol name.
    pub base_seed: u64,
}

impl CodebookConfig {
    /// Create a new configuration.
    pub fn new(dimensions: (usize, usize), base_seed: u64) -> Self {
        Self {
            dimensions,
            base_seed,
        }
    }
}

/// Maps symbols to optical rotor fields.
///
/// Codebooks are deterministically generated from seeds, enabling
/// regeneration without storing full fields. This is crucial for
/// systems like Minuet that need compact checkpointing.
///
/// # Seed-Based Generation
///
/// Each symbol is associated with a seed derived from the base seed
/// and the symbol name. This ensures:
/// - Identical fields are generated for the same symbol across sessions
/// - Compact serialization (only store seeds, not full fields)
/// - Lazy generation (compute fields only when needed)
///
/// # Example
///
/// ```ignore
/// use amari_holographic::optical::{OpticalCodebook, CodebookConfig, SymbolId};
///
/// let config = CodebookConfig::new((64, 64), 12345);
/// let mut codebook = OpticalCodebook::new(config);
///
/// // Register symbols
/// codebook.register("AGENT".into());
/// codebook.register("ACTION".into());
///
/// // Get field (generated on first access, cached thereafter)
/// let agent_field = codebook.get(&"AGENT".into()).unwrap();
/// ```
pub struct OpticalCodebook {
    config: CodebookConfig,
    /// Symbol to seed mapping (seed determines field)
    seeds: BTreeMap<SymbolId, u64>,
    /// Cached generated fields
    cache: BTreeMap<SymbolId, OpticalRotorField>,
}

impl OpticalCodebook {
    /// Create a new empty codebook with the given configuration.
    pub fn new(config: CodebookConfig) -> Self {
        Self {
            config,
            seeds: BTreeMap::new(),
            cache: BTreeMap::new(),
        }
    }

    /// Register a symbol with a specific seed.
    ///
    /// Overwrites any existing registration for this symbol.
    /// Invalidates cached field for this symbol.
    pub fn register_with_seed(&mut self, symbol: SymbolId, seed: u64) {
        self.seeds.insert(symbol.clone(), seed);
        self.cache.remove(&symbol);
    }

    /// Register a symbol with auto-generated seed.
    ///
    /// The seed is derived deterministically from the base seed
    /// and the symbol name.
    pub fn register(&mut self, symbol: SymbolId) {
        let seed = self.generate_seed(&symbol);
        self.register_with_seed(symbol, seed);
    }

    /// Register multiple symbols at once.
    pub fn register_all(&mut self, symbols: impl IntoIterator<Item = SymbolId>) {
        for symbol in symbols {
            self.register(symbol);
        }
    }

    /// Get or generate field for a symbol.
    ///
    /// Returns `None` if the symbol is not registered.
    /// On first access, generates and caches the field.
    pub fn get(&mut self, symbol: &SymbolId) -> Option<&OpticalRotorField> {
        if !self.seeds.contains_key(symbol) {
            return None;
        }

        if !self.cache.contains_key(symbol) {
            let seed = self.seeds[symbol];
            let field = OpticalRotorField::random(self.config.dimensions, seed);
            self.cache.insert(symbol.clone(), field);
        }

        self.cache.get(symbol)
    }

    /// Get field without caching (always regenerates).
    ///
    /// Useful when memory is constrained.
    pub fn generate(&self, symbol: &SymbolId) -> Option<OpticalRotorField> {
        self.seeds
            .get(symbol)
            .map(|&seed| OpticalRotorField::random(self.config.dimensions, seed))
    }

    /// Check if a symbol is registered.
    pub fn contains(&self, symbol: &SymbolId) -> bool {
        self.seeds.contains_key(symbol)
    }

    /// Get the seed for a registered symbol.
    pub fn get_seed(&self, symbol: &SymbolId) -> Option<u64> {
        self.seeds.get(symbol).copied()
    }

    /// Number of registered symbols.
    pub fn len(&self) -> usize {
        self.seeds.len()
    }

    /// Check if codebook is empty.
    pub fn is_empty(&self) -> bool {
        self.seeds.is_empty()
    }

    /// Iterate over registered symbol IDs.
    pub fn symbols(&self) -> impl Iterator<Item = &SymbolId> {
        self.seeds.keys()
    }

    /// Clear the field cache (seeds are retained).
    ///
    /// Useful for freeing memory while keeping symbol registrations.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Remove a symbol from the codebook.
    pub fn remove(&mut self, symbol: &SymbolId) -> bool {
        self.cache.remove(symbol);
        self.seeds.remove(symbol).is_some()
    }

    /// Export seeds for persistence.
    ///
    /// This is the minimal data needed to reconstruct the codebook.
    pub fn export_seeds(&self) -> BTreeMap<SymbolId, u64> {
        self.seeds.clone()
    }

    /// Import seeds (for restoration).
    ///
    /// Clears existing registrations and cache.
    pub fn import_seeds(&mut self, seeds: BTreeMap<SymbolId, u64>) {
        self.seeds = seeds;
        self.cache.clear();
    }

    /// Get the configuration.
    pub fn config(&self) -> &CodebookConfig {
        &self.config
    }

    /// Generate a deterministic seed from base seed and symbol name.
    fn generate_seed(&self, symbol: &SymbolId) -> u64 {
        // Use a simple hash-based seed generation
        let mut hasher = SimpleHasher::new();
        self.config.base_seed.hash(&mut hasher);
        symbol.0.hash(&mut hasher);
        hasher.finish()
    }
}

/// Simple hasher for deterministic seed generation.
///
/// We use a custom hasher to ensure determinism across platforms.
struct SimpleHasher {
    state: u64,
}

impl SimpleHasher {
    fn new() -> Self {
        Self {
            state: 0xcbf29ce484222325,
        } // FNV offset basis
    }
}

impl Hasher for SimpleHasher {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= byte as u64;
            self.state = self.state.wrapping_mul(0x100000001b3); // FNV prime
        }
    }
}

#[cfg(feature = "serialize")]
impl Serialize for OpticalCodebook {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("OpticalCodebook", 2)?;
        state.serialize_field("config", &self.config)?;

        // Convert BTreeMap to Vec of tuples for serialization
        let seeds_vec: Vec<_> = self.seeds.iter().collect();
        state.serialize_field("seeds", &seeds_vec)?;
        state.end()
    }
}

#[cfg(feature = "serialize")]
impl<'de> Deserialize<'de> for OpticalCodebook {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct CodebookData {
            config: CodebookConfig,
            seeds: Vec<(SymbolId, u64)>,
        }

        let data = CodebookData::deserialize(deserializer)?;
        Ok(Self {
            config: data.config,
            seeds: data.seeds.into_iter().collect(),
            cache: BTreeMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_id() {
        let id1 = SymbolId::new("AGENT");
        let id2: SymbolId = "AGENT".into();
        let id3: SymbolId = String::from("AGENT").into();

        assert_eq!(id1, id2);
        assert_eq!(id2, id3);
        assert_eq!(id1.as_str(), "AGENT");
    }

    #[test]
    fn test_codebook_registration() {
        let config = CodebookConfig::new((32, 32), 12345);
        let mut codebook = OpticalCodebook::new(config);

        assert!(codebook.is_empty());

        codebook.register("AGENT".into());
        codebook.register("ACTION".into());

        assert_eq!(codebook.len(), 2);
        assert!(codebook.contains(&"AGENT".into()));
        assert!(codebook.contains(&"ACTION".into()));
        assert!(!codebook.contains(&"UNKNOWN".into()));
    }

    #[test]
    fn test_codebook_deterministic() {
        let config = CodebookConfig::new((64, 64), 12345);

        let mut codebook1 = OpticalCodebook::new(config.clone());
        let mut codebook2 = OpticalCodebook::new(config);

        codebook1.register("AGENT".into());
        codebook2.register("AGENT".into());

        let field1 = codebook1.get(&"AGENT".into()).unwrap();
        let field2 = codebook2.get(&"AGENT".into()).unwrap();

        // Should be identical
        assert_eq!(field1.scalars(), field2.scalars());
        assert_eq!(field1.bivectors(), field2.bivectors());
    }

    #[test]
    fn test_codebook_different_symbols() {
        let config = CodebookConfig::new((64, 64), 12345);
        let mut codebook = OpticalCodebook::new(config);

        codebook.register("AGENT".into());
        codebook.register("ACTION".into());

        let field1 = codebook.get(&"AGENT".into()).unwrap().clone();
        let field2 = codebook.get(&"ACTION".into()).unwrap();

        // Should be different
        assert_ne!(field1.scalars(), field2.scalars());
    }

    #[test]
    fn test_codebook_caching() {
        let config = CodebookConfig::new((32, 32), 12345);
        let mut codebook = OpticalCodebook::new(config);

        codebook.register("AGENT".into());

        // First access generates and caches
        let field1_ptr = codebook.get(&"AGENT".into()).unwrap() as *const _;

        // Second access should return cached
        let field2_ptr = codebook.get(&"AGENT".into()).unwrap() as *const _;

        assert_eq!(field1_ptr, field2_ptr);
    }

    #[test]
    fn test_codebook_clear_cache() {
        let config = CodebookConfig::new((32, 32), 12345);
        let mut codebook = OpticalCodebook::new(config);

        codebook.register("AGENT".into());
        let _ = codebook.get(&"AGENT".into()); // Populate cache

        codebook.clear_cache();

        // Symbol should still be registered
        assert!(codebook.contains(&"AGENT".into()));

        // But cache should be cleared (new pointer on next access)
        let _ = codebook.get(&"AGENT".into());
    }

    #[test]
    fn test_export_import_seeds() {
        let config = CodebookConfig::new((32, 32), 12345);
        let mut codebook1 = OpticalCodebook::new(config.clone());

        codebook1.register("AGENT".into());
        codebook1.register("ACTION".into());
        let _ = codebook1.get(&"AGENT".into()); // Populate cache

        // Export
        let seeds = codebook1.export_seeds();

        // Import into fresh codebook
        let mut codebook2 = OpticalCodebook::new(config);
        codebook2.import_seeds(seeds);

        // Should regenerate same fields
        let field1 = codebook1.generate(&"AGENT".into()).unwrap();
        let field2 = codebook2.get(&"AGENT".into()).unwrap();

        assert_eq!(field1.scalars(), field2.scalars());
    }

    #[test]
    fn test_register_with_seed() {
        let config = CodebookConfig::new((32, 32), 12345);
        let mut codebook = OpticalCodebook::new(config);

        codebook.register_with_seed("CUSTOM".into(), 99999);

        assert_eq!(codebook.get_seed(&"CUSTOM".into()), Some(99999));
    }

    #[test]
    fn test_remove() {
        let config = CodebookConfig::new((32, 32), 12345);
        let mut codebook = OpticalCodebook::new(config);

        codebook.register("AGENT".into());
        assert!(codebook.contains(&"AGENT".into()));

        assert!(codebook.remove(&"AGENT".into()));
        assert!(!codebook.contains(&"AGENT".into()));

        // Removing non-existent returns false
        assert!(!codebook.remove(&"UNKNOWN".into()));
    }

    #[test]
    fn test_symbols_iterator() {
        let config = CodebookConfig::new((32, 32), 12345);
        let mut codebook = OpticalCodebook::new(config);

        codebook.register("ALPHA".into());
        codebook.register("BETA".into());
        codebook.register("GAMMA".into());

        let symbols: Vec<_> = codebook.symbols().collect();
        assert_eq!(symbols.len(), 3);

        // BTreeMap is ordered, so symbols should be sorted
        assert_eq!(symbols[0].as_str(), "ALPHA");
        assert_eq!(symbols[1].as_str(), "BETA");
        assert_eq!(symbols[2].as_str(), "GAMMA");
    }
}
