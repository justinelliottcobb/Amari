# Amari 0.25.0 → 1.0.0 Release Sequence

- Date: 2026-08-12
- Current baseline: 0.24.1 shipped; 0.25.0 in flight (rewrite/inverse
  cohorts); this document succeeds
  [`V0_20_0_TO_V0_26_0_RELEASE_SEQUENCE.md`](V0_20_0_TO_V0_26_0_RELEASE_SEQUENCE.md)
  for everything after 0.26.0.

## Release posture

Each minor release has one coherent center of gravity. Patch releases
remain bug-fix only. Structural refactors are sequenced so that audited
code is never rewritten afterward, and so that new subsystems are born
on post-refactor templates rather than migrated.

The sequence is:

1. 0.25.0 — comprehensive rewrite research, including first-class
   inverse rewriting (in flight);
2. 0.25.x — `amari-relativistic` P1 correctness fixes;
3. 0.26.0 — GPU/current-`wgpu`/Borsalino modernization, catalog
   storage mitigation, Insum evaluation;
4. 0.27.0 — `amari-wasm` decomposition **and** `amari-symplectic`
   (one release, deliberately paired);
5. 0.28.0 — `amari-discovery` refactored to depend on Lonis;
6. 0.29.0 — audit cycle and v1 readiness re-baseline;
7. 1.0.0 — stable release.

## 0.25.0 — rewrite/inverse research (in flight)

Unchanged from the prior sequence document. Symbolic relation
authority, constrained backward/bidirectional search, tree automata and
grammar extraction, closure theorems, bounded KB completion,
hypothesis-directed synthesis, Candle/Z3/holographic guidance, and
discovery surfacing. The symbolic relation remains authoritative;
learned/SMT guidance never creates rewrite transitions.

### Post-rewrite discovery cohort (queued)

Queued for the end of the 0.25.0 rewrite cohorts, sourced from
`docs/development/discovery-feedback-2026-08-22-knopper.md`
(practitioner feedback from running amari-discovery against a foreign
workspace):

1. **Two-tier catalog** — curated entries plus an auto-extracted,
   explicitly-unverified public-surface tier, so code that exists but
   is not yet curated (the Knopper run's dominant failure mode) is
   still discoverable with honest confidence labels.
2. **Codebase-aware search fallback** — catalog misses fall back to
   structural source search instead of returning empty.
3. **Runnable probe validation for foreign consumers** — probes
   verify the consumer environment can actually execute what a plan
   recommends.
4. **Catalog entries for the Schubert machinery** — `SchubertCalculus`,
   `schubert_cell_of`, `CayleyTable`, GA3-native Cayley navigation.
5. **Hook friction follow-ups** — a skip path for docs-only commits
   and a scheduled full-clippy cron so develop lint rot surfaces
   continuously instead of tripping unrelated commits.

These items start only after the rewrite/inverse cohorts close; they
are discovery-layer work and do not gate 0.25.0's symbolic-relation
scope.

## 0.25.x — `amari-relativistic` P1 correctness fixes

Theme: known correctness bugs are fixed where they are found, not
deferred to the audit.

Source: `IA-documents/RESEARCH_REPORTS/RABBIT_HOLE_2026-08-12_Starstrider.md`
(the Starstrider integration surfaced both defects with reproduction
cases and velocity-regime matrices):

1. Geodesic adaptive-step collapse on gentle trajectories — the
   stiff-but-gentle regime that covers most solar-system-scale use.
2. Particle internal/coordinate time never advancing during `propagate`
   after classical propagation.

These are behavior corrections (wrong → right) on existing APIs and are
therefore patch-release candidates. If either fix requires an API
change, it moves to 0.26.0 and is called out in that changelog.

## 0.26.0 — GPU modernization (unchanged scope)

Carried forward unchanged: Borsalino integration where measured,
current-`wgpu` migration as a dedicated effort, CPU baselines,
release-mode/Criterion benchmarks, hardware-aware calibrated dispatch,
GB10/RTX 5080 revalidation, and tracked issues #137–#142. Do not
combine the backend migration with unrelated minor-release work.

Additionally in scope:

- **Catalog storage mitigation.** `amari-discovery/catalog/generated.json`
  is ~17.6 MB / ~467k lines of pretty JSON and churns on every
  catalog-touching change. Options recorded 2026-08-12 (agent memory,
  topic `discovery-catalog`): per-crate split, compressed artifact,
  hybrid split+minify, or schema diet (deduplicate `cfg_gates` vs
  `items`). Decision and implementation land here — before the
  Lonis refactor, so the format is not migrated twice.
- **Insum evaluation.** See
  [`INSUM_INDIRECT_EINSUM_FUTURE_RESEARCH.md`](INSUM_INDIRECT_EINSUM_FUTURE_RESEARCH.md):
  indirect-Einsum sparse GPU kernels, evaluated after the Borsalino
  modernization lands. Research thread, not committed scope.

## 0.27.0 — `amari-wasm` decomposition + `amari-symplectic`

Theme: the WASM surface becomes modular, and the newest mathematical
crate is born on the new template.

Primary outcomes:

- Decompose the `amari-wasm` monolith into per-domain binding crates
  (e.g. `amari-wasm-core`, `amari-wasm-tropical`, …) with `amari-wasm`
  retained as a facade so the published npm package surface
  (`@justinelliottcobb/amari-wasm`) is preserved unchanged.
- New `amari-symplectic` crate (symplectic geometry), implemented as
  the **first consumer of the new binding template** — born modular,
  never added to the old monolith.

Rationale for pairing: a brand-new domain crate is the cheapest possible
validation of the binding template. If the template cannot cleanly
absorb symplectic bindings, the template is wrong, and that is
discovered while the facade still works. Splitting these into two
releases remains a fallback if either grows teeth; the default is one
release.

## 0.28.0 — `amari-discovery` on Lonis

Theme: shared protocol authority moves to Lonis.

Precondition: **Lonis ≥ 0.2.0** with the schema-superset contract
proven. Per the locked architecture direction (2026-08-10): Lonis
replaces, generalizes, and extends amari-discovery's machinery;
dependency direction is `amari-discovery → lonis-schema/lonis-core`,
never the reverse; `lonis-schema` types are a superset of
amari-discovery's `protocol.rs` (Envelope, Provenance, replay metadata,
resource limits, schema registry) so that `protocol.rs` is **deleted**
in favor of lonis-schema.

Primary outcomes:

- `amari-discovery` re-exports/uses Lonis protocol and schema types;
  wire compatibility for existing consumers is either preserved or the
  break is explicit and changelogged.
- Catalog/inspection machinery remains in amari-discovery unless a
  decision record moves it.

This release must come after 0.26.0's catalog-storage decision so the
storage format is not migrated twice.

## 0.29.0 — audit cycle and v1 readiness re-baseline

Theme: verify, then freeze.

Primary outcomes:

- Re-baseline `V1_READINESS_PLAN.md` — its gap tables date to v0.17.0
  and predate the 0.20–0.25 workspace. Rewrite it against the current
  crate set with measurable gates.
- `#[ignore]`d-test burn-down. The Starstrider report documents how
  "✅-shaped" features survive when failing suites are disabled:
  ignored suites must be fixed, re-scoped with named owners, or
  honestly removed from feature claims.
- Full correctness audit across the workspace, with `amari-relativistic`
  (post-0.25.x fixes) as the regression exemplar.
- Documentation and benchmark coverage gates per the re-baselined plan.

No structural refactors in this cycle; audited code must not be
rewritten afterward.

## 1.0.0 — stable release

Gates: green 0.29.0 audit with no open P1 correctness issues; the
re-baselined readiness plan fully met; API freeze policy published;
changelog and migration guide for the 0.x → 1.0 transition.

## Explicitly out of scope for this sequence

- GPU training backends; external solver executables (0.25.0 non-goals
  carry forward).
- Puiseux/Hahn asymptotic extensions (possible `amari-asymptotic`
  extraction) — no committed version.
- Morphogen integration beyond the rewrite/inverse substrate it
  motivated; expected to inform post-1.0 priorities.

## Decision log

- 2026-08-12 — sequence drafted and approved in principle by the
  maintainer; wasm decomposition and symplectic confirmed as a single
  paired release; catalog-storage mitigation confirmed for 0.26.0
  (deferred from 2026-08-12 discussion).
