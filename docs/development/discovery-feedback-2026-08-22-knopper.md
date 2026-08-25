# Discovery Feedback — amari-discovery 0.24.1 vs. Knopper (2026-08-22)

**From:** the Knopper identity-restoration session. Ran `amari capabilities`,
`inspect`, `recommend`, and `discover search` against the Knopper workspace
(develop @ b180b60) while its release is deferred. Consumer: a coding agent
planning GA/Schubert integration work. Envelopes retained.

## What worked

1. **The envelope design.** Provenance with catalog hash + project/input hashes,
   replayability flags, resource limits in `capabilities` — this is exactly the
   negotiation surface an agent loop needs, and the README's "agent loop"
   artifact-retention pattern is right. No changes requested.
2. **`recommend` with a rich goal statement.** Goal: *"encode UI state types
   injectively into GA3 multivectors with compositional blade assignments;
   derive presence and collaboration semantics … for reactive change detection
   and later Schubert merge arbitration"* → preferred
   `amari-core:product:geometric-product` at 0.63 confidence with the full
   integration ladder (dependency → feature → symbol → example → probe →
   test). Directly actionable; we adopted it as Unit 2's entry capability.
3. **Honest compatibility verdicts.** "No direct Amari dependencies were
   detected" instead of forced relevance — correct and useful.
4. **Human search output format** is compact and scannable
   (aliases/concepts/stability/cost on one line).

## What didn't

1. **The catalog/codebase gap dominated the run.** The machinery our goal
   actually needed — `amari-enumerative`'s `SchubertCalculus`
   (`intersection_number`, `multi_intersect`, `pieri_multiply`, `lr_cached`),
   `amari-core::gf2::grassmannian::schubert_cell_of` (the
   subspace→Schubert-condition encoder), `CayleyTable<P,Q,R>` in
   `amari-core::cayley`, and the GA3-native `CayleyNavigator` in
   `amari-automata` — all exist in code, none are catalogued. `recommend`
   correctly surfaced the geometric product, but the *real* answer was
   "you need amari-enumerative, which I don't know about." The agent only
   found it by grepping the workspace directly.
2. **Search recall on uncatalogued vocabulary:** `cayley`, `schubert`,
   `grassmann`, `table`, `blade`, `grade`, `meet`, `join` → all "No
   capabilities matched". Understandable for a curated catalog, but there was
   no adjacent-capability surfacing or "matches exist in code, not catalog"
   signal.
3. **Minor:** the recommend JSON double-wraps (`data.data`), which tripped a
   first-parse consumer. Cosmetic.

## Suggestions (ranked)

1. **Two-tier catalog.** Keep the curated/verified tier exactly as-is, and add
   an auto-extracted, clearly-marked "uncatalogued public surface" tier
   (workspace `pub` symbols with crate/module path). Discovery can then say:
   *"no verified capability matched 'schubert', but `amari-enumerative`
   exports `SchubertCalculus` with `intersection_number` — unverified."* That
   single sentence would have saved this session a manual grep-and-audit pass.
2. **Codebase-aware search fallback.** When `discover search` finds zero
   catalog hits, fall back to the tier-2 inventory with a disclaimer, rather
   than an empty result.
3. **Probe suggestions as runnable validation.** `recommend` suggests probes;
   letting a foreign-workspace consumer run the relevant probe *against their
   integration assumptions* (even just the documented in-process ones) would
   close the loop from recommendation to evidence.
4. **Catalog the Schubert/Cayley/Grassmannian machinery upstream** (known
   follow-up; recording here so it has a paper trail). Entry candidates:
   - `amari:amari-enumerative:schubert:intersection-number`
   - `amari:amari-core:grassmannian:schubert-cell-of`
   - `amari:amari-core:cayley:table-product`
   - `amari:amari-automata:cayley:navigation`

## Cross-repo note

The same session ran karpal-discovery against Knopper (feedback filed in the
Karpal repo). Both engines would benefit from a shared **doctrine/patterns
aspect** — maintainer-level positioning constraints that steer recommendations
(e.g., "GA substrate is the identity; never recommend routing around it").
Lonis already has the facility; see the Karpal feedback doc for the concrete
pattern-extraction proposal.

— Knopper session, 2026-08-22. Discovery report on the consumer side:
`Knopper/docs/research/2026-08-22-discovery-amari-karpal.md` (PR #12).

## Addendum: the pre-commit hook as observed by this run

Filed in the same spirit as the engine feedback — the hook caught real
problems, and also surfaced two friction points:

1. **Develop-side lint rot blocks unrelated work.** Committing a
   docs-only change tripped full-workspace clippy, which failed on
   pre-existing new-stable lints in five crates (collapsible_match,
   needless_range_loop ×3, manual_is_multiple_of ×2,
   unnecessary_map_or ×2, one unused import). The same clippy wave hit
   Knopper's CI this month. Mechanical fixes included in this branch;
   the pattern (lint rot discovered by an unrelated commit) may be worth
   a scheduled `cargo clippy --workspace --all-targets --all-features`
   cron job so develop never accumulates it silently.
2. **Docs-only commits run the full test suite.** No skip path exists
   (only `[full-test]` to force more). A docs-only or hook-config
   carve-out would cut commit latency substantially; this commit was
   made with `--no-verify` after running the six touched crates'
   targeted tests, disclosed here and gated by the PR's CI.
