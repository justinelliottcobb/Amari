# Indirect Einsums for Sparse GPU Compute — Future Research Thread

- **Status:** Parked. Targets **v0.26.0+** (after the Borsalino GPU modernization lands).
- **Logged:** 2026-08-12
- **Source:** Jaeyeon Won, Willow Ahrens, Joel S. Emer, Suman Amarasinghe.
  *Insum: Sparse GPU Kernels Simplified and Optimized with Indirect Einsums.*
  arXiv:2510.17505 [cs.PL], October 2025.
- **Companion analysis:** agent memory `Industrial Algebra/sparse-compute-indirect-einsum`
  and `Amari/indirect-einsum-future-work`.

> This note records a research/design thread, not a committed plan. It exists so
> the idea is rediscoverable when v0.26.0 GPU work begins — not to expand current
> scope.

## TL;DR

The Insum paper shows that **sparse GPU kernels can be expressed as dense
kernels plus indirection** (gather/scatter), lowered through an ordinary dense
tensor compiler with the indirection fused in. This is directly relevant to
Amari's v0.25.0 rewrite research and v0.26.0 GPU/Borsalino modernization. The
honest assessment: the *expressiveness* transfers cleanly to Amari's path; the
*peak performance* is tied to a Triton-class lowering that Amari's planned
WGSL/naga-via-Borsalino path does not give for free. The genuinely Amari-shaped
contribution is not "reimplement Insum" but **prove (or refute) its rewrite**
— Insum validates the format-agnostic → indirect-Einsum rewrite only by testing,
and Amari's formal-verification posture is built to do better.

## The approach, briefly

Insum takes a **format-agnostic Einsum** (e.g. `C[m,n] = A[m,k] * B[k,n]` with
`A` sparse) and, given `A`'s storage format, rewrites it into a
**format-conscious *indirect* Einsum** in which the sparse format's value array
(`AV`) and coordinate-index arrays (`AM`, `AK`) become ordinary dense tensors and
the sparsity is expressed purely through indirection:

```text
C[AM[p], n] += AV[p] * B[AK[p], n]      # COO SpMM: gather B at AK[p], scatter to C at AM[p]
```

Three sub-ideas carry the result:

1. **Indirection as the sparse↔dense bridge.** RHS indirection is a gather; LHS
   indirection is a scatter with summation-collision semantics. No merge lattices,
   no intersection/union control flow — the sparse kernel becomes a dense kernel
   plus gather/scatter.
2. **Fixed-length formats.** Einsums require fixed loop bounds, which rules out
   CSR's data-dependent per-row loop. Insum introduces **GroupCOO** and
   **BlockGroupCOO** (between COO and ELL), with a group-size heuristic
   (√(nnz / rows), rounded to a power of two) that minimizes indirect accesses.
3. **Lowering to a dense compiler with fusion.** The indirect Einsum is lowered
   to PyTorch → TorchInductor → Triton, with a codegen extension that
   pattern-matches broadcasted-mult-then-sum into `tl.dot` (Tensor Cores) and
   applies "Lazy Broadcasting" so the `tl.dot` operands land in the right shape
   without reshape/transpose. The gather + matmul + scatter fuse into one kernel.

Reported outcomes: 1 line of Einsum vs. 202–4491 lines of hand-written CUDA/Triton;
1.14×–3.81× over cuSPARSE / Sputnik / TorchBSR / TorchSparse / e3nn, across
SpMM (structured + unstructured), point-cloud sparse convolution, and equivariant
tensor products.

## Why it matters to Amari

- **Amari is the rewrite layer.** v0.25.0 is "comprehensive rewrite research,
  including first-class inverse rewriting." An indirect-Einsum pass is *another
  rewrite* in that vocabulary: a computation paired with a sparse format rewrites
  to an indirect computation over dense tensors. It is a natural extension of the
  rewrite machinery already being expanded in v0.25.0.
- **The GPU lowering is being modernized in v0.26.0.** v0.26.0 is the
  GPU / current-`wgpu` / Borsalino modernization. Whatever sparse-GPU story Amari
  adopts should be decided *as part of* that modernization, not bolted on after.
- **Real sparse workloads exist upstream.** Minoru's sparse Hahn-series
  convolution (already flagged as an O(n·m) bottleneck) and Schubert's
  enumerative sums (Littlewood-Richardson, Atiyah-Bott — sums over combinatorial
  objects where most terms are zero) are exactly the sparse-tensor territory
  Insum targets. Both lower to the GPU through Amari → Borsalino.

## Layering (do not put this in Borsalino)

Borsalino is a **dispatcher** (WGSL → SPIR-V via naga; dispatch + numerical
verification; its flagship kernel is the geometric product of multivectors). It
is not a tensor compiler and should not grow one. The indirect-Einsum lowering
belongs in Amari's compute-front-end (the layer that lowers to Borsalino
dispatches). Borsalino's only role is to run the generated gather/scatter/dot
kernels. Keeping this boundary clean is consistent with
[Zunesha ADR 0003 (cross-crate proof agreement)](https://github.com/Industrial-Algebra/Zunesha/blob/main/docs/adr/0003-cross-crate-proof-agreement.md):
Amari owns the rewrite/structure, Borsalino owns numerical exactness, Zunesha
owns device/buffer safety.

## The tension to decide in v0.26.0

Insum's peak performance is **Triton-specific**: the `tl.dot` Tensor-Core fusion,
the 2D tiling, and Lazy Broadcasting all live in TorchInductor's loop-level IR.
Amari's planned path is **WGSL/SPIR-V via naga over Borsalino** (raw FFI, no
Triton/PyTorch), and v0.26.0 is precisely the release that moves Amari off
`wgpu`.

So there is a fork, to be made consciously during v0.26.0:

- **Expressiveness only.** Adopt indirect Einsum as an IR + the fixed-length
  formats + the rewrite, lower to whatever Borsalino dispatches (WGSL/SPIR-V).
  Forfeit the fused-Tensor-Core codegen; the gather/scatter/dot become separate
  kernels or hand-fused SPIR-V. Cheap to adopt; modest perf.
- **Adopt a Triton-class lowering target.** Pursue Insum's fusion wins directly.
  A real departure from the raw-FFI/WGSL lineage and a meaningful scope increase.
- **Hand-roll fused gather+matmul+scatter SPIR-V.** Keep the raw-FFI lineage,
  chase the fusion manually. Substantial engineering.

The decision belongs with the v0.26.0 GPU-strategy work and should not be
pre-empted here.

## The Amari-shaped contribution: certify the rewrite

Insum's correctness rests on the format-agnostic → indirect-Einsum rewrite being
semantics-preserving (the gather/summation ≡ the original reduction; the
scatter's write-collision ≡ accumulation). **They prove none of it** — it is
validated by testing.

Amari's formal-verification posture (see `docs/technical/FORMAL_VERIFICATION.md`,
and the Karpal/Lean verification stack) is built to do better. The distinctive
research contribution is therefore:

> A **certified** sparse-Einsum lowering: a machine-checked proof (Lean/Karpal)
> that the indirect-Einsum rewrite preserves the semantics of the original
> computation for each supported format — or a **refutation** showing where/when
> it does not.

This is the part worth pursuing as research. Insum is the prior art to cite, not
the system to port. The proof/refutation framing is endorsed as the follow-on
research direction.

## What can be done at the ground floor, cheaply (if/when unblocked)

Independent of the v0.26.0 lowering decision, two low-cost design moves keep the
door open without expanding scope:

1. Treat **indirection (gather/scatter) as a first-class operation** in Amari's
   compute IR, so "dense computation + indirection = sparse/irregular
   computation" is expressible without a special sparse subsystem.
2. Reserve a **computation ⊗ format → indirect computation** seam in the rewrite
   vocabulary, and log the rewrite-soundness claim as a Karpal verification
   obligation rather than a test-suite belief.

Neither requires choosing a lowering target today.

## Open questions for v0.26.0

- Which upstream sparse workloads (Minoru Hahn series, Schubert enumerative sums)
  are actually bound for the GPU, and on what formats?
- Does the v0.26.0 GPU strategy keep WGSL/naga-via-Borsalino, or open the door to
  a Triton-class lowering — and if the latter, where does that lowering live?
- What is the minimal format set to support first (COO → GroupCOO →
  BlockGroupCOO), and is the group-size heuristic worth porting?
- Where does the rewrite-soundness proof live — Karpal, Lean, or both — and what
  is the first format for which a full proof is tractable?

## References

- Won, Ahrens, Emer, Amarasinghe. *Insum: Sparse GPU Kernels Simplified and
  Optimized with Indirect Einsums.* arXiv:2510.17505, 2025.
- Amari release sequence: [`V0_20_0_TO_V0_26_0_RELEASE_SEQUENCE.md`](V0_20_0_TO_V0_26_0_RELEASE_SEQUENCE.md)
- Cross-crate proof agreement:
  [Zunesha ADR 0003](https://github.com/Industrial-Algebra/Zunesha/blob/main/docs/adr/0003-cross-crate-proof-agreement.md)
