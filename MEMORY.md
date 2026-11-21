# Memory requirements

| Operation/profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
|---|---:|---:|---:|---|
| Expanded behavior response training | 1 | 18 | 20 | batch 8; prompts≤ 512 tokens including target instruction; BF16/SDPA; all 36 blocks checkpointed; 138 cachedpairs |
| Full-answer native generation | 1 | 12 | 20 | batch 1; ≤ 512 prompt+4096 new; hybrid9full/27sliding4096 KV; BF16/SDPA; native EOS |
| Remote Jev observer / report tools | 0 | 0 | 2 | one remote request at a time; ≤ 4096texttokens; local model unloaded |

These are **calculated planning budgets, not measured peaks or hard runtime ceilings**. GiB means 2³⁰ bytes. The table bounds one operation at a time; do not add sequential phases, and do add concurrently resident models/jobs. Longer contexts, larger batches, reference audio or different backends require recalculation; the package does not have a universal finite maximum for every accepted request. No benchmark, model construction, forward, GPU query or training run was performed for this document.

Frozen BF16/FP16 weights cost `2P`; FP32 costs `4P`. LoRA costs `r × Σ(in + out)`. The root particle bank is counted once (128 × 4 FP32 values). Rank-8 shared RoutedMLP branches have 6,476 parameters each: router 8→16→16→16→4 and network 12→48→48→48→8; graph variants below use this conservative branch allowance. Critic allocations are included. Frozen weights have no parameter gradients or optimizer moments.

For ParticleGAN's `bispf_optimized`, DualNorm has zero momentum: do **not** add Adam's two full-sized moments. We reserve **64 bytes per trainable scalar** for FP32 parameters/gradients, old/displaced parameters and normalization/constraint temporaries, including protected-normal `.double()` buffers; small scalar step/constraint state fits the workspace allowance. This is a conservative transient envelope, not 64 bytes of persistent optimizer state. No model-weight EMA copy is allocated in the supported RAW policy. A separately identified supervised AdamW comparator uses its actual two FP32 moments.

Activation estimates deliberately include FP32 routed branches and two FP32 attention-sized buffers for eager/math attention. With `L` layers, width `H`, FF width `I`, query heads `N`, tokens `T` and effective accumulated batch `B`, the uncheckpointed training allowance is `64BLTH + 12BLTI + 8BLNT²` bytes. With block checkpointing it is `8BLTH + 64BTH + 12BTI + 8BNT²`; hybrid attention uses only its full-attention layer count in the quadratic term, with recurrent state added separately. These are explicit planning allowances for saved tensors/recomputation, not a promise about every allocator/kernel. Sampling prefill uses `16BTH + 8BTI + 8BNT²`, plus actual KV/recurrent state and any materialized logits. Runtime/workspace allowances below cover Python/framework/CUDA workspaces, allocator fragmentation and the small game state; an executor must still enforce live resource checks.

Host budgets count the stated CPU loading peak, tensor caches and serialization copies plus workspace. They exclude an optional **reclaimable OS file cache** of the complete checkpoint: add the checkpoint file size if deliberately retaining that cache in RAM. Model files remaining on disk are not resident GPU weights. Training duration/update count changes checkpoint disk consumption, not this live memory calculation. RAW/FULL serialization can add CPU copies of trainables; those are included in the host allowances.

Model: **CohereLabs/tiny-aya-global**, revision **`af89d219b53ed9b13b8a4645f9c8028973510324`**.

The table rounds each calculated phase sum upward to the next2 GiB. The byte ledger below gives the unrounded calculation; per-GPU split estimates are listed in device order.

| Phase | Allocation ledger (GiB before rounding) |
|---|---|
| Expanded behavior response training | frozen: 6.2384; trainable_transient: 0.2880; checkpoint_activations: 3.5039; cache: 0.0021; workspace: 6.0000 |
| Full-answer native generation | frozen: 6.2384; adapter: 0.0180; hybrid_KV: 0.2900; prefill_logits_upper: 0.3389; workspace: 4.0000 |
| Remote Jev observer / report tools | local_text_JSON_HTTP_workspace: 2.0000 |

Detailed arithmetic and assumptions:

- **BF16 frozen bytes**: `6698455040`.
- **LoRA elements**: `3833856`.
- **projections**: `144`.
- **generator plus critic elements**: `4832545`.
- **hybrid KV formula**: `2*(9*4608+27*4096)*1*4*128*2`.
- **tied embedding**: `token embedding/head counted once in safetensors total`.
- **host loading**: `complete checkpoint +largest4,992,396,352-byte shard buffer +8 GiB +138-pair response cache +exports`.

Scope and limits:

- This repository trains the TinyAya model, so the table includes actual model memory rather than treating it as generic infrastructure. The tied vocabulary head is not counted twice.
- 512 prompt tokens is an explicit planning bound; Source has a4096-new-token generation budget and8192model context. For a4096-token prompt+4096completion, replace prefills/KV with those lengths; the table is not an8192-token guarantee.
- The primary intended-effect observer is remote Jev, with no local8B observer model allocation. Optional local Qwen3-VL observer tools are a separate model/run and are not included in the remote-observer row. No new score or model qualification is implied.

Source facts (read as text; no model execution):

- [configs/cohort.json](configs/cohort.json)
- [src/evidence_dial/binding.py](src/evidence_dial/binding.py)
- [src/evidence_dial/behavior_expanded_binding_v2.py](src/evidence_dial/behavior_expanded_binding_v2.py)
- [src/evidence_dial/behavior_fullanswer_generate_v2.py](src/evidence_dial/behavior_fullanswer_generate_v2.py)
- [src/evidence_dial/evaluate.py](src/evidence_dial/evaluate.py)
- [src/evidence_dial/protocol.py](src/evidence_dial/protocol.py)
- [Pinned model metadata](https://huggingface.co/CohereLabs/tiny-aya-global/tree/af89d219b53ed9b13b8a4645f9c8028973510324) (architecture/parameter metadata or exact file-size lock as described above).
- ParticleGAN 0.11.1, `recipes.py:bispf_optimized`, `optim/dualnorm.py`, `optim/direction_blend.py` and `optim/constraint_geometry.py`; shared core `routed_projection.py`, `reference.py:RoutedMLP`, `audio.py:AudioResponseCritic`. These optimizer formulas are shared dependencies, not a new optimizer implementation.

Calculated 2026-11-14. Counts, live runtime qualification and memory requirements are separate facts.
