# Changes for TraceRigor

VERL build used to train the agents in
[RLVR Makes Agents Win More, but Not Necessarily Reason Better](https://openreview.net/forum?id=jsTJu11CIh)
with [TraceRigor](https://github.com/LukasWill/tracerigor), based on
[JamesKrW/verl](https://github.com/JamesKrW/verl) at `f0c9812` (VERL `0.2.0.dev`).

## Changes that affect training

| File | Change |
| --- | --- |
| `verl/trainer/ppo/core_algos.py` | `compute_policy_loss`: asymmetric clipping (ε_low = 0.2, ε_high = 0.28) plus dual-clip (c = 3.0) for negative advantages, fixed in code (`cliprange` is ignored). Used for all paper runs. |
| `verl/workers/critic/dp_critic.py` | Skips the optimizer step on a non-finite gradient norm; dynamic micro-batching. |
| `verl/models/transformers/modeling_qwen_2_5_vl_patch.py` | Zero-weighted dummy vision pass on text-only batches, keeping vision-tower gradients in sync under FSDP. |
| `verl/models/transformers/monkey_patch.py` | Ulysses flash-attention patch for Qwen2-VL / Qwen2.5-VL (multimodal RoPE). |

## Analysis and evaluation additions

| File | Change |
| --- | --- |
| `verl/workers/actor/dp_actor.py` | `compute_log_prob` also returns per-token entropies; `compute_attention_mass` probes attention between prompt, reasoning, action, and vision tokens. |
| `verl/workers/fsdp_workers.py` | Exposes the attention probe and entropies to the trainer. |
| `verl/utils/attention_probe.py` | Chunked segment attention-mass computation used by the probe. |
| `verl/utils/debug/`, `verl/utils/profiler/` | Debugging, trajectory tracking, and profiling helpers. |
| `scripts/eval/`, `scripts/model_merger_verl.py` | Evaluation and checkpoint-merging scripts. |

## Infrastructure backported from newer VERL

`verl/utils/fsdp_utils.py` (FSDP2 helpers), `verl/utils/seqlen_balancing.py`
(Karmarkar–Karp sequence balancing), `verl/utils/ulysses.py`,
`verl/utils/kernel/`, `verl/utils/seed.py`, `verl/base_config.py`,
`verl/trainer/config/`, `verl/trainer/ppo/metric_utils.py`,
`verl/trainer/ppo/reward.py`, `verl/third_party/`,
`verl/utils/dataset/dataset_utils.py`, `verl/utils/dataset/multiturn_sft_dataset.py`,
and `verl/workers/reward_manager/agent_utils.py`.

## Research variants not used by TraceRigor

The `*_archer.py`, `*_rema.py`, `*_backup04Mar.py`, and `*_verl_source.py`
copies of `core_algos`, `dp_actor`, `dp_critic`, `rl_dataset`,
`torch_functional`, the reward manager, and the debug utilities, plus
`llama_arpo.py`, `qwen2_5_vl_arpo.py`, `llama_verl_source.py`, and
`dense_common.py`, are kept for reference. Nothing in the training path imports
them.

## Install

```bash
git clone https://github.com/LukasWill/verl-tracerigor.git
cd verl-tracerigor
pip install --no-deps -e .
```

The tag `tracerigor-neurips2026` marks the revision used for the paper.
