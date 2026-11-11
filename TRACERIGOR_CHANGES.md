# Changes for TraceRigor

This repository is the VERL build used to train the agents in
[RLVR Makes Agents Win More, but Not Necessarily Reason Better](https://openreview.net/forum?id=jsTJu11CIh)
(NeurIPS 2026 workshops) with [TraceRigor](https://github.com/LukasWill/tracerigor).

It is based on [JamesKrW/verl](https://github.com/JamesKrW/verl) at commit
`f0c9812` ("Merge pull request #4 from JamesKrW/vllm8", VERL `0.2.0.dev`),
which in turn derives from [volcengine/verl](https://github.com/volcengine/verl).
The files listed below were modified or added by Wenhao Lu (2025–2026). All
other files are unchanged from the base commit. VERL is distributed under the
Apache License 2.0 (`LICENSE`); upstream notices are kept in `Notice.txt` and
in the individual files.

## Changes that affect training

| File | Change |
| --- | --- |
| `verl/trainer/ppo/core_algos.py` | `compute_policy_loss` uses asymmetric clipping (ε_low = 0.2, ε_high = 0.28) plus dual-clip (c = 3.0) for negative advantages. The values are fixed in code; the `cliprange` argument is not used. All runs in the paper used this loss. |
| `verl/workers/critic/dp_critic.py` | Skips the optimizer step when the gradient norm is non-finite; dynamic micro-batching via `prepare_dynamic_batch`. |
| `verl/models/transformers/modeling_qwen_2_5_vl_patch.py` | Zero-weighted dummy vision pass on text-only batches so vision-tower gradients stay synchronized under FSDP. |
| `verl/models/transformers/monkey_patch.py` | Ulysses flash-attention patch for Qwen2-VL / Qwen2.5-VL (multimodal RoPE). |

## Analysis and evaluation additions

| File | Change |
| --- | --- |
| `verl/workers/actor/dp_actor.py` | `compute_log_prob` also returns per-token entropies; new `compute_attention_mass` probe that measures attention between prompt, reasoning, action, and vision token segments. |
| `verl/workers/fsdp_workers.py` | Exposes `compute_attention_mass` and the entropy outputs to the trainer. |
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
`verl/workers/reward_manager/agent_utils.py`, and the one-line comment change in
`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`.

## Research variants not used by TraceRigor

Kept for reference; nothing in the training path imports them:
`*_archer.py`, `*_rema.py`, `*_backup04Mar.py`, and `*_verl_source.py` copies of
`core_algos`, `dp_actor`, `dp_critic`, `rl_dataset`, `torch_functional`, the
reward manager, and the debug utilities, plus `llama_arpo.py`,
`qwen2_5_vl_arpo.py`, `llama_verl_source.py`, and `dense_common.py`.

## Install

```bash
git clone https://github.com/LukasWill/verl-tracerigor.git
cd verl-tracerigor
pip install --no-deps -e .
```

The tag `tracerigor-neurips2026` marks the revision used for the paper.
