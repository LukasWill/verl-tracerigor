"""
Cross-segment attention mass probe for mechanistic analysis.

Inspired by:
- "Reasoning Models Don't Always Say What They Think" (arXiv 2602.03978v1)
  Section B.7: Cross-Segment Attention Analysis
- "What does RL improve for Visual Reasoning? A Frankenstein-Style Analysis"
  (arXiv 2602.12395) Section 2.3: Aggregated Attention from Reasoning to
  Vision Tokens — motivates the V (Vision) segment for VLM models.
- Adapted for multi-turn agentic setting (both LLM and VLM).

Segments (for multi-turn agent reasoning):
- P (Prompt/Observation): Language-only prompt + environment text observations
- R (Reasoning): <think>...</think> token spans
- A (Action): Model-generated tokens outside <think> blocks
- V (Vision): Image/video tokens (VLM only; auto-detected via image_token_id)

For LLM (e.g. Qwen 2.5 for SciWorld):  3×3 matrix over {P, R, A}
For VLM (e.g. Qwen 2.5 VL for Sokoban): 4×4 matrix over {P, R, A, V}

The probe monkey-patches attention layers to capture post-RoPE Q, K
(following the approach in Frankenstein's vision_token_swapper.py) and
computes cross-segment attention mass in memory-efficient chunks — never
materializes the full T×T attention matrix.

Per-layer output is an N×N attention mass matrix M where:
    M[i, j] = (1/|S_i|) * Σ_{q∈S_i} Σ_{k∈S_j} softmax(Q·K^T/√d)[q,k]

This gives the average attention mass that tokens in segment i pay to segment j.
Rows approximately sum to 1.

Usage:
    probe = AttentionSegmentProbe(model, layer_stride=4, chunk_size=128)
    # For LLM: 3 masks (P, R, A)
    segment_masks = [p_mask, r_mask, a_mask]
    # For VLM: 4 masks (P, R, A, V)
    segment_masks = [p_mask, r_mask, a_mask, v_mask]
    with probe.capture(segment_masks):
        model(input_ids=..., attention_mask=..., position_ids=..., use_cache=False)
    # probe.results: dict[layer_idx] -> {'mass': (N,N), 'mass_per_head': (H,N,N)}
"""

import math
import functools
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch


SEGMENT_NAMES_LLM = ['P', 'R', 'A']
SEGMENT_NAMES_VLM = ['P', 'R', 'A', 'V']


# ---------------------------------------------------------------------------
# Lazy-loaded transformer helper functions
# ---------------------------------------------------------------------------

_CACHED_FNS: Optional[Dict] = None


def _load_transform_fns() -> Dict:
    """Lazy-load RoPE and repeat_kv functions from transformers.

    Returns dict with possible keys:
        'rope_multimodal' — apply_multimodal_rotary_pos_emb (Qwen2.5-VL / Qwen2-VL)
        'rope_standard'   — apply_rotary_pos_emb (Qwen2 / Qwen2.5 LLM)
        'repeat_kv'       — GQA head expansion
    """
    global _CACHED_FNS
    if _CACHED_FNS is not None:
        return _CACHED_FNS

    fns: Dict = {}

    # Qwen2.5-VL multimodal RoPE
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            apply_multimodal_rotary_pos_emb, repeat_kv,
        )
        fns['rope_multimodal'] = apply_multimodal_rotary_pos_emb
        fns['repeat_kv'] = repeat_kv
    except ImportError:
        pass

    # Qwen2-VL fallback
    if 'rope_multimodal' not in fns:
        try:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                apply_multimodal_rotary_pos_emb, repeat_kv,
            )
            fns['rope_multimodal'] = apply_multimodal_rotary_pos_emb
            if 'repeat_kv' not in fns:
                fns['repeat_kv'] = repeat_kv
        except ImportError:
            pass

    # Standard Qwen2 / Qwen2.5 RoPE (for LLM-only models)
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        fns['rope_standard'] = apply_rotary_pos_emb
        if 'repeat_kv' not in fns:
            from transformers.models.qwen2.modeling_qwen2 import repeat_kv
            fns['repeat_kv'] = repeat_kv
    except ImportError:
        pass

    if not fns:
        print("[ATTN_PROBE] WARNING: Could not import any RoPE or repeat_kv "
              "functions. Probe will use pre-RoPE Q, K (less accurate).")

    _CACHED_FNS = fns
    return fns


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    attn_module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the correct RoPE variant based on model type.

    Dispatches between multimodal RoPE (Qwen2.5-VL, with mrope_section)
    and standard RoPE (Qwen2.5 LLM), mirroring Frankenstein's approach of
    directly reading ``self.rope_scaling["mrope_section"]``.

    Args:
        q: (bsz, num_heads, seq_len, head_dim) post-projection Q
        k: (bsz, num_kv_heads, seq_len, head_dim) post-projection K
        position_embeddings: (cos, sin) from the decoder layer
        attn_module: the attention nn.Module (for reading rope_scaling config)
    """
    if position_embeddings is None:
        return q, k

    cos, sin = position_embeddings
    fns = _load_transform_fns()

    # Detect multimodal RoPE (Qwen2.5-VL / Qwen2-VL)
    rope_scaling = getattr(attn_module, 'rope_scaling', None)
    mrope_section = None
    if isinstance(rope_scaling, dict):
        mrope_section = rope_scaling.get('mrope_section', None)

    if mrope_section is not None and 'rope_multimodal' in fns:
        q, k = fns['rope_multimodal'](q, k, cos, sin, mrope_section)
    elif 'rope_standard' in fns:
        q, k = fns['rope_standard'](q, k, cos, sin)
    # else: pre-RoPE Q, K used (logged warning at import time)

    return q, k


def _expand_kv_heads(k: torch.Tensor, attn_module) -> torch.Tensor:
    """Expand K heads for GQA (grouped-query attention) if needed."""
    num_kv_groups = getattr(attn_module, 'num_key_value_groups', 1)
    if num_kv_groups > 1:
        fns = _load_transform_fns()
        repeat_fn = fns.get('repeat_kv', None)
        if repeat_fn is not None:
            k = repeat_fn(k, num_kv_groups)
    return k


# ---------------------------------------------------------------------------
# Chunked segment mass computation
# ---------------------------------------------------------------------------

def compute_segment_mass_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    segment_masks: List[torch.Tensor],
    chunk_size: int = 128,
) -> torch.Tensor:
    """
    Compute N×N segment attention mass from Q, K without materializing
    the full T×T attention matrix.

    For each source-target segment pair (i, j):
        mass[i,j] = (1/|S_i|) * Σ_{q∈S_i} Σ_{k∈S_j} softmax(Q·K^T/√d)[q,k]

    Args:
        q: (bsz, num_heads, seq_len, head_dim)
        k: (bsz, num_heads, seq_len, head_dim) — must be GQA-expanded
        segment_masks: list of N bool tensors, each (bsz, seq_len)
        chunk_size: number of query positions to process at once

    Returns:
        mass: (bsz, num_heads, N, N) attention mass matrix
    """
    bsz, num_heads, seq_len, head_dim = q.shape
    N = len(segment_masks)
    device = q.device
    scale = 1.0 / math.sqrt(head_dim)

    mass = torch.zeros(bsz, num_heads, N, N, device=device, dtype=torch.float32)
    count = torch.zeros(bsz, N, device=device, dtype=torch.float32)

    # Valid key positions = union of all segments (non-padding)
    valid_key = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
    for m in segment_masks:
        valid_key |= m

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)

        # Query chunk: (bsz, H, chunk_len, d)
        q_chunk = q[:, :, chunk_start:chunk_end, :]

        # Attention scores: (bsz, H, chunk_len, seq_len)
        scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale

        # Causal mask: query at position p can only attend to positions <= p
        query_pos = torch.arange(chunk_start, chunk_end, device=device)
        key_pos = torch.arange(seq_len, device=device)
        causal_mask = key_pos.unsqueeze(0) > query_pos.unsqueeze(1)  # (chunk, seq)
        scores.masked_fill_(causal_mask[None, None, :, :], float('-inf'))

        # Mask invalid (padding) key positions
        scores.masked_fill_(~valid_key[:, None, None, :], float('-inf'))

        # Softmax over key dimension
        attn = torch.softmax(scores, dim=-1)  # (bsz, H, chunk, seq)
        attn = attn.nan_to_num(0.0)  # handle NaN from all-inf rows

        # Accumulate mass for each segment pair
        for src_idx in range(N):
            src_chunk_mask = segment_masks[src_idx][:, chunk_start:chunk_end]  # (bsz, chunk)
            if not src_chunk_mask.any():
                continue

            for tgt_idx in range(N):
                tgt_mask = segment_masks[tgt_idx]  # (bsz, seq)
                # Sum attention to target segment: (bsz, H, chunk)
                tgt_attn = (attn * tgt_mask[:, None, None, :].float()).sum(dim=-1)
                # Sum over source positions in this chunk: (bsz, H)
                mass[:, :, src_idx, tgt_idx] += (
                    tgt_attn * src_chunk_mask[:, None, :].float()
                ).sum(dim=-1)

            count[:, src_idx] += src_chunk_mask.float().sum(dim=-1)

        del scores, attn  # free memory

    # Normalize by number of source positions per segment
    for src_idx in range(N):
        nonzero = count[:, src_idx] > 0
        if nonzero.any():
            mass[nonzero, :, src_idx, :] /= count[nonzero, src_idx, None, None]

    return mass  # (bsz, num_heads, N, N)


# ---------------------------------------------------------------------------
# Main probe class
# ---------------------------------------------------------------------------

class AttentionSegmentProbe:
    """
    Context-manager probe that monkey-patches transformer attention layers
    to capture post-RoPE Q, K during a forward pass and compute cross-segment
    attention mass in memory-efficient chunks.

    Works with both VLM (Qwen2.5-VL) and LLM (Qwen2.5) models.

    Following the Frankenstein paper's approach (vision_token_swapper.py), the
    probe intercepts Q, K projections inside the attention forward, applies
    RoPE using ``position_embeddings`` (cos, sin) already available in the
    call, and computes segment mass — then lets the original flash-attention
    forward proceed.  This avoids:
      1. Having to reimplement the full attention forward (brittle).
      2. Disabling flash attention for ``output_attentions=True`` (slow, ARPO approach).
    """

    def __init__(self, model, layer_stride: int = 4, chunk_size: int = 128):
        """
        Args:
            model: nn.Module — the transformer model (may be FSDP-wrapped)
            layer_stride: probe every Nth layer (default 4)
            chunk_size: query positions per chunk for memory-efficient computation
        """
        self.model = model
        self.layer_stride = layer_stride
        self.chunk_size = chunk_size

        self._results: Dict[int, Dict[str, torch.Tensor]] = {}
        self._original_forwards: Dict[int, tuple] = {}
        self._segment_masks: Optional[List[torch.Tensor]] = None
        self._store_qk: bool = False

    def _unwrap_model(self):
        """Get the underlying model, unwrapping FSDP/DDP if needed."""
        model = self.model
        while hasattr(model, 'module'):
            model = model.module
        return model

    def _get_attention_layers(self) -> List[Tuple[int, object]]:
        """Find attention layers and their indices."""
        model = self._unwrap_model()
        if not (hasattr(model, 'model') and hasattr(model.model, 'layers')):
            raise RuntimeError(
                "Cannot find model.model.layers — model structure not recognized. "
                "AttentionSegmentProbe requires a HuggingFace-style transformer."
            )
        layers = []
        num_layers = len(model.model.layers)
        for idx in range(0, num_layers, self.layer_stride):
            layers.append((idx, model.model.layers[idx].self_attn))
        return layers

    @contextmanager
    def capture(self, segment_masks: List[torch.Tensor], store_qk: bool = False):
        """
        Context manager that patches attention layers and captures segment
        attention mass during the enclosed forward pass.

        Args:
            segment_masks: list of N bool tensors, each (bsz, seq_len).
                           N=3 for LLM (P, R, A) or N=4 for VLM (P, R, A, V).
            store_qk: if True, also store post-RoPE Q, K per probed layer
                       for downstream per-turn mass computation.
        """
        self._segment_masks = segment_masks
        self._store_qk = store_qk
        self._results = {}
        self._original_forwards = {}

        attention_layers = self._get_attention_layers()
        try:
            for layer_idx, attn_module in attention_layers:
                original = attn_module.forward
                self._original_forwards[layer_idx] = (attn_module, original)
                attn_module.forward = self._make_capturing_forward(
                    attn_module, original, layer_idx
                )
            yield self
        finally:
            for layer_idx, (attn_module, original) in self._original_forwards.items():
                attn_module.forward = original
            self._original_forwards.clear()
            self._segment_masks = None

    def _make_capturing_forward(self, attn_module, original_forward, layer_idx):
        """Create a monkey-patched forward that captures post-RoPE Q, K.

        Following Frankenstein's ``forward_attention_collect_kv`` pattern:
        - Intercept Q, K after projection and RoPE application
        - Compute segment attention mass
        - Then delegate to the original forward for actual flash-attention

        We re-run q_proj/k_proj rather than replacing the entire forward to
        avoid model-specific brittleness.  Within the same module forward
        call, FSDP keeps parameters unsharded, so the projections are safe.
        """
        probe = self

        @functools.wraps(original_forward)
        def capturing_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            # Run original forward for actual computation (flash attention)
            result = original_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # Re-run Q, K projections for the probe
            with torch.no_grad():
                bsz, seq_len, _ = hidden_states.shape

                q = attn_module.q_proj(hidden_states)
                k = attn_module.k_proj(hidden_states)

                # Attribute names differ: Qwen2_5_VLAttention has self.num_heads,
                # Qwen2Attention (4.49.0) only has self.config.num_attention_heads
                _num_heads = getattr(attn_module, 'num_heads', None) or attn_module.config.num_attention_heads
                _num_kv_heads = getattr(attn_module, 'num_key_value_heads', None) or attn_module.config.num_key_value_heads
                _head_dim = attn_module.head_dim

                q = q.view(bsz, seq_len, _num_heads,
                           _head_dim).transpose(1, 2)
                k = k.view(bsz, seq_len, _num_kv_heads,
                           _head_dim).transpose(1, 2)

                # Apply rotary position embedding (dispatches VL vs LLM)
                q, k = _apply_rope(q, k, position_embeddings, attn_module)

                # GQA: expand K heads to match Q heads
                k = _expand_kv_heads(k, attn_module)

                # Q, K: (bsz, num_heads, seq_len, head_dim)
                mass = compute_segment_mass_chunked(
                    q, k, probe._segment_masks, chunk_size=probe.chunk_size
                )
                # mass: (bsz, num_heads, N, N)
                N = len(probe._segment_masks)

                result_entry = {
                    'mass': mass.mean(dim=(0, 1)).cpu(),           # (N, N) averaged
                    'mass_per_head': mass.mean(dim=0).cpu(),       # (num_heads, N, N)
                }

                # Optionally store Q, K for per-turn mass computation
                if probe._store_qk:
                    result_entry['q'] = q.detach()  # (bsz, H, T, d) keep on device
                    result_entry['k'] = k.detach()  # (bsz, H, T, d) keep on device

                probe._results[layer_idx] = result_entry

                del q, k, mass

            return result

        return capturing_forward

    @property
    def results(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Get computed attention mass per probed layer."""
        return self._results

    @property
    def probed_layer_indices(self) -> List[int]:
        """Get sorted list of probed layer indices."""
        return sorted(self._results.keys())
