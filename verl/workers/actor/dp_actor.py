# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            
            
            #DEBUG
            # print(f"[DEBUG] input_ids.shape: {input_ids.shape}")
            # print(f"[DEBUG] attention_mask.shape: {attention_mask.shape}")
            # print(f"[DEBUG] position_ids.shape: {position_ids.shape}")
            # print(f"[DEBUG] batch_size, seqlen: {batch_size} , {seqlen}")
            
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                # print(f"[DEBUG] input_ids_rmpad.shape: {input_ids_rmpad.shape}")
                # print(f"[DEBUG] position_ids_rmpad.shape: {position_ids_rmpad.shape}")
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                # print(f"[DEBUG] input_ids.shape: {input_ids.shape}")
                # print(f"[DEBUG] attention_mask.shape: {attention_mask.shape}")
                # print(f"[DEBUG] position_ids.shape: {position_ids.shape}")
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        
        # Calculate gradient norm
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        
        
        grad_norm_threshold = self.config.grad_norm_threshold  
        
        # Only update if grad_norm is below threshold
        if not torch.isfinite(grad_norm) or (grad_norm_threshold is not None and grad_norm >= grad_norm_threshold):
            # Skip the update
            print(f"[DEBUG] Skipping optimizer step due to high gradient norm: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            print(f"[DEBUG] Performing optimizer step with gradient norm: {grad_norm}")
            self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
            entropy_lst.append(entropy)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            entropys = entropys[revert_indices]

        return log_probs, entropys

    def compute_attention_mass(self, data: DataProto) -> dict:
        """
        Run a probe forward pass to compute cross-segment attention mass.
        Uses monkey-patched attention layers to capture post-RoPE Q, K and
        computes segment attention mass in chunks — never materializes full T×T.

        Segments:
            P — Prompt/Observation language tokens (non-model-generated)
            R — Reasoning tokens (<think>...</think>)
            A — Action tokens (model-generated outside <think>)
            V — Vision/image tokens (VLM only; auto-detected via config)

        For LLM models: returns 3×3 mass matrix over {P, R, A}
        For VLM models: returns 4×4 mass matrix over {P, R, A, V}

        Processes one sample at a time for memory safety.

        Args:
            data: DataProto with input_ids, attention_mask, position_ids,
                  responses, loss_mask, think_mask.
                  meta_info keys:
                    attn_layer_stride (int, default 4)
                    attn_chunk_size (int, default 128)

        Returns:
            dict with:
              'attention_mass': (batch_size, num_probed_layers, N, N) cpu tensor
              'probed_layers': list[int]
              'segment_names': list[str] — ['P','R','A'] or ['P','R','A','V']
        """
        from verl.utils.attention_probe import AttentionSegmentProbe, compute_segment_mass_chunked

        self.actor_module.eval()

        batch_size = data.batch.batch_size[0]
        if batch_size == 0:
            return {}

        seq_len = data.batch['input_ids'].shape[1]
        response_len = data.batch['responses'].shape[1]

        layer_stride = data.meta_info.get('attn_layer_stride', 4)
        chunk_size = data.meta_info.get('attn_chunk_size', 128)

        # ---- Build full-sequence segment masks (bsz, seq_len) ----
        device = data.batch['input_ids'].device

        if 'loss_mask' in data.batch.keys():
            loss_mask_full = data.batch['loss_mask'].bool()
        else:
            loss_mask_full = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            loss_mask_full[:, -response_len:] = data.batch['attention_mask'][:, -response_len:].bool()

        think_mask_full = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        if 'think_mask' in data.batch.keys():
            think_mask_full[:, -response_len:] = data.batch['think_mask'].bool()

        attention_mask = data.batch['attention_mask'].bool()

        # ---- Detect vision tokens for VLM models ----
        # Following Frankenstein paper: vision tokens are identified by
        # image_token_id / video_token_id in the model config (Qwen2.5-VL).
        model = self.actor_module
        while hasattr(model, 'module'):
            model = model.module
        image_token_id = getattr(getattr(model, 'config', None), 'image_token_id', None)
        video_token_id = getattr(getattr(model, 'config', None), 'video_token_id', None)

        v_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        if image_token_id is not None:
            v_mask |= (data.batch['input_ids'] == image_token_id)
        if video_token_id is not None:
            v_mask |= (data.batch['input_ids'] == video_token_id)
        has_vision = v_mask.any().item()

        # P = language prompt/observation. For VLM, exclude vision tokens from P.
        if has_vision:
            p_mask = attention_mask & ~loss_mask_full & ~v_mask
        else:
            p_mask = attention_mask & ~loss_mask_full
        r_mask = think_mask_full & loss_mask_full           # reasoning (<think>/<reflection> tokens)
        a_mask = loss_mask_full & ~think_mask_full          # action tokens

        # ---- Attention sink mitigation ----
        # The first content token (typically <|im_start|>) acts as an
        # attention sink (arXiv 2309.17453).  Exclude it from P to avoid
        # distorting cross-segment mass.  Dynamic detection handles variable
        # left-padding in the rollout-replay path.
        first_attended = attention_mask.long().argmax(dim=1)  # (bsz,)
        if data.meta_info.get('attn_exclude_first_token', True):
            for b in range(batch_size):
                p_mask[b, first_attended[b].item()] = False

        # ---- Prompt sub-segmentation (history vs current observation) ----
        # When split_prompt_segments=True and obs_boundary_pos is available,
        # split P into P_hist and P_curr using a chat-template boundary:
        #   P_hist = system prompt + prior user/assistant turn pairs
        #   P_curr = last user turn (current observation) + generation prompt
        # The boundary is the <|im_start|> of the last user turn.
        # This enables measuring R→P_hist (temporal consistency: how much
        # reasoning references past conversation context) vs R→P_curr
        # (observation grounding: how much reasoning attends to the
        # current turn's input).
        split_prompt = data.meta_info.get('split_prompt_segments', False)
        has_obs_boundary = 'obs_boundary_pos' in data.non_tensor_batch
        if split_prompt and has_obs_boundary:
            obs_boundary = data.non_tensor_batch['obs_boundary_pos']
            p_hist_mask = torch.zeros_like(p_mask)
            p_curr_mask = torch.zeros_like(p_mask)
            for b in range(batch_size):
                # Use position of first attended token as the actual left-padding
                # offset.  The previous code used
                #     pad_offset = seq_len - attention_mask[b].sum()
                # which OVER-estimates left-padding by (response_length -
                # valid_resp_len) because it counts both left-side padding AND
                # response-after-EOS padding.  obs_boundary_pos is relative to
                # prompt start, so only left-padding should be added.
                content_start = int(first_attended[b].item())
                boundary = content_start + int(obs_boundary[b])
                p_hist_mask[b, :boundary] = p_mask[b, :boundary]
                p_curr_mask[b, boundary:] = p_mask[b, boundary:]
        else:
            split_prompt = False  # disable if boundary info unavailable

        # Diagnostic: log segment counts for the first sample
        if batch_size > 0:
            p_cnt = int(p_mask[0].sum().item())
            r_cnt = int(r_mask[0].sum().item())
            a_cnt = int(a_mask[0].sum().item())
            v_cnt = int(v_mask[0].sum().item()) if has_vision else 0
            total_attn = int(attention_mask[0].sum().item())
            diag_msg = (f"[ATTN_PROBE] Segment counts (sample 0): "
                        f"P={p_cnt}, R={r_cnt}, A={a_cnt}")
            if has_vision:
                diag_msg += f", V={v_cnt}"
            if split_prompt:
                p_hist_cnt = int(p_hist_mask[0].sum().item())
                p_curr_cnt = int(p_curr_mask[0].sum().item())
                diag_msg += f"  (P_hist={p_hist_cnt}, P_curr={p_curr_cnt})"
            diag_msg += f"  (total_attended={total_attn}, seq_len={seq_len})"
            print(diag_msg)
            if r_cnt == 0:
                print("[ATTN_PROBE] WARNING: R segment is empty! "
                      "think_mask may not match the reasoning format "
                      "(e.g. <reflection> vs <think>).")

        if split_prompt:
            if has_vision:
                segment_names = ['P_hist', 'P_curr', 'R', 'A', 'V']
            else:
                segment_names = ['P_hist', 'P_curr', 'R', 'A']
        else:
            segment_names = ['P', 'R', 'A', 'V'] if has_vision else ['P', 'R', 'A']
        N = len(segment_names)

        has_mm = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Check if per-turn mass is requested
        compute_per_turn = data.meta_info.get('attn_per_turn', False)
        has_eor = 'end_of_response_position_mask' in data.batch.keys()
        compute_per_turn = compute_per_turn and has_eor

        probe = AttentionSegmentProbe(
            self.actor_module, layer_stride=layer_stride, chunk_size=chunk_size
        )

        all_mass = []
        all_per_turn_mass = []  # list of per-sample turn mass lists

        for i in range(batch_size):
            sample_input_ids = data.batch['input_ids'][i:i + 1]
            sample_attn_mask = data.batch['attention_mask'][i:i + 1]
            sample_pos_ids = data.batch['position_ids'][i:i + 1]
            if sample_pos_ids.dim() == 3:   # qwen2vl mrope: (1,3,seq) -> (3,1,seq)
                sample_pos_ids = sample_pos_ids.transpose(0, 1)

            if split_prompt:
                if has_vision:
                    sample_seg_masks = [
                        p_hist_mask[i:i + 1], p_curr_mask[i:i + 1],
                        r_mask[i:i + 1], a_mask[i:i + 1], v_mask[i:i + 1]
                    ]
                else:
                    sample_seg_masks = [
                        p_hist_mask[i:i + 1], p_curr_mask[i:i + 1],
                        r_mask[i:i + 1], a_mask[i:i + 1]
                    ]
            elif has_vision:
                sample_seg_masks = [
                    p_mask[i:i + 1], r_mask[i:i + 1], a_mask[i:i + 1], v_mask[i:i + 1]
                ]
            else:
                sample_seg_masks = [
                    p_mask[i:i + 1], r_mask[i:i + 1], a_mask[i:i + 1]
                ]

            mm_kwargs = {}
            if has_mm:
                mm = data.non_tensor_batch['multi_modal_inputs'][i]
                for key, val in mm.items():
                    mm_kwargs[key] = val

            try:
                with probe.capture(sample_seg_masks, store_qk=compute_per_turn):
                    with torch.no_grad():
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            self.actor_module(
                                input_ids=sample_input_ids,
                                attention_mask=sample_attn_mask,
                                position_ids=sample_pos_ids,
                                **mm_kwargs,
                                use_cache=False,
                            )

                layer_indices = probe.probed_layer_indices
                sample_mass = torch.stack(
                    [probe.results[li]['mass'] for li in layer_indices], dim=0
                )  # (num_probed_layers, N, N)
                all_mass.append(sample_mass)

                # ---- Per-turn attention mass ----
                if compute_per_turn:
                    eor_full = data.batch['end_of_response_position_mask'][i].bool()
                    # eor positions are in full-sequence space
                    eor_positions = eor_full.nonzero(as_tuple=True)[0].tolist()

                    per_turn_mass_i = []  # list of (num_layers, 2, N) per turn
                    # 2 rows = [R_t query, A_t query], N cols = target segments

                    prev_pos = 0
                    for turn_idx, eor_pos in enumerate(eor_positions):
                        # Turn t response region: [prev_pos, eor_pos] in full seq
                        # R_t = r_mask within this turn's response
                        # A_t = a_mask within this turn's response
                        turn_r_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
                        turn_a_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
                        turn_r_mask[0, prev_pos:eor_pos + 1] = r_mask[i, prev_pos:eor_pos + 1]
                        turn_a_mask[0, prev_pos:eor_pos + 1] = a_mask[i, prev_pos:eor_pos + 1]

                        r_count = int(turn_r_mask.sum().item())
                        a_count = int(turn_a_mask.sum().item())

                        if r_count == 0 and a_count == 0:
                            prev_pos = eor_pos + 1
                            continue

                        # Build turn-local source masks + full-sequence target masks
                        # Source: [R_t, A_t], Target: [P, R, A, (V)]
                        turn_src_masks = [turn_r_mask, turn_a_mask]
                        # Full target masks stay the same as sample_seg_masks
                        # Combine into unified segment list for compute_segment_mass_chunked:
                        # We use source+target and compute the 2×N sub-matrix
                        combined_masks = turn_src_masks + sample_seg_masks
                        # combined_masks: [R_t, A_t, P, R, A, (V)]
                        # We want mass[0, 2:], mass[1, 2:] = R_t→{P,R,A,(V)}, A_t→{P,R,A,(V)}

                        layer_turn_mass = []
                        for li in layer_indices:
                            stored = probe.results[li]
                            full_mass = compute_segment_mass_chunked(
                                stored['q'], stored['k'], combined_masks,
                                chunk_size=chunk_size,
                            )  # (1, H, 2+N, 2+N)
                            # Extract the 2×N sub-matrix (source=[R_t,A_t], target=[P,R,A,(V)])
                            turn_mass_layer = full_mass[0, :, :2, 2:].mean(dim=0)  # (2, N)
                            layer_turn_mass.append(turn_mass_layer.cpu())
                            del full_mass

                        # Stack over layers: (num_layers, 2, N)
                        turn_mass_stacked = torch.stack(layer_turn_mass, dim=0)
                        per_turn_mass_i.append({
                            'turn': turn_idx,
                            'mass': turn_mass_stacked,  # (num_layers, 2, N)
                            'r_count': r_count,
                            'a_count': a_count,
                        })
                        prev_pos = eor_pos + 1

                    # Clean up stored Q, K to free memory
                    for li in layer_indices:
                        probe.results[li].pop('q', None)
                        probe.results[li].pop('k', None)

                    all_per_turn_mass.append(per_turn_mass_i)

            except Exception as e:
                print(f"[ATTN_PROBE] WARNING: Sample {i} failed: {e}")
                # Clean up stored Q, K on failure
                for li in probe.results:
                    probe.results[li].pop('q', None)
                    probe.results[li].pop('k', None)
                continue

        if not all_mass:
            return {}

        attention_mass = torch.stack(all_mass, dim=0)  # (batch, num_layers, N, N)

        result = {
            'attention_mass': attention_mass.cpu(),
            'probed_layers': layer_indices,
            'segment_names': segment_names,
        }

        if compute_per_turn and all_per_turn_mass:
            result['per_turn_mass'] = all_per_turn_mass

        return result

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        if 'loss_mask' in data.batch.keys():
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages','loss_mask']
        else:
            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    #attention_mask = data['attention_mask']
                    
                    # Note:
                    # In agent setting, the prompt is the initial obs, and reponse is the whole trajectory except initial obs
                    # The loss mask has the same shape of attention mask, which has both prompt and response, it masks:
                    # the prompt, the padding of response (right padded), and the obs given by the environment in the reponse
                    
                    if "loss_mask" in data:
                        loss_mask = data['loss_mask']
                    else:
                        print("DEBUG: warning, loss_mask not found in actor update")
                        loss_mask=data["attention_mask"]
                    response_mask = loss_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                  log_prob=log_prob,
                                                                                  advantages=advantages,
                                                                                  eos_mask=response_mask,
                                                                                  cliprange=clip_ratio)
                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
