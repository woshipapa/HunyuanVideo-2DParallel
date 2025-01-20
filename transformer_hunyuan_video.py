# Copyright 2024 The Hunyuan Team and The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)



# ...............................my modify.............................................................
data_parallel_group = None
seq_parallel_group = None
parallel_manager = None

import torch.distributed as dist


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class HunyuanVideoAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_length = 0
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
            encoder_length = encoder_hidden_states.shape[1]
        # logger.info(f'[enter attn processor]hidden_states = {hidden_states.shape},encoder_hidden_states = {encoder_hidden_states.shape}, attention_mask ={attention_mask.shape},')
        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)



        # all2all
        from utils import all_to_all_comm, remove_padding, add_padding, remove_extra_encoder, add_extra_encoder
        if parallel_manager.is_enable_seq_parallel():
            sp_size = parallel_manager.get_seq_parallel_size()
            assert (
                attn.heads % sp_size == 0
            ), f"Number of heads {attn.heads} must be divisible by sequence parallel size {sp_size}"
            # 每个机器上头的数量 hc
            attn_heads = attn.heads // sp_size
            query, key, value = map(
                lambda x: all_to_all_comm(x, seq_parallel_group, scatter_dim=2, gather_dim=1),
                [query, key, value],
            )            
        else:
            attn_heads = attn.heads

        # logger.info(f'[first all2all]query shape is {query.shape}, key shape is {key.shape}')# dual (1,9640,768)   single (1, 9640+226*4=10544, 768)

       # 这里是如果是sp的话，这个inner_dim 就是 dim / N
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn_heads

        query = query.unflatten(2, (-1, head_dim)).transpose(1, 2)
        key = key.unflatten(2, (-1, head_dim)).transpose(1, 2)
        value = value.unflatten(2, (-1, head_dim)).transpose(1, 2)
        # logger.info(f'[Rank {torch.distributed.get_rank()}]_[hidden_states] query shape is {query.shape}, key shape is {key.shape}')

        # logger.info(f'query shape is {query.shape}, key shape is {key.shape}') # (1,6,9640,128)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)


        # remove padding for rotary and attention_mask
        if parallel_manager.is_enable_seq_parallel():
            if attn.add_q_proj is not None:
                # dual
                # (b, video_length, d)
                query, key ,value = map(
                    lambda x: remove_padding(x, seq_parallel_group, padding_key = "pad", dim = 2),
                    [query, key, value]
                )
            else:
                # single
                # (b, video_length + text_length ,d)
                query, key ,value = map(
                    # remove extra_encoder and video padding
                    lambda x: remove_extra_encoder(x, encoder_length, seq_parallel_group, padding_key = "pad", dim = 2),
                    [query, key, value]
                )
        # logger.info(f'[after remove extra encoder]query shape is {query.shape}, key shape is {key.shape}')


        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            emb_len = image_rotary_emb[0].shape[0]
            # logger.info(f'image_rotary_emb : {image_rotary_emb[0].shape}')
            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(
                            query[:, :, : -encoder_hidden_states.shape[1]],
                            image_rotary_emb,
                        ),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(
                            key[:, :, : -encoder_hidden_states.shape[1]],
                            image_rotary_emb,
                        ),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query[: , :, :emb_len,:] = apply_rotary_emb(query[: , :, :emb_len,:], image_rotary_emb)
                key[: , :, :emb_len,:] = apply_rotary_emb(key[: , :, :emb_len,:], image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            if parallel_manager.is_enable_seq_parallel():
                sp_size = parallel_manager.get_seq_parallel_size()
                # 每个机器上头的数量 hc
                attn_heads = attn.heads // sp_size
                encoder_query, encoder_key, encoder_value = map(
                    lambda x: all_to_all_comm(x, seq_parallel_group, scatter_dim=2, gather_dim=1),
                    [encoder_query, encoder_key, encoder_value],
                )            
            else:
                attn_heads = attn.heads




            encoder_query = encoder_query.unflatten(2, (attn_heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn_heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn_heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            if parallel_manager.is_enable_seq_parallel():
                encoder_query, encoder_key, encoder_value = map(
                    lambda x: remove_padding(x, seq_parallel_group, padding_key="encoder_pad", dim = 2),
                    [encoder_query, encoder_key, encoder_value]
                )
            encoder_length = encoder_query.shape[2]     


            # logger.info(f'[text]_encoder_query shape is {encoder_query.shape}, key shape is {encoder_key.shape}')
            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # logger.info(f'[Rank {torch.distributed.get_rank()}] query shape is {query.shape}, key shape is {key.shape}')
        # 5. Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        # logger.info(f'[Rank {torch.distributed.get_rank()}]_[after attention][_hidden_states]  hidden_states  shape is {hidden_states.shape}')
        

        
        # 6. Output projection
        if encoder_hidden_states is not None:


            # logger.info(f'[Rank {torch.distributed.get_rank()}]_[before add_padding and alltoall][_hidden_states]  hidden_states  shape is {hidden_states.shape}')
            # video 
            if parallel_manager.is_enable_seq_parallel():
                if attn.add_q_proj is not None:
                    hidden_states, encoder_hidden_states = (
                        hidden_states[:, : -encoder_length],
                        hidden_states[:, -encoder_length :]
                    )                    
                    hidden_states = add_padding(hidden_states, seq_parallel_group, padding_key = "pad", dim=1)
                    
                    encoder_hidden_states = add_padding(encoder_hidden_states, seq_parallel_group, padding_key= "encoder_pad",dim=1)
                    hidden_states, encoder_hidden_states = map(
                        lambda x :all_to_all_comm(x, seq_parallel_group, scatter_dim=1, gather_dim=2),
                        [hidden_states, encoder_hidden_states]
                    )
                else:
                    hidden_states = add_extra_encoder(hidden_states, encoder_length,seq_parallel_group, padding_key = "pad", dim=1)  
                    hidden_states = all_to_all_comm(hidden_states, seq_parallel_group, scatter_dim=1, gather_dim=2)   
                    hidden_states, encoder_hidden_states = (
                        hidden_states[:, : -encoder_length],
                        hidden_states[:, -encoder_length :]
                    )
            else:
                hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )
            # logger.info(f'[Rank {torch.distributed.get_rank()}]_[after add_padding and alltoall][_hidden_states]  hidden_states  shape is {hidden_states.shape}')

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class HunyuanVideoPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 3, # 32
        embed_dim: int = 768, # 3072
    ) -> None:
        super().__init__()

        patch_size = (
            (patch_size, patch_size, patch_size)
            if isinstance(patch_size, int)
            else patch_size
        )
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCFHW -> BNC
        return hidden_states


class HunyuanVideoAdaNorm(nn.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()

        out_features = out_features or 2 * in_features
        self.linear = nn.Linear(in_features, out_features)
        self.nonlinearity = nn.SiLU()

    def forward(
        self, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class HunyuanVideoIndividualTokenRefinerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.ff = FeedForward(
            hidden_size,
            mult=mlp_width_ratio,
            activation_fn="linear-silu",
            dropout=mlp_drop_rate,
        )

        self.norm_out = HunyuanVideoAdaNorm(hidden_size, 2 * hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )
        # logger.info(f'[Rank {torch.distributed.get_rank()}]_[TokenRefinerBlock]_ attn_output shape is {attn_output.shape,attn_output.dtype}')
        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class HunyuanVideoIndividualTokenRefiner(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        self.refiner_blocks = nn.ModuleList(
            [
                HunyuanVideoIndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self_attn_mask = None
        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            seq_len = attention_mask.shape[1]
            attention_mask = attention_mask.to(hidden_states.device).bool()
            self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(
                1, 1, seq_len, 1
            )
            self_attn_mask_2 = self_attn_mask_1.transpose(2, 3)
            self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            self_attn_mask[:, :, :, 0] = True

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, self_attn_mask)

        return hidden_states


class HunyuanVideoTokenRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.token_refiner = HunyuanVideoIndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1) # (1,226,4096)---->(1,4096)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(
                dim=1
            ) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections) # (1,3072)
        hidden_states = self.proj_in(hidden_states) # (1,226,3072)
        # logger.info(f'[Rank {torch.distributed.get_rank()}]_[TokenRefiner]temb shape is {temb.shape}, hidden_states is {hidden_states.shape}')
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        return hidden_states


class HunyuanVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_size_t: int,
        rope_dim: List[int],
        theta: float = 256.0,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [
            num_frames // self.patch_size_t,
            height // self.patch_size,
            width // self.patch_size,
        ]

        axes_grids = []
        for i in range(3):
            # Note: The following line diverges from original behaviour. We create the grid on the device, whereas
            # original implementation creates it on CPU and then moves it to device. This results in numerical
            # differences in layerwise debugging outputs, but visually it is the same.
            grid = torch.arange(
                0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32
            )
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")  # [W, H, T]
        grid = torch.stack(grid, dim=0)  # [3, W, H, T]

        freqs = []
        for i in range(3):
            freq = get_1d_rotary_pos_embed(
                self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True
            )
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)
        return freqs_cos, freqs_sin


class HunyuanVideoSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
            pre_only=True,
        )

        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # from vast.train.utils import logger
        # if torch.distributed.get_rank() == 0:
            # logger.info(f'allocated_mm = {torch.cuda.memory_allocated() / 1024**3}G, reserved_mm = {torch.cuda.memory_reserved() / 1024**3}G, max allocated mm = {torch.cuda.max_memory_allocated() / 1024**3}G, max reserved mm = {torch.cuda.max_memory_reserved() / 1024**3}G')
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states
        
        # logger.info(f'before norm  hidden_states shape(cat hidden and text) is {hidden_states.shape}, input encoder_shape is {encoder_hidden_states.shape}')
        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )
        # logger.info(f'after norm  hidden_states shape(split hidden and text) is {norm_hidden_states.shape} , norm_encoder shape is {norm_encoder_hidden_states.shape}')
        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        # logger.info(f'after attention  hidden_states shape(cat result hidden and text) is {attn_output.shape} , context_attn_output shape is {context_attn_output.shape}, mlp_hs = {mlp_hidden_states.shape}')
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)
        
        # 3. Modulation and residual connection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states) # back to 3072 in dim 2
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        # logger.info(f'after single stream  hidden_states shape(cat hidden and text) is {hidden_states.shape}, input encoder_shape is {encoder_hidden_states.shape}')
        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(
            hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate"
        )

        self.norm2_context = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.ff_context = FeedForward(
            hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # from vast.train.utils import logger
        # if torch.distributed.get_rank() == 0:
            # logger.info(f'allocated_mm = {torch.cuda.memory_allocated() / 1024**3}G, reserved_mm = {torch.cuda.memory_reserved() / 1024**3}G, max allocated mm = {torch.cuda.max_memory_allocated() / 1024**3}, max reserved mm = {torch.cuda.max_memory_reserved() / 1024**3}')
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )
        # (1, 3072)
        # logger.info(f'[Rank {dist.get_rank()}] [video] gate_msa = {gate_msa.shape}, shift_mlp = {shift_mlp.shape}, scale_mlp = {scale_mlp.shape}')
        # logger.info(f'[Rank {dist.get_rank()}] [text] c_gate_msa = {c_gate_msa.shape}, shift_mlp = {c_shift_mlp.shape}, scale_mlp = {c_scale_mlp.shape}')
        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = (
            encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)
        )

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        )

        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    A Transformer model for video-like data used in [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo).

    Args:
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `24`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of dual-stream blocks to use.
        num_single_layers (`int`, defaults to `40`):
            The number of layers of single-stream blocks to use.
        num_refiner_layers (`int`, defaults to `2`):
            The number of layers of refiner blocks to use.
        mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the hidden layer size to the input size in the feedforward network.
        patch_size (`int`, defaults to `2`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        qk_norm (`str`, defaults to `rms_norm`):
            The normalization to use for the query and key projections in the attention layers.
        guidance_embeds (`bool`, defaults to `True`):
            Whether to use guidance embeddings in the model.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        pooled_projection_dim (`int`, defaults to `768`):
            The dimension of the pooled projection of the text embeddings.
        rope_theta (`float`, defaults to `256.0`):
            The value of theta to use in the RoPE layer.
        rope_axes_dim (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions of the axes to use in the RoPE layer.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (16, 56, 56),
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # logger.info(f'[Rank {torch.distributed.get_rank()}]_[PatchEmbed]patch_size_t : {patch_size_t}, patch_size : {patch_size}, in_channels {in_channels}, inner_dim = {inner_dim}')
        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed(
            (patch_size_t, patch_size, patch_size), in_channels, inner_dim
        )
        # logger.info(f'[Rank {torch.distributed.get_rank()}]_[ContextEmbed] text_embedd_dim : {text_embed_dim}, num_heads : {num_attention_heads}, head_dim {attention_head_dim},num_refiner_layers = {num_refiner_layers}')
        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim,
            num_attention_heads,
            attention_head_dim,
            num_layers=num_refiner_layers,
        )
        # logger.info(f'[Rank {torch.distributed.get_rank()}]_[CombinedTimeStepEmbed] pooled_projection_dim :{pooled_projection_dim}')
        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
            inner_dim, pooled_projection_dim
        )

        # 2. RoPE
        self.rope = HunyuanVideoRotaryPosEmbed(
            patch_size, patch_size_t, rope_axes_dim, rope_theta
        )

        # 3. Dual stream transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Single stream transformer blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                HunyuanVideoSingleTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(
            inner_dim, inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            inner_dim, patch_size_t * patch_size * patch_size * out_channels
        )

        self.gradient_checkpointing = True

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # from vast.train.utils import logger
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
        
        # logger.info(f'trainsformer')
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        # logger.info(f'transformer ........................ {engine}')
        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)
        # logger.info(f'[Rank {torch.distributed.get_rank()}]_[Before embedding]timestep:{timestep.shape, timestep.dtype}, hidden_states = {hidden_states.shape, hidden_states.dtype},encoder_hidden_states = {encoder_hidden_states.shape, encoder_hidden_states.dtype}')
        # 3. Attention mask preparation
        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep, guidance, pooled_projections)  # (1,3072)
        hidden_states = self.x_embedder(hidden_states)  # (1, 9639, 3072)
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states, timestep, encoder_attention_mask
        ) # (1,226,3072)

        latent_sequence_length = hidden_states.shape[1] #9639
        condition_sequence_length = encoder_hidden_states.shape[1] #226


        # 3. split video and text
            
        # from vast.train.trainers.trainer import engine
        # print(f'............................................................{engine.mesh_device}')
        from utils import gather_sequence, ParallelManager
        global data_parallel_group,seq_parallel_group, parallel_manager
        parallel_manager = ParallelManager()
        data_parallel_group = parallel_manager.get_data_parallel_group()
        # if hasattr(engine, "seq_parallel_group"):
        #     seq_parallel_group = engine.seq_parallel_group
        #     sq_size = dist.get_world_size(seq_parallel_group)
        padding_sequence, video_padding, text_padding = 0, 0, 0
        # logger.info(f'[Rank {dist.get_rank()}] dp_size = {dist.get_world_size(data_parallel_group)}')
        from utils import set_pad,get_pad,all_to_all_with_pad,split_sequence
        
        if parallel_manager.is_enable_seq_parallel():
            seq_parallel_group = parallel_manager.get_seq_parallel_group()
            set_pad("pad", hidden_states.shape[1], seq_parallel_group)
            set_pad("encoder_pad",encoder_hidden_states.shape[1],seq_parallel_group)
            video_padding = get_pad("pad")
            padding_sequence += video_padding
            hidden_states = split_sequence(hidden_states, seq_parallel_group, dim=1, pad=get_pad("pad")) # (1, 2410, 3072)
            text_padding = get_pad("encoder_pad")
            padding_sequence += text_padding
            encoder_hidden_states = split_sequence(encoder_hidden_states, seq_parallel_group, dim=1, pad=get_pad("encoder_pad")) # (1, 57, 3072)
            # logger.info(f'[Rank {dist.get_rank()}]_[after split] hidden_states = {hidden_states.shape}, encoder_hidden_states = {encoder_hidden_states.shape}')

        # 3. Attention mask preparation

        # add padding_length
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = torch.zeros(
            batch_size,
            sequence_length,
            sequence_length,
            device=hidden_states.device,
            dtype=torch.bool,
        )  # [B, N, N]

        effective_condition_sequence_length = encoder_attention_mask.sum(
            dim=1, dtype=torch.int
        )  # [B,]
        # logger.info(f'[Rank {torch.distributed.get_rank()}]_ encoder_attention_mask: {encoder_attention_mask},effective_length : {effective_condition_sequence_length}')
        
        effective_sequence_length = (
            latent_sequence_length + effective_condition_sequence_length
        )
        # logger.info(f'[Rank {torch.distributed.get_rank()}]_effective_length : {effective_sequence_length[0]}') 
        for i in range(batch_size):
            attention_mask[
                i, :effective_sequence_length[i], : effective_sequence_length[i]
            ] = True

        # logger.info(f'attention_mask shape is {attention_mask.shape}, attention_mask is {attention_mask[0,latent_sequence_length, :]}, video_length = {latent_sequence_length}, video_padding = {video_padding}')        
        # from vast.train.utils import ProfilerWrapper, GPU_Memory_Tracker
        # profiler = ProfilerWrapper(is_st=False,enable_record_cuda_mm=True,enable_print_summary=True)
        # tracker = GPU_Memory_Tracker(prefix='transformer',interval=0.1,use_distributed=True)
        # tracker.start_monitoring()
        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            for i, block in enumerate(self.transformer_blocks):
                    # with profiler:
                    if dist.get_rank() == 0:
                        logger.info(f'..............enter the {i+1} dual-stream block.............')    
                    hidden_states, encoder_hidden_states = (
                            torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                hidden_states,
                                encoder_hidden_states,
                                temb,
                                attention_mask,
                                image_rotary_emb,
                                **ckpt_kwargs,
                            )
                        )


            # dual to single 
            # text need to all-gather
            if parallel_manager.is_enable_seq_parallel():
                encoder_hidden_states = gather_sequence(encoder_hidden_states, seq_parallel_group, dim = 1, pad = get_pad("encoder_pad"))

            for i, block in enumerate(self.single_transformer_blocks):
                # with profiler:    
                if dist.get_rank() == 0:
                    logger.info(f'..............enter the {i+1} single-stream block.............')    
                hidden_states, encoder_hidden_states = (
                        torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            attention_mask,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )
                    )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                )


        if parallel_manager.is_enable_seq_parallel():
            hidden_states = gather_sequence(hidden_states, seq_parallel_group, dim = 1, pad=get_pad("pad"))


        # tracker.start_monitoring()
        # profiler.leave()
        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            -1,
            p_t,
            p,
            p,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
