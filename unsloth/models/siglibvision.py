# unsloth_zoo/vision/siglip.py
# Integrate SiglipVisionModel into Unsloth for optimized vision performance

import torch
from typing import Optional, Tuple

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..kernels import (fast_layernorm,
                       patch_layernorm,
                       fast_linear_forward,
                       fast_lora_forward,
                       fast_lora
                       )

from transformers.models.siglip2.modeling_siglip2 import (
    Siglip2VisionModel,
    Siglip2VisionConfig,
    Siglip2Attention,
    Siglip2Encoder,
    Siglip2VisionEmbeddings,
    Siglip2MultiheadAttentionPoolingHead,
    Siglip2EncoderLayer, Siglip2VisionTransformer

)

# Attempt to import FlashAttention variants
# for SigLib we don't have FlashAttention implementation,
# but we follow the same import of another model for convenience
try:
    from transformers.models.siglip2.modeling_siglip2 import Siglip2FlashAttention2, Siglip2SdpaAttention
except ImportError:
    SiglipFlashAttention2 = Siglip2Attention
    SiglipSdpaAttention = Siglip2Attention


def fast_siglip2_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Inline Triton-optimized forward for Siglip2MLP.
    """
    # First projection
    proj1 = fast_linear_forward(self.fc1, hidden_states)
    # PyTorchGELUTanh activation I found
    # that this activation function is used with SigLib and Also used with Gemma So I didn't use the custom kernel Glu
    activated = ACT2FN[self.config.hidden_act](proj1)
    # Second projection
    output = fast_linear_forward(self.fc2, activated)
    return output


def Siglip2Attention_fast_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Triton-optimized forward for Siglip2Attention, analogous to LlamaAttention_fast_forward.
    """
    # Input shape: (batch_size, seq_len, embed_dim)
    bsz, seq_len, _ = hidden_states.size()

    # Q, K, V projections
    Q = fast_linear_forward(self.q_proj, hidden_states)
    K = fast_linear_forward(self.k_proj, hidden_states)
    V = fast_linear_forward(self.v_proj, hidden_states)

    # Reshape to (batch_size, num_heads, seq_len, head_dim)
    Q = Q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    K = K.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    V = V.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    # Select attention implementation
    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # Compute attention
    attn_output, attn_weights = attention_interface(self, Q, K, V,
                                                    attention_mask,
                                                    is_causal=self.is_causal,
                                                    scaling=self.scale,
                                                    dropout=0.0 if not self.training else self.dropout,
                                                    )

    # Reshape output and apply out projection
    attn_output = attn_output.reshape(bsz, seq_len, self.embed_dim).contiguous()
    attn_output = fast_linear_forward(self.out_proj, attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights


def Siglip2VisionEncoderLayer_fast_forward(self, hidden_states: torch.Tensor,
                                           attention_mask: torch.Tensor,
                                           output_attentions: Optional[bool] = False,
                                           ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    residual = hidden_states
    hidden_states = fast_layernorm(self.layer_norm1, hidden_states)

    hidden_states, attn_weights = Siglip2Attention_fast_forward(self.self_attn,
                                                                hidden_states=hidden_states,
                                                                attention_mask=attention_mask,
                                                                output_attentions=output_attentions,
                                                                )
    hidden_states = residual + hidden_states

    # Second residual block + MLP
    residual = hidden_states
    hidden_states = fast_layernorm(self.layer_norm2, hidden_states)
    hidden_states = fast_siglip2_mlp(self.mlp, hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)
    return outputs


def SigLib2VisionEncoder_fast_forward(self,
                         inputs_embeds,
                         attention_mask: Optional[torch.Tensor] = None,
                         output_attentions: Optional[bool] = None,
                         output_hidden_states: Optional[bool] = None,
                         ):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    hidden_states = inputs_embeds
    for encoder_layer in self.layers:
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                Siglip2VisionEncoderLayer_fast_forward(encoder_layer.__call__,
                                                       hidden_states,
                                                       attention_mask,
                                                       output_attentions,
                                                       )
            )
        else:
            layer_outputs = Siglip2VisionEncoderLayer_fast_forward(encoder_layer.__call__,
                                                                   hidden_states,
                                                                   attention_mask,
                                                                   output_attentions=output_attentions,
                                                                   )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=encoder_states,
        attentions=all_attentions,
    )


