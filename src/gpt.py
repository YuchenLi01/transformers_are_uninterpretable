import torch
import torch.nn as nn
from tqdm import tqdm
import math
from typing import Optional, Tuple, Union
from transformers import  GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2Block,
    GPT2Model,
    GPT2LMHeadModel
)

class GPT2AttentionCustom(GPT2Attention):
    def __init__(self, config):
        super().__init__(config)
        if(not config.attn_output_fc):
            self.c_proj = torch.nn.Identity()
        self.uniform = config.freeze_uniform_attention
        # Forcing a small initialization
        if(config.small_initialization):
            self.reinitialize = False
        else:
            self.reinitialize = True
        
    
    def __reinitialize__(self):
        if(not self.reinitialize):
            self.reinitialize = True
            M = 100000
            self.c_attn.weight = torch.nn.Parameter(self.c_attn.weight / M)
            self.c_attn.bias = torch.nn.Parameter(self.c_attn.bias / M)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            
        query = self._split_heads(query, self.num_heads, self.head_dim)
        # Add Here to Perform Uniform Attention
        if(self.uniform):
            query = torch.zeros(query.shape, device = query.device)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, value)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class GPT2BlockCustom(GPT2Block):
    def __init__(self, config, layer_idx = None):
        super().__init__(config, layer_idx)
        self.attn = GPT2AttentionCustom(config)
        self.ln_1 = torch.nn.Identity()
        if(not config.layer_norm):
            self.ln_2 = torch.nn.Identity()
        if(not config.gpt_intermediate):
            self.mlp = torch.nn.Identity()
        if config.add_cross_attention:
            self.crossattention = GPT2AttentionCustom(config, is_cross_attention=True, layer_idx=layer_idx)
            if(not config.layer_norm):
                self.ln_cross_attn = torch.nn.Identity()
        self.residual = config.residual
        self.first_residual = config.first_residual
        self.another_first_residual = config.another_first_residual
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        if(self.first_residual):
            hidden_states = attn_output + residual
        else:
            hidden_states = attn_output

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        if(self.another_first_residual):
            hidden_states = hidden_states + residual
            residual = hidden_states
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        if(self.residual):
            hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPT2ModelCustom(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([GPT2BlockCustom(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        if(not config.layer_norm):
            self.ln_f = torch.nn.Identity()
        if(config.one_layer_norm):
            self.ln_f = torch.nn.Identity()


class GPT2LMHeadModelCustom(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2ModelCustom(config)
        # Must be here, to make sure we can force the lm head to be I in the following
        self.post_init()
        if config.freeze_decoder_to_I:
            one_hot_embeddings = torch.zeros((config.vocab_size, config.hidden_size))
            for i in range(config.vocab_size):
                one_hot_embeddings[i][i] = 1.0
            self.lm_head.weight.data = one_hot_embeddings
            self.lm_head.weight.requires_grad = False
        for block in self.transformer.h:
            block.attn.__reinitialize__()
