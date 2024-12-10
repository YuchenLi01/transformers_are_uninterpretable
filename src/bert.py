import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel
from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
    BertAttention,
    BertLayer,
    BertIntermediate,
    BertEncoder,
    BertModel,
    BertEmbeddings,
    BertPooler,
    BertLMPredictionHead,
    BertOnlyMLMHead,
)


class BertSelfOutputCustom(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.attn_output_fc:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.dense = None
        if config.layer_norm:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.LayerNorm = nn.Identity()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.residual = config.residual
        self.attn_output_fc = config.attn_output_fc

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.attn_output_fc:
            hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.residual:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertSelfAttentionCustom(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        if config.freeze_uniform_attention:
            self.key.weight.data = torch.zeros((config.hidden_size, config.hidden_size))
            self.key.weight.requires_grad = False
            self.key.bias.data = torch.zeros((config.hidden_size,))
            self.key.bias.requires_grad = False

            self.query.weight.data = torch.zeros((config.hidden_size, config.hidden_size))
            self.query.weight.requires_grad = False
            self.query.bias.data = torch.zeros((config.hidden_size,))
            self.query.bias.requires_grad = False


class BertAttentionCustom(BertAttention):
    def __init__(self, config, position_embedding_type=None,):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.self = BertSelfAttentionCustom(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutputCustom(config)
        self.pruned_heads = set()


class BertOutputCustom(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        if config.layer_norm:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.LayerNorm = nn.Identity()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.residual = config.residual

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.residual:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLayerCustom(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttentionCustom(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttentionCustom(config, position_embedding_type="absolute")
        self.bert_intermediate = config.bert_intermediate
        if config.bert_intermediate:
            self.intermediate = BertIntermediate(config)
        else:
            self.intermediate = None
        self.bert_output = config.bert_output
        if config.bert_output:
            self.output = BertOutputCustom(config)
        else:
            self.output = None

    def feed_forward_chunk(self, attention_output):
        if self.bert_intermediate:
            intermediate_output = self.intermediate(attention_output)
        else:
            intermediate_output = attention_output
        if self.bert_output:
            layer_output = self.output(intermediate_output, attention_output)
        else:
            layer_output = intermediate_output
        return layer_output


class BertEncoderCustom(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([BertLayerCustom(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False


class BertModelCustom(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoderCustom(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        # self.post_init()  # commented out because it disrupts our special initialization


class BertLMPredictionHeadCustom(BertLMPredictionHead):
    def __init__(self, config):
        super().__init__(config)
        self.bert_head_transform = config.bert_head_transform
        if not config.bert_head_transform:
            self.transform = None
        if config.freeze_decoder_to_I:
            one_hot_embeddings = torch.zeros((config.vocab_size, config.hidden_size))
            for i in range(config.vocab_size):
                one_hot_embeddings[i][i] = 1.0
            self.decoder.weight.data = one_hot_embeddings
            self.decoder.weight.requires_grad = False
            self.decoder.bias.data = torch.zeros((config.vocab_size,))
            self.decoder.bias.requires_grad = False

    def forward(self, hidden_states):
        if self.bert_head_transform:
            hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHeadCustom(BertOnlyMLMHead):
    def __init__(self, config):
        super().__init__(config)
        self.predictions = BertLMPredictionHeadCustom(config)


class BertForMaskedLMCustom(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelCustom(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHeadCustom(config)

        # Initialize weights and apply final processing
        # self.post_init()  # commented out because it disrupts our special initialization
