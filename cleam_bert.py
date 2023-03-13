import torch
import torch.nn as nn
from transformers import BertPreTrainedModel
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.models.bert.modeling_bert import BertPooler, BertIntermediate, BertEmbeddings, BertAttention


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, intermediate_output, attention_output, hidden_states=None):
        intermediate_output = self.dense(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)
        if hidden_states is not None:
            intermediate_output = self.LayerNorm(intermediate_output + attention_output + hidden_states)
        else:
            intermediate_output = self.LayerNorm(intermediate_output + attention_output)
        return intermediate_output


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        cross_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        attention_output = cross_attention_outputs[0]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, (attention_output, hidden_states)
        )

        self_attention_outputs = self.attention(
            layer_output,
            attention_mask,
        )
        attention_output = self_attention_outputs[0]
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        return layer_output

    def feed_forward_chunk(self, residual_features):
        hidden_states = None
        if type(residual_features) == tuple:
            attention_output, hidden_states = residual_features
        else:
            attention_output = residual_features
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, hidden_states)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        text_hidden_states=None,
        text_attention_mask=None,
        label_hidden_states=None,
        label_attention_mask=None,
    ):
        for i, layer_modules in enumerate(zip(self.layer, self.layer_)):
            text_layer_outputs = layer_modules[0](
                text_hidden_states,
                text_attention_mask,
                label_hidden_states,
                label_attention_mask,
            )

            label_layer_outputs = layer_modules[1](
                label_hidden_states,
                label_attention_mask,
                text_hidden_states,
                text_attention_mask,
            )

            text_hidden_states = text_layer_outputs
            label_hidden_states = label_layer_outputs

        return text_hidden_states, label_hidden_states


class BertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        text_input_ids=None,
        text_attention_mask=None,
        text_token_type_ids=None,
        label_input_ids=None,
        label_attention_mask=None,
        label_token_type_ids=None,
    ):
        text_input_shape = text_input_ids.size()
        label_input_shape = label_input_ids.size()

        device = text_input_ids.device

        text_token_type_ids = (
            torch.zeros(text_input_shape, dtype=torch.long, device=device)
            if text_token_type_ids is None
            else text_token_type_ids
        )
        label_token_type_ids = (
            torch.zeros(label_input_shape, dtype=torch.long, device=device)
            if label_token_type_ids is None
            else label_token_type_ids
        )

        text_extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            text_attention_mask, text_input_shape
        )
        label_extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            label_attention_mask, label_input_shape
        )

        text_embedding_output = self.embeddings(
            input_ids=text_input_ids,
            token_type_ids=text_token_type_ids,
        )

        label_embedding_output = self.embeddings_(
            input_ids=label_input_ids,
            token_type_ids=label_token_type_ids,
        )

        text_outputs, label_outputs = self.encoder(
            text_hidden_states=text_embedding_output,
            text_attention_mask=text_extended_attention_mask,
            label_hidden_states=label_embedding_output,
            label_attention_mask=label_extended_attention_mask,
        )

        return text_outputs, label_outputs
