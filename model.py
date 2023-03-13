import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from cleam_bert import BertModel


class MHXAttention(nn.Module):
    def __init__(self, params):
        super(MHXAttention, self).__init__()
        config = params["bert_config"]
        self.classes = params["classes"]
        self.words_per_label = params["words_per_label"]
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, source, target, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(source))
        key_layer = self.transpose_for_scores(self.key(target))
        value_layer = self.transpose_for_scores(self.value(target))
        value_layer = value_layer.view(value_layer.size()[:-2] + (self.classes, self.words_per_label, self.head_size))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))

        if attention_mask is not None:
            attention_mask = torch.where(
                attention_mask == 0,
                torch.tensor(-10000, dtype=attention_scores.dtype, device=attention_scores.device),
                torch.zeros(1, dtype=attention_scores.dtype, device=attention_scores.device),
            )
            attention_scores = attention_scores + attention_mask

        attention_scores = attention_scores.view(attention_scores.size()[:-2] + (self.classes, 1, self.words_per_label))
        attention_probs = F.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer).squeeze(-2)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size()[:-2] + (self.all_head_size,))

        return context_layer


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        hidden_size = params["bert_config"].hidden_size
        self.label_input_ids = params["label_tokens"]["input_ids"]
        self.label_attention_mask = params["label_tokens"]["attention_mask"]

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.embeddings_ = copy.deepcopy(self.bert.embeddings)
        self.bert.encoder.layer_ = copy.deepcopy(self.bert.encoder.layer)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

        self.mhxatt = MHXAttention(params)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, texts, masks):
        batch_size = texts.size(0)

        label_input_ids = self.label_input_ids.repeat(batch_size, 1)
        label_attention_mask = self.label_attention_mask.repeat(batch_size, 1)

        text_outputs, label_outputs = self.bert(
            text_input_ids=texts,
            text_attention_mask=masks,
            label_input_ids=label_input_ids,
            label_attention_mask=label_attention_mask,
        )

        text_cls_state = text_outputs[:, 0].unsqueeze(1)
        label_words_states = label_outputs[:, 1:-1]
        label_words_attention_mask = label_attention_mask[:, 1:-1].view(batch_size, 1, 1, -1)
        label_features = self.mhxatt(text_cls_state, label_words_states, label_words_attention_mask)

        label_features = self.LayerNorm(label_features)
        outputs = self.linear(label_features).squeeze(-1)

        return outputs
