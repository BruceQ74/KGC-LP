# coding = utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import math

from pytorch_transformers import BertModel,BertConfig

class BiLSTM(nn.Module):
    def __init__(self, in_feature, out_feature, num_layers=1, batch_first = True):
        super(BiLSTM, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=in_feature,
            hidden_size=out_feature,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True
        )

    def rand_init_hidden(self, batch_size, device):
        return (torch.zeros(2 * self.num_layers, batch_size, self.out_feature).to(device),
                torch.zeros(2 * self.num_layers, batch_size, self.out_feature).to(device))

    def forward(self, input):
        batch_size, seq_len, hidden_size = input.shape
        hidden = self.rand_init_hidden(batch_size, input.device)
        output, hidden = self.lstm(input, hidden)
        return output.contiguous().view(batch_size, seq_len, self.out_feature * 2)

class Transformer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=512, dropout=0.2, num_layers=1):
        super(Transformer, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, input):
        output = self.transformer(input.transpose(0, 1))
        return output.transpose(0, 1)

class KGCBaseline(nn.Module):
    def __init__(self, label_size, bert_path = None, encoder = "transformer", num_layers = 2):
        super(KGCBaseline, self).__init__()
        bert_config = BertConfig.from_pretrained(bert_path)
        self.hidden_size = bert_config.hidden_size
        self.bert = BertModel.from_pretrained(bert_path)
        self.num_layers = num_layers
        self.label_size = label_size
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        if encoder == "transformer":
            self.entity_encoder = Transformer(self.hidden_size, nhead=8, dim_feedforward=2048, dropout=0.1,
                                              num_layers=num_layers)
        elif encoder == "bilstm":
            self.entity_encoder = BiLSTM(in_feature=self.hidden_size, out_feature=self.hidden_size, num_layers=num_layers,
                                          batch_first=True)
        self.compress = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, label_size)

    def _reset_params(self, initializer):
        for child in self.children():
            if type(child) == BertModel:
                continue
            for p in child.parameters():
                if p.requires_grad:
                    if len(p.shape) > 1:
                        initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_valid_seq_output(self, sequence_output, valid_ids):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp
        return valid_output

    def get_entity(self, sequence_output):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, feat_dim, dtype=sequence_output.dtype, device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][0]
            valid_output[i] = temp
        return valid_output

    def forward(self, input_ids1, input_ids2, labels = None, token_type_ids = None):
        input_out1, _ = self.bert(input_ids1, token_type_ids)
        input_out2, _ = self.bert(input_ids2, token_type_ids)

        input_out1 = self.get_entity(input_out1)
        input_out2 = self.get_entity(input_out2)
        
        pair_out = torch.cat([input_out1, input_out2], dim=-1)
        pair_out = self.dropout(pair_out)
        pair_out = self.compress(pair_out)

        logits = self.classifier(pair_out)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index = 0)
            total_loss = loss_fct(logits.view(-1, self.label_size), labels.view(-1))
            return total_loss, logits

        return logits