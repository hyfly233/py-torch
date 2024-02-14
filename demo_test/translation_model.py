import random

import torch
import torch.nn as nn


# 基于 LSTM 的序列到序列（seq2seq）模型


class Encoder(nn.Module):
    """
    把源序列映射到 LSTM 的隐状态
    nn.Embedding -> dropout -> nn.LSTM
    """

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src 形状 [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return (
            hidden,
            cell,
        )  # hidden、cell 的形状为 [n_layers, batch_size, hid_dim]，可直接用于 decoder 的初始状态


class Decoder(nn.Module):
    """
    给定上一步的 token（或目标 token），输出下一个时间步的词分布
    把单步输入 token 用 embedding 编码（先 unsqueeze(0) 变成 [1, batch]），然后经过 dropout、LSTM，最后经 fc_out 映射到词表维度
    """

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input: [batch_size]
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return (
            prediction,
            hidden,
            cell,
        )  # prediction（[batch_size, output_dim] logits）、hidden、cell


class Seq2Seq(nn.Module):
    """
    在训练时按 teacher forcing 比例逐步解码并收集所有时间步的输出 logits
    用 encoder 得到 hidden, cell。
    用 trg[0, :]（通常是 <sos>）作为初始 decoder 输入。
    对每个时间步 t 从 1 到 trg_len-1：调用 decoder，得到 output（[batch, trg_vocab_size]），写入 outputs[t]，根据 teacher forcing 决定下一步输入是 trg[t] 还是 top1 = output.argmax(1)。
    返回 outputs，形状 [trg_len, batch, trg_vocab_size]（注意 outputs[0] 保持为初始的全零）
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        input = trg[0, :]  # <sos> token

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs
