import torch
import torch.nn as nn
import torch.nn.functional as F


# 编码器
class InsEncoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_word_embeddings,
                 num_char_embeddings,
                 kernels,
                 num_input_channels,
                 num_output_channels,
                 rnn_hidden_dim,
                 num_layers,
                 bidirectional,
                 word_padding_idx=0,
                 char_padding_idx=0):
        super(InsEncoder, self).__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 嵌入层
        self.word_embeddings = nn.Embedding(
            num_embeddings=num_word_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=word_padding_idx)
        self.char_embeddings = nn.Embedding(
            num_embeddings=num_char_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=char_padding_idx)

        # 卷积层权重
        self.conv = nn.ModuleList([
            nn.Conv1d(
                in_channels=num_input_channels,
                out_channels=num_output_channels,
                kernel_size=kernel) for kernel in kernels
        ])

        # GRU层权重
        self.gru = nn.GRU(
            input_size=embedding_dim * (len(kernels) + 1),
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)

        # ReLU
        self.relu = nn.ReLU(inplace=True)

    def get_char_level_embeddings(self, x):
        # x: (N, seq_len, word_len)
        batch_size, seq_len, word_len = x.size()
        # (N * seq_len, word_len)
        x = x.view(-1, word_len)

        # 嵌入层
        # (N * seq_len, word_len, embedding_dim)
        x = self.char_embeddings(x)

        # 重排使得embedding_dim在第一维
        # (N * seq_len, embedding_dim, word_len)
        x = x.transpose(1, 2)

        # 卷积层
        z = [self.relu(conv(x)) for conv in self.conv]

        # 池化
        z = [F.max_pool1d(zz, zz.size(2)).squeeze(2) for zz in z]
        z = [zz.view(batch_size, seq_len, -1) for zz in z]

        # 连接卷积输出得到字符级嵌入
        # (N, seq_size, embedding_dim * len(kernels))
        z = torch.cat(z, 2)

        return z

    def forward(self, x_word, x_char, x_lengths, device):
        # x_word: (N, seq_size)
        # x_char: (N, seq_size, word_len)

        # 词级嵌入层
        # (N, seq_size, embedding_dim)
        z_word = self.word_embeddings(x_word)

        # 字符级嵌入层
        # (N, seq_size, embedding_dim * len(kernels))
        z_char = self.get_char_level_embeddings(x_char)

        # 连接结果
        # (N, seq_size, embedding_dim * (len(kernels) + 1))
        z = torch.cat([z_word, z_char], 2)

        # 向RNN输入
        num_directions = 2 if self.bidirectional else 1
        initial_h = torch.zeros(self.num_layers * num_directions, z.size(0),
                                self.gru.hidden_size).to(device)
        out, h_n = self.gru(z, initial_h)

        return out


# 解码器
class InsDecoder(nn.Module):
    def __init__(self, rnn_hidden_dim, hidden_dim, output_dim, dropout_p):
        super(InsDecoder, self).__init__()

        # 注意力机制的全连接模型
        self.fc_attn = nn.Linear(rnn_hidden_dim, rnn_hidden_dim)
        self.v = nn.Parameter(torch.rand(rnn_hidden_dim))

        # 全连接参数
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_outputs, apply_softmax=False):

        # 软注意
        z = torch.tanh(self.fc_attn(encoder_outputs))
        z = z.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        z = torch.bmm(v, z).squeeze(1)  # [B*T]
        attn_scores = F.softmax(z, dim=1)
        context = torch.matmul(
            encoder_outputs.transpose(-2, -1),
            attn_scores.unsqueeze(dim=2)).squeeze()
        if len(context.size()) == 1:
            context = context.unsqueeze(0)

        # 全连接层
        z = self.dropout(context)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.dropout(z)
        y_pred = self.fc2(z)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return attn_scores, y_pred


if __name__ == '__main__':
    input_ = torch.LongTensor([[[0, 0, 0, 0, 0, 0, 0], [0, 54, 0, 0, 0, 0, 0],
                                [0, 0, 90, 0, 1, 0, 0], [0, 0, 0, 39, 0, 0, 0],
                                [0, 0, 0, 0, 49, 0, 0], [0, 0, 0, 0, 0, 52, 0],
                                [0, 0, 0, 0, 0, 0, 0]],
                               [[0, 0, 0, 0, 0, 0, 0], [0, 54, 0, 0, 0, 0, 0],
                                [0, 0, 90, 0, 1, 0, 0], [0, 0, 0, 39, 0, 0, 0],
                                [0, 0, 0, 0, 49, 0, 0], [0, 0, 0, 0, 0, 52, 0],
                                [0, 0, 0, 0, 0, 0, 0]],
                               [[0, 0, 0, 0, 0, 0, 0], [0, 54, 0, 0, 0, 0, 0],
                                [0, 0, 90, 0, 1, 0, 0], [0, 0, 0, 39, 0, 0, 0],
                                [0, 0, 0, 0, 49, 0, 0], [0, 0, 0, 0, 0, 52, 0],
                                [0, 0, 0, 0, 0, 0, 0]]])
    print(input_.size())
    batch_size, seq_len, word_len = input_.size()
    x = input_.view(-1, word_len)
    print(x.size())
    char_embeddings = nn.Embedding(100, 128, 0)
    x = char_embeddings(x)
    print(x.size())
    x = x.transpose(1, 2)
    print(x.size())

    convs = nn.ModuleList([
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=kernel)
        for kernel in [3, 5]
    ])

    z = [F.relu(conv(x)) for conv in convs]

    print(z[0].size(), z[1].size())

    z = [F.max_pool1d(zz, zz.size(2)).squeeze(2) for zz in z]

    print(z[0].size(), z[1].size())

    z = [zz.view(batch_size, seq_len, -1) for zz in z]

    print(z[0].size(), z[1].size())

    # 连接卷积输出得到字符级嵌入
    z = torch.cat(z, 2)

    print(z.size())

    gru = nn.GRU(
        input_size=256,
        hidden_size=64,
        num_layers=1,
        batch_first=True,
        bidirectional=False)
    z = gru(z)[0]
    print(z.size())

    fca = nn.Linear(64, 64)
    z = fca(z)
    print(z.size())
