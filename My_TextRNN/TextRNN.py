# coding: UTF-8
import torch
import torch.nn as nn


class TextRNN(nn.Module):
    """
    Recurrent Neural Network for Text Classification with Multi-Task Learning
    """
    def __init__(self, embed_dim=300, hidden_size=128, num_layers=2, dropout=0.5, num_classes=2):
        super(TextRNN, self).__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x的shape为 [batch_size, seq_len, embeding]
        out, _ = self.lstm(x)  # [batch_size, seq_len, 2 * hidden_size]
        # out[:, -1:, :]的shape为 [batch_size, 2 * hidden_size]
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        # out的shape为[batch_size, num_classes]
        return out


if __name__ == '__main__':
    # x.shape [batch_size, seq_len, embed_dim]
    x = torch.randn((2, 10, 300))
    model = TextRNN()
    out = model(x)
    print(out.shape)
