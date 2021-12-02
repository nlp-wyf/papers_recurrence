# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRNN_ATT(nn.Module):
    """
    Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
    """
    def __init__(self, embed_dim=300, hidden_size=128, hidden_size2=64, num_layers=2,
                 dropout=0.5, num_classes=2):
        super(TextRNN_ATT, self).__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.tanh1 = nn.Tanh()
        # 随机赋值，有可能出现nan值，导致训练失败
        # self.w = nn.Parameter(torch.Tensor(hidden_size * 2))
        # 改为如下方式:
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size2)
        self.fc = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        # x的shape为 [batch_size, seq_len, embeding]
        H, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size * num_direction]

        M = self.tanh1(H)
        # [batch_size, seq_len, 1]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        # [batch_size, seq_len, hidden_size * num_direction]
        out = H * alpha

        out = torch.sum(out, 1)  # # [batch_size, hidden_size * num_direction]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [batch_size, num_classes]
        return out


if __name__ == '__main__':
    # x.shape [batch_size, seq_len, embed_dim]
    x = torch.randn((2, 10, 300))
    model = TextRNN_ATT()
    out = model(x)
    print(out.shape)
