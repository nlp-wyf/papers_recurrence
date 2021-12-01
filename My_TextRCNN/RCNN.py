# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class RCNN(nn.Module):
    """
    Recurrent Convolutional Neural Networks for Text Classification
    """
    def __init__(self, embed_dim=300, hidden_size=256, num_layers=2, num_classes=2, dropout=0.5):
        super(RCNN, self).__init__()

        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2 + embed_dim, num_classes)

    def forward(self, x):
        # x.shape [batch_size, seq_len, embed_dim]
        out, _ = self.lstm(x)  # [batch_size, seq_len, 2*hidden_dim]
        out = torch.cat((x, out), 2)  # [batch_size, seq_len, embed+ 2*hid_dim]
        out = F.relu(out)
        out = out.permute(0, 2, 1)  # [batch_size, embed+2*hidden_dim, seq_len]
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        out = self.fc(out)   # [batch_size, num_classes]
        return out


if __name__ == '__main__':
    # x.shape [batch_size, seq_len, embed_dim]
    x = torch.randn((2, 10, 300))
    model = RCNN()
    out = model(x)
    print(out.shape)
