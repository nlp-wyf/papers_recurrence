# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    Convolutional Neural Networks for Sentence Classification
    """
    def __init__(self, embed_dim=300, num_filters=256, num_classes=2, dropout=0.5):
        super(TextCNN, self).__init__()
        self.filter_sizes = [2, 3, 4]
        self.convs = nn.ModuleList(
            # conv : [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
            [nn.Conv2d(1, num_filters, (k, embed_dim)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        # x的shape为[batch_size, 1, seq, embed]
        #  conv(x)的shape为[batch_size, num_filters, seq, 1]
        x = F.relu(conv(x)).squeeze(3)   # [batch_size, num_filters, seq]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # [batch_size, num_filters]
        return x

    def forward(self, x):
        # x.shape[batch_size, seq_len, embed_dim]
        out = x.unsqueeze(1)   # [batch_size, 1, seq, embed]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)  # [batch_size, config.num_filters * len(config.filter_sizes)]
        out = self.fc(out)  # [batch_size, num_classes]
        return out


if __name__ == '__main__':
    # x.shape [batch_size, seq_len, embed_dim]
    x = torch.randn((2, 10, 300))
    model = TextCNN()
    out = model(x)
    print(out.shape)
