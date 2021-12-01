# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPCNN(nn.Module):
    """
    Deep Pyramid Convolutional Neural Networks for Text Categorization
    """
    def __init__(self, num_filters=250, embed_dim=300, num_classes=2):
        super(DPCNN, self).__init__()
        self.conv_region = nn.Conv2d(1, num_filters, (3, embed_dim), stride=1)
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        # padding（int ，tuple）–填充的大小。如果为 int ，则在所有边界中使用相同的填充。
        # 如果是4 tuple ，则使用(padding_left,padding_right,padding_top,padding_bottom)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # x.shape [batch_size, seq_len, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, emb]

        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]
        
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        
        while x.size()[2] >= 2:
            x = self._block(x)
        # [batch_size, 250, 1, 1]
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding2(x)
        # [batch_size, 250, seq_len-1, 1]
        px = self.max_pool(x)
        # [batch_size, 250, ((seq_len-1-3) / 2)+1), 1]
        x = self.padding1(px)
        # [batch_size, 250, (seq_len-1-3) / 2)+1 + 2, 1]
        x = F.relu(x)
        x = self.conv(x)
        # [batch_size, 250, (seq_len / 2) + 1 - 3 + 1, 1]
        x = self.padding1(x)
        # [batch_size, 250, (seq_len / 2) + 1 - 3 + 1 + 2, 1]
        x = F.relu(x)
        x = self.conv(x)
        # [batch_size, 250, (seq_len / 2) + 1 - 3 + 1, 1]
        # Short Cut
        x = x + px
        return x


if __name__ == '__main__':
    # x.shape [batch_size, seq_len, embed_dim]
    x = torch.randn((2, 10, 300))
    model = DPCNN()
    out = model(x)
    print(out.shape)
