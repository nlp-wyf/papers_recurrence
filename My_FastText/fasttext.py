import torch
import torch.nn as nn


class FastText(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, classes):
        super(FastText, self).__init__()
        # 创建embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed.weight.requires_grad = True  # 需要计算梯度，即embedding层需要被训练
        self.fc = nn.Sequential(  # 序列函数
            nn.Linear(embed_dim, hidden_size),  # 这里的意思是先经过一个线性转换层
            nn.BatchNorm1d(hidden_size),  # 再进入一个BatchNorm1d
            nn.ReLU(inplace=True),  # 再经过Relu激活函数
            nn.Linear(hidden_size, classes)  # 最后再经过一个线性变换
        )

    def forward(self, x):
        # x.shape [batch_size, seq_len]
        x = self.embed(x)  # 先将词id转换为对应的词向量
        # 将句子中所有的单词在Embedding空间中进行平均
        out = self.fc(torch.mean(x, dim=1))  # 这使用torch.mean()将向量进行平均
        return out


if __name__ == '__main__':
    # x.shape [batch_size, seq_len]
    x = torch.arange(100).reshape(2, 50)
    model = FastText(vocab_size=5000, embed_dim=300, hidden_size=128, classes=2)
    output = model(x)
    print(output.shape)
