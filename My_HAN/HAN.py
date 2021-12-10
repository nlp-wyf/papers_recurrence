import torch
import torch.nn as nn
from torch.nn import functional as F


class HAN(nn.Module):
    '''
    Hierarchical Attention Networks for Document Classification
    '''

    def __init__(self, vocab_size, embedding_dim, gru_size, class_num):
        super(HAN, self).__init__()

        self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        # 词注意力
        self.word_gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_size,
                               num_layers=1, bidirectional=True, batch_first=True)
        # 公式中的u(w)
        self.word_query = nn.Parameter(torch.Tensor(2 * gru_size, 1), requires_grad=True)
        self.word_fc = nn.Linear(2 * gru_size, 2 * gru_size)

        self.sentence_gru = nn.GRU(input_size=2 * gru_size, hidden_size=gru_size,
                                   num_layers=1, bidirectional=True, batch_first=True)
        # 公式中的u(s)
        self.sentence_query = nn.Parameter(torch.Tensor(2 * gru_size, 1), requires_grad=True)
        self.sentence_fc = nn.Linear(2 * gru_size, 2 * gru_size)

        self.class_fc = nn.Linear(2 * gru_size, class_num)

    def word_attention(self, x):
        # x.shape [batch_size, sentence_num, sentence_len]
        sentence_num, sentence_len = x.size(1), x.size(2)
        # [batch_size * sentence_num, sentence_len]
        x = x.view(-1, sentence_len)
        # [batch_size * sentence_num , sentence_len, embedding_dim]
        embed_x = self.word_embed(x)
        # word_output: [batch_size * sentence_num, sentence_len, 2*gru_size]
        word_output, word_hidden = self.word_gru(embed_x)
        # 计算u(it)
        # word_attention: [batch_size * sentence_num, sentence_len, 2*gru_size]
        word_attention = torch.tanh(self.word_fc(word_output))
        # 计算词注意力向量weights: a(it)
        # [batch_size * sentence_num, sentence_len, 1]
        weights = torch.matmul(word_attention, self.word_query)
        # [batch_size * sentence_num, sentence_len, 1]
        weights = F.softmax(weights, dim=1)

        # [batch_size * sentence_num, sentence_len, 1]
        x = x.unsqueeze(2)
        # 去掉x中padding为0位置的attention比重
        # [batch_size * sentence_num, sentence_len, 1]
        weights = torch.where(x != 0, weights, torch.full_like(x, 0, dtype=torch.float))
        # 将x中padding后的结果进行归一化处理，为了避免padding处的weights为0无法训练，加上一个极小值1e-4
        # [batch_size * sentence_num, sentence_len, 1]
        weights = weights / (torch.sum(weights, dim=1).unsqueeze(1) + 1e-4)

        # 计算句子向量si = sum(a(it) * h(it))
        # 维度变化 [batch_size * sentence_num, 2 * gru_size] -> [batch_size, sentence_num, 2 * gru_size]
        sentence_vector = torch.sum(weights * word_output, dim=1).view(-1, sentence_num, word_output.size(2))

        return sentence_vector

    def sentence_attention(self, sentence_vector, x):
        # x.shape [batch_size, sentence_num, sentence_len]
        # sentence_vector.shape [batch_size, sentence_num, 2 * gru_size]

        # sentence_output.shape [batch_size, sentence_num, 2 * gru_size]
        sentence_output, sentence_hidden = self.sentence_gru(sentence_vector)
        # 计算ui
        # sentence_attention.shape [batch_size, sentence_num, 2 * gru_size]
        sentence_attention = torch.tanh(self.sentence_fc(sentence_output))
        # 计算句子注意力向量sentence_weights: a(i)
        # sentence_weights.shape [batch_size, sentence_num, 1]
        sentence_weights = torch.matmul(sentence_attention, self.sentence_query)
        sentence_weights = F.softmax(sentence_weights, dim=1)

        # [batch_size, sentence_num, 1]
        x = torch.sum(x, dim=2).unsqueeze(2)

        # [batch_size, sentence_num, 1]
        sentence_weights = torch.where(x != 0, sentence_weights, torch.full_like(x, 0, dtype=torch.float))
        # [batch_size, sentence_num, 1]
        sentence_weights = sentence_weights / (torch.sum(sentence_weights, dim=1).unsqueeze(1) + 1e-4)

        # 计算文档向量v
        # document_vector.shape [batch_size, 2 * gru_size]
        document_vector = torch.sum(sentence_weights * sentence_output, dim=1)
        return document_vector

    def forward(self, x):
        # x.shape [batch_size, sentence_num, sentence_len]
        sentence_vector = self.word_attention(x)
        document_vector = self.sentence_attention(sentence_vector, x)
        # [batch_size, class_num]
        cls_doc = self.class_fc(document_vector)
        return cls_doc


if __name__ == '__main__':
    model = HAN(vocab_size=3000, embedding_dim=200, gru_size=50, class_num=4)
    # x.shape [batch_size, sentence_num, sentence_len]
    x = torch.zeros(64, 50, 100).long()
    x[0][0][0:10] = 1
    document_class = model(x)
    # [64, 4]
    print(document_class.shape)
