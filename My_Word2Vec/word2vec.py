import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random

import scipy

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

C = 3  # context window
K = 15  # number of negative samples
epochs = 2
MAX_VOCAB_SIZE = 10000
EMBEDDING_SIZE = 100
batch_size = 32
lr = 0.2

with open('text8.train.txt') as f:
    text = f.read()  # 得到文本内容

text = text.lower().split()  # 分割成单词列表
vocab_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))  # 得到单词字典表，key是单词，value是次数
vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values()))  # 把不常用的单词都编码为"<UNK>"
word2idx = {word: i for i, word in enumerate(vocab_dict.keys())}
idx2word = {i: word for i, word in enumerate(vocab_dict.keys())}
word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word2idx, word_freqs):
        ''' text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            word_freqs: the frequency of each word
        '''
        # 通过父类初始化模型，然后重写两个方法
        super(WordEmbeddingDataset, self).__init__()
        # 把单词数字化表示。如果不在词典中，也表示为unk
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]
        # nn.Embedding需要传入LongTensor类型
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)

    def __len__(self):
        # 返回所有单词的总数，即item的总数
        return len(self.text_encoded)

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''
        # 取得中心词
        center_words = self.text_encoded[idx]
        # 先取得中心左右各C个词的索引
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        # 为了避免索引越界，所以进行取余处理
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        # tensor(list)
        pos_words = self.text_encoded[pos_indices]
        """
        torch.multinomial(input, num_samples, replacement=False, out=None) → LongTensor
        作用是对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标。
        输入是一个input张量，一个取样数量，和一个布尔值replacement。
        input张量可以看成一个权重张量，每一个元素代表其在该行中的权重。如果有元素为0，
        那么在其他不为0的元素被取干净之前，这个元素是不会被取到的。
        n_samples是每一行的取值次数，该值不能大于每一样的元素数，否则会报错。
        replacement指的是取样时是否是有放回的取样，True是有放回，False无放回。
        """

        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 每采样一个正确的单词(positive word)，就采样K个错误的单词(negative word)，pos_words.shape[0]是正确单词数量
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], replacement=True)
        # pos_indices  [15304683, 15304684, 15304685, 1, 2, 3]
        # while 循环是为了保证 neg_words中不能包含背景词
        # pos_indices.numpy().tolist()
        while len(set(pos_indices) & set(neg_words.numpy().tolist())) > 0:
            neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_words, pos_words, neg_words


dataset = WordEmbeddingDataset(text, word2idx, word_freqs)
dataloader = tud.DataLoader(dataset, batch_size, shuffle=True)


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        ''' input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]

            return: loss, [batch_size]
        '''
        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (window * 2 * K), embed_size]

        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1]

        pos_dot = torch.bmm(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]

        neg_dot = torch.bmm(neg_embedding, -input_embedding)  # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]

        log_pos = F.logsigmoid(pos_dot).sum(1)  # .sum()结果只为一个数，.sum(1)结果是一维的张量
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg

        return -loss

    def input_embedding(self):
        return self.in_embed.weight.detach().numpy()


model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for e in range(1):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print('epoch', e, 'iteration', i, loss.item())

embedding_weights = model.input_embedding()
torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))


def find_nearest(word):
    # 写个函数，找出与某个词相近的一些词，比方说输入 good，他能帮我找出 nice，better，best 之类的
    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx2word[i] for i in cos_dis.argsort()[:10]]


for word in ["two", "america", "computer"]:
    print(word, find_nearest(word))
