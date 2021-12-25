import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import BertConfig, BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

from scipy.stats import spearmanr
from tqdm import tqdm

# 基本参数
EPOCHS = 1
SAMPLES = 10000
BATCH_SIZE = 32
LR = 1e-5
DROPOUT = 0.3
MAXLEN = 64
POOLING = 'cls'  # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 预训练模型目录
model_path = 'hfl/chinese-bert-wwm-ext'

# 微调后参数存放位置
SAVE_PATH = './saved_model/simcse_unsup.pt'
# 数据目录
STS_TRAIN = './data/STS-B/cnsd-sts-train_unsup.txt'
STS_DEV = './data/STS-B/cnsd-sts-dev.txt'
STS_TEST = './data/STS-B/cnsd-sts-test.txt'


def load_sts_data(path):
    """加载数据集"""
    with open(path, 'r', encoding='utf8') as f:
        all_data = []
        for line in f.readlines():
            line = line.strip('\n')
            items = line.split("||")
            all_data.append((items[1], items[2], items[3]))
        return all_data


def load_sts_data_v2(path):
    with open(path, 'r', encoding='utf8') as f:
        all_data = []
        for line in f.readlines():
            line = line.strip('\n')
            all_data.append(line)
        return all_data


class TrainDataset(Dataset):
    """数据集, 重写__getitem__和__len__方法"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        # 添加自身两次, 经过bert编码之后, 互为正样本
        return tokenizer([text, text],
                         max_length=MAXLEN,
                         truncation=True,
                         padding='max_length',
                         return_tensors='pt')

    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])


class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法"""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return tokenizer(text,
                         max_length=MAXLEN,
                         truncation=True,
                         padding='max_length',
                         return_tensors='pt')

    def __getitem__(self, index: int):
        da = self.data[index]
        return self.text_2_id(da[0]), self.text_2_id(da[1]), int(da[2])


class SimCSE(nn.Module):
    """SimCSE无监督模型定义"""

    def __init__(self, pretrained_model, pooling):
        super(SimCSE, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        # 修改config的dropout系数
        config.attention_probs_dropout_prob = DROPOUT
        config.hidden_dropout_prob = DROPOUT
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids,
                        attention_mask,
                        token_type_ids,
                        output_hidden_states=True)

        if self.pooling == 'cls':
            # [batch, 768]
            return out.last_hidden_state[:, 0]
        if self.pooling == 'pooler':
            # [batch, 768]
            return out.pooler_output
        if self.pooling == 'last-avg':
            # [batch, 768, seq_len]
            last = out.last_hidden_state.transpose(1, 2)
            # [batch, 768]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        if self.pooling == 'first-last-avg':
            # [batch, 768, seq_len]
            first = out.hidden_states[1].transpose(1, 2)
            # [batch, 768, seq_len]
            last = out.hidden_states[-1].transpose(1, 2)
            # [batch, 768]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)
            # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
            # [batch, 2, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)
            # [batch, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)


def SimCSE_loss(y_pred, tau=0.05):
    """
    无监督的损失函数
    y_pred: bert的输出, [batch_size * 2, 768]
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(0, y_pred.shape[0], device=DEVICE)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    # similarities.shape [batch * 2, batch * 2]
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    # 屏蔽对角矩阵，即自身相等的loss
    similarities = similarities - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 相似度矩阵除以温度系数
    similarities = similarities / tau
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(similarities, y_true)
    # F.cross_entropy默认已经是mean均值的形式了，因此torch.mean操作是多余的，没必要再算一次
    # loss = torch.mean(loss)
    return loss


def train(model, train_dl, dev_dl, optimizer):
    """模型训练函数"""
    best = 0
    model.train()
    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
        for batch_idx, source in enumerate(tqdm(train_dl), start=1):
            # source[input_ids]、source[attention_mask]、source[token_type_ids]的shape
            # 为[batch, 2, seq_len]
            # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
            real_batch_num = source.get('input_ids').shape[0]
            input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
            attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
            token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)
            # out.shape [batch_size * 2, 768]
            out = model(input_ids, attention_mask, token_type_ids)
            loss = SimCSE_loss(out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                logger.info(f'loss: {loss.item():.4f}')
                corrcoef = eval(model, dev_dl)
                model.train()
                if best < corrcoef:
                    best = corrcoef
                    torch.save(model.state_dict(), SAVE_PATH)
                    logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")

    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')


def eval(model, dataloader):
    """模型评估函数
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    label_array = np.array([])
    with torch.no_grad():
        for source, target, label in dataloader:
            # source  [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
            # source_pred.shape [batch_size, 768]
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target  [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(DEVICE)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(DEVICE)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(DEVICE)
            # target_pred.shape [batch_size, 768]
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            # sim.shape [batch]
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            # sim_tensor.shape [batch]
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            label_array = np.append(label_array, np.array(label))
    # corrcoef
    return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation


if __name__ == '__main__':
    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # load data
    train_data = load_sts_data_v2(STS_TRAIN)
    # 随机采样
    train_data = random.sample(train_data, SAMPLES)
    dev_data = load_sts_data(STS_DEV)
    test_data = load_sts_data(STS_TEST)
    train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE)
    dev_dataloader = DataLoader(TestDataset(dev_data), batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(TestDataset(test_data), batch_size=BATCH_SIZE)
    # load model
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimCSE(pretrained_model=model_path, pooling=POOLING)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # train
    train(model, train_dataloader, dev_dataloader, optimizer)
    # eval
    model.load_state_dict(torch.load(SAVE_PATH))
    dev_corrcoef = eval(model, dev_dataloader)
    test_corrcoef = eval(model, test_dataloader)
    logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
    logger.info(f'test_corrcoef: {test_corrcoef:.4f}')
