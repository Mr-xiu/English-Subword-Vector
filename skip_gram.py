import torch
from torch.utils.data import Dataset
import random


class SkipGram(torch.nn.Module):
    """
    基于pytorch构建的SkipGram模型
    """

    def __init__(self, vocab_size, embedding_size, init_range=0.1):
        """
        SkipGram的初始化方法
        :param vocab_size: 传入词表的大小
        :param embedding_size: 嵌入向量的维度大小
        :param init_range: embedding层初始化权重的范围
        """
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size  # 词表大小
        self.embedding_size = embedding_size  # 嵌入向量的维度大小

        # 构造一个词向量参数
        # 参数的初始化方法为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            padding_idx=None,
            max_norm=None,
            norm_type=2,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=torch.FloatTensor(vocab_size, embedding_size).uniform_(-init_range, init_range)
        )

        # 构造另外一个词向量参数
        self.embedding_out = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            padding_idx=None,
            max_norm=None,
            norm_type=2,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=torch.FloatTensor(vocab_size, embedding_size).uniform_(-init_range, init_range)
        )

    # 定义网络的前向计算逻辑
    def forward(self, center_words, target_words, label):
        """
        前向传播并计算loss
        :param center_words: tensor类型，表示中心词
        :param target_words: tensor类型，表示目标词
        :param label: tensor类型，表示词的样本标签（正样本为1，负样本为0）
        :return: 前向传播得到的loss
        """
        # 通过embedding参数，将mini-batch中的词转换为词向量
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)

        # 通过点乘的方式计算中心词到目标词的输出概率
        word_sim = torch.sum(center_words_emb * target_words_emb, dim=-1)

        # 通过估计的输出概率定义损失函数
        # 包含sigmoid和cross entropy两步
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(word_sim, label.float())
        # 返回loss
        return loss


class SkipGramDataset(Dataset):
    def __init__(self, corpus, window_size, negative_sample_num):
        """
        初始化数据集的方法
        :param corpus: 传入语料
        :param window_size: window_size代表了window_size的大小，程序会根据window_size从左到右扫描整个语料
        :param negative_sample_num: negative_sample_num代表了对于每个正样本，我们需要随机采样多少负样本用于训练
        """
        self.corpus = corpus
        self.window_size = window_size
        self.negative_sample_num = negative_sample_num
        # 先构建正样本表，避免随机选择负样本时误选择到正样本
        self.positive_target = dict()
        for index in range(len(corpus)):
            positive_word_range = (max(0, index - window_size), min(len(corpus) - 1, index + window_size))
            if not corpus[index] in self.positive_target:
                self.positive_target[corpus[index]] = set()
            for i in range(positive_word_range[0], positive_word_range[1] + 1):
                self.positive_target[corpus[index]].add(corpus[i])

    def __len__(self):
        return len(self.corpus) * (2 * self.window_size + self.negative_sample_num)

    def __getitem__(self, index):
        """
        读取数据集中的项目
        """
        if index < len(self.corpus) * 2 * self.window_size:
            # 正样本定义域，根据 window_size 均匀划分，
            # 从而选择唯一的 center_word 与 target_word
            center_index = index // (2 * self.window_size)
            shift_index = index % (2 * self.window_size) - self.window_size
            if shift_index >= 0:
                shift_index += 1
            target_index = max(0, min(len(self.corpus) - 1, center_index + shift_index))
            if target_index == center_index:
                if target_index - 1 >= 0:
                    target_index -= 1
                else:
                    target_index += 1
            return self.corpus[center_index], self.corpus[target_index], 1
        else:
            # 负样本定义域
            # 根据 negative_size 找到唯一 center_word，并随机采集负样本
            center_index = (index - len(self.corpus) * 2 * self.window_size) // self.negative_sample_num
            while True:
                target_index = random.randint(0, len(self.corpus) - 1)
                if self.corpus[target_index] not in self.positive_target[self.corpus[center_index]]:
                    return self.corpus[center_index], self.corpus[target_index], 0
