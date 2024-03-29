import numpy as np

import scipy

from queue import PriorityQueue


class SVD:
    """
    SVD类
    """

    def __init__(self, train_path: str = 'data/text8.txt', vocab_max_size=10000):
        """
        初始化的方法
        :param train_path: 训练语料的位置
        :param vocab_max_size: 词表最大的大小
        """
        self.word_vectors = None
        self.corpus = []  # 语料列表
        # 先读取语料
        with open(train_path, 'r', encoding='UTF-8') as f:
            self.corpus = f.read().strip("\n")
            f.close()

        # 预处理语料
        self.preprocess_corpus()

        # 根据词频为每个词构建对应的ID
        self.id2word_dict = {}  # key为id，值为对应的word
        self.word2id_dict = {}  # key为word，值为对应的id
        self.freq_dict = {}  # key为word，值为对应的频率
        self.corpus_size = 0  # 语料中词的数量
        self.create_id()  # 开始构建
        # 词表大小
        self.vocab_size = len(self.word2id_dict) if vocab_max_size >= len(self.word2id_dict) or vocab_max_size == 0 else vocab_max_size

        print(f'构建id完成，语料中共有{self.vocab_size}个词~')

    def preprocess_corpus(self):
        """
        预处理语料的方法
        """
        self.corpus = self.corpus.strip().lower()
        self.corpus = self.corpus.split(" ")

    def create_id(self):
        """
        为语料中的每个词根据出现的频率构建ID
        """
        # 遍历语料
        for word in self.corpus:
            self.corpus_size += 1
            if word not in self.freq_dict:
                self.freq_dict[word] = 1
            else:
                self.freq_dict[word] += 1

        # 按照词频排序
        self.freq_dict = sorted(self.freq_dict.items(), key=lambda x: x[1], reverse=True)

        # 构建ID（频率越高id越小）
        for word, _ in self.freq_dict:
            my_id = len(self.word2id_dict)
            # print(word, _)
            self.word2id_dict[word] = my_id
            self.id2word_dict[my_id] = word

    def build_svd_vector(self, save_path='model/svd.npy', vector_dim=100, window_size=5):
        """
        通过svd构建词向量的方法
        :param save_path: 词向量表保存的位置
        :param vector_dim: 词向量的维度
        :param window_size: 共现窗口的大小
        """
        # 初始化共现矩阵
        co_matrix = np.zeros((self.vocab_size, self.vocab_size), dtype='uint16')
        # 从左到右，枚举每个中心点的位置
        for center_word_idx in range(len(self.corpus)):
            # 当前的中心词
            center_word = self.corpus[center_word_idx]
            if self.word2id_dict[center_word] >= self.vocab_size:
                continue
            # 上下文词列表
            context_words_list = self.corpus[max(0, center_word_idx - window_size): center_word_idx] + self.corpus[
                                                                                                       center_word_idx + 1: center_word_idx + window_size + 1]
            # 更新共现矩阵
            for context_word in context_words_list:
                if self.word2id_dict[context_word] < self.vocab_size:
                    co_matrix[self.word2id_dict[center_word], self.word2id_dict[context_word]] += 1
        print('构建共现矩阵完成，开始进行SVD分解~')
        # 对共现矩阵进行SVD分解，得到U、Σ和V矩阵
        # U, S, V = scipy.linalg.svd(co_matrix)
        # 先转换为稀疏矩阵
        co_matrix = scipy.sparse.csr_matrix(co_matrix).asfptype()
        U, S, V = scipy.sparse.linalg.svds(co_matrix, k=vector_dim)
        print(f'计算了{len(S)}个奇异值，计算的奇异值之和为：{np.sum(S)}~')
        self.word_vectors = U

        np.save(save_path, np.array(self.word_vectors))

    def load_svd_vector(self, model_path='model/svd.npy'):
        self.word_vectors = np.load(model_path)
        print('模型加载成功！')

    def get_cos_sim(self, word1, word2):
        """
        计算传入词余弦相似度的方法
        :param word1: 第一个词
        :param word2: 第二个词
        :return: 这两个词的余弦相似度
        """
        # 如果词不在词典中，就忽略
        if (word1 not in self.word2id_dict) or (word2 not in self.word2id_dict) or (self.word2id_dict[word1] >= self.vocab_size) or (
                self.word2id_dict[word2] >= self.vocab_size):
            return 0
        word1_vec = self.word_vectors[self.word2id_dict[word1]]
        word2_vec = self.word_vectors[self.word2id_dict[word2]]
        cos_sim = np.dot(word1_vec, word2_vec) / (np.linalg.norm(word1_vec) * np.linalg.norm(word2_vec))
        return cos_sim


def get_svd_result(has_train=True, vocab_max_size=100000, vector_dim=100, window_size=5, model_path='model/svd.npy',
                   test_path='data/wordsim353_agreed.txt',
                   result_path='data/svd_result.txt'):
    """
    在wordsim353_agreed.txt中测试训练模型表现的方法
    :param has_train: 若为True，表示模型已经训练成功，只需在内存中加载svd词向量矩阵
    :param vocab_max_size: 词表最大的大小
    :param vector_dim: 词向量的维度
    :param window_size: 共现窗口的大小
    :param model_path: svd词向量矩阵的位置
    :param test_path: 测试文件的位置
    :param result_path: 测试结果文件保存的位置
    """
    if not has_train:
        svd = SVD(train_path='data/text8.txt', vocab_max_size=vocab_max_size)
        svd.build_svd_vector(save_path=model_path, vector_dim=vector_dim, window_size=window_size)
    else:
        # 读取模型
        svd = SVD(train_path='data/text8.txt', vocab_max_size=vocab_max_size)
        svd.load_svd_vector(model_path)

    # 读取测试文本
    with open(test_path, 'r', encoding='UTF-8') as f:
        test_lines = f.readlines()
        f.close()
    f = open(result_path, 'w', encoding='UTF-8')
    # 开始按行测试
    for i in range(len(test_lines)):
        line = test_lines[i].strip('\n').split('\t')
        if len(line) == 0:
            continue
        word1 = line[1]
        word2 = line[2]
        true_result = line[3]
        sim_sgns = svd.get_cos_sim(word1, word2)
        f.write(f'{word1}\t{word2}\t{true_result}\t{(sim_sgns + 1) * 5:.2f}\n')
    f.close()


def get_10_most_similar(word, has_train=True, vocab_max_size=100000, vector_dim=100, window_size=5, model_path='model/svd.npy'):
    """
    测试与输入词十个最相似词的方法
    :param word: 输入的词
    :param has_train: 若为True，表示模型已经训练成功，只需在内存中加载svd词向量矩阵
    :param vocab_max_size: 词表最大的大小
    :param vector_dim: 词向量的维度
    :param window_size: 共现窗口的大小
    :param model_path: svd词向量矩阵的位置
    """
    if not has_train:
        svd = SVD(train_path='data/text8.txt', vocab_max_size=vocab_max_size)
        svd.build_svd_vector(save_path=model_path, vector_dim=vector_dim, window_size=window_size)
    else:
        # 读取模型
        svd = SVD(train_path='data/text8.txt', vocab_max_size=vocab_max_size)
        svd.load_svd_vector(model_path)
    if word not in svd.word2id_dict:
        print(f'error:{word}没有在词表中~')
        return
    q = PriorityQueue()  # 最相似的词的序列
    for i in range(svd.vocab_size):
        word2 = svd.id2word_dict[i]
        if word2 == word:
            continue
        sim_sgns = svd.get_cos_sim(word, word2)
        q.put((-sim_sgns, word2))
    print(f'与{word}最相似的十个词为：')
    for i in range(10):
        next_item = q.get()
        print(f'{next_item[1]}\t{-next_item[0]:.2f}')


if __name__ == "__main__":
    get_svd_result(has_train=False, test_path='data/wordsim353_agreed.txt', vocab_max_size=100000)
    # get_10_most_similar('study')
