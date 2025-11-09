from torch.utils.data import Dataset
import datetime
import numpy as np
import random
'''
定义一个名为DateDataset的类，是继承自torch.utils.data.Dataset的自定义数据集类。
该数据集用于生成随机的日期数据，并进行数据预处理和编码。
'''
class DateDataset(Dataset):
    def __init__(self, n):
        # 初始化中文和英文日期列表
        self.date_cn = []
        self.date_en = []
        for _ in range(n):
            #随机生成年月日
            year = random.randint(1950, 2050)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date = datetime.date(year, month, day)
            # 格式化日期并添加到对应列表(中文格式为"年-月-日"，英文格式为"日/月/年")
            self.date_cn.append(date.strftime("%y-%m-%d"))  # e.g. 23-03-15
            self.date_en.append(date.strftime("%d/%b/%Y"))  # e.g. 11/May/2022
        # 创建一个词汇集，包含所有可能出现的字符
        self.vocab = set([str(i) for i in range(0, 10)] + ["-","/"] + [i.split("/")[1] for i in self.date_en])
        # 创建一个词汇到索引的映射字典，其中<PAD>用于填充，<SOS>表示序列开始，<EOS>表示序列结束
        self.word2index = {v: i for i, v in enumerate(sorted(self.vocab), start=3)}
        self.word2index["<SOS>"] = 0
        self.word2index["<EOS>"] = 1
        self.word2index["<PAD>"] = 2
        # 将开始、结束和填充标记的索引添加到词汇集中
        self.vocab.add("<SOS>")
        self.vocab.add("<EOS>")
        self.vocab.add("<PAD>")
        # 创建一个索引到词汇的映射字典
        self.index2word = {i: v for v, i in self.word2index.items()}
        # 初始化输入和目标列表
        self.input, self.target = [], []
        # 对date_cn和date_en进行编码处理（tokenize），每次循环编码一个序列
        for cn_date, en_date in zip(self.date_cn, self.date_en):
            # 将日期字符串转换为索引列表，然后添加到输入和目标列表
            # 输入序列为逐字符转换
            self.input.append([self.word2index[v] for v in cn_date])
            # 输出序列起始为<SOS>，结束为<EOS>，中间部分逐字符转换，其中月份缩写整体（三个字符）作为一个token
            self.target.append(        # target序列长度为11，注意target序列长度应与RNN解码器的最大长度一致
                [self.word2index["<SOS>"], ] +
                [self.word2index[v] for v in en_date[:3]] +
                [self.word2index[en_date[3:6]]] +   
                [self.word2index[v] for v in en_date[6:]] +
                [self.word2index["<EOS>"], ]
            )
        # 将输入和目标列表转换为numpy数组，每行为一个序列
        self.input, self.target = np.array(self.input), np.array(self.target)

    def __len__(self):
        # 返回数据集的长度，即输入的数量
        return len(self.input)

    def __getitem__(self, index):
        # 返回给定索引的输入、目标和目标的长度
        return self.input[index], self.target[index], len(self.target[index])
    @property
    def num_words(self):
        # 返回词汇表的大小
        return len(self.vocab)

#test
if __name__ == "__main__":
    dataset = DateDataset(1000)
    print(dataset.date_cn[:5])
    print(dataset.date_en[:5])
    print(dataset.input[:5])
    print(dataset.target[:5])
    print(dataset.num_words)
