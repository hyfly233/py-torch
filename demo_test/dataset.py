import spacy
import torch
from torch.nn.utils.rnn import pad_sequence

# 加载 spacy 的英语和德语模型
spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")


def tokenize_en(text):
    """
    使用 spacy 分词，并转换为小写
    """
    return [tok.text.lower() for tok in spacy_en(text)]


def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de(text)]


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, src_sents, tgt_sents, src_vocab, tgt_vocab):
        self.src_sents = src_sents  # 源句列表
        self.tgt_sents = tgt_sents  # 目标句列表
        self.src_vocab = src_vocab  # 源语言词表
        self.tgt_vocab = tgt_vocab  # 目标语言词表

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.src_sents)

    def __getitem__(self, idx):
        """
        返回 idx 对应的样本，包含源句和目标句的张量表示
        句子格式：<bos> ... <eos>
        其中 <bos> 和 <eos> 分别是句子开始和结束的特殊符号
        词表中未出现的词用 <unk> 表示
        词表假设包含 <pad>, <bos>, <eos>, <unk>
        其中 <pad> 用于填充
        例如，假设 src_vocab = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3, "hello":4, "world":5}
        则句子 "Hello unknown world" 会被转换为 [1, 4, 3, 5, 2]
        其中 "Hello" 被转换为 4，"unknown" 不在词表中被转换为 3
        句子张量的形状为 [seq_len]
        例如上面的句子张量为 torch.tensor([1, 4, 3, 5, 2])
        这里假设 <pad>=0, <bos>=1, <eos>=2, <unk>=3
        其他词从 4 开始编号
        例如，假设 tgt_vocab = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3, "hallo":4, "welt":5}
        则句子 "Hallo unbekannt welt" 会被转换为 [1, 4, 3, 5, 2]
        """
        src = [
            self.src_vocab.get(w, self.src_vocab["<unk>"])
            for w in tokenize_en(self.src_sents[idx])
        ]
        tgt = [
            self.tgt_vocab.get(w, self.tgt_vocab["<unk>"])
            for w in tokenize_de(self.tgt_sents[idx])
        ]
        return torch.tensor(
            [self.src_vocab["<bos>"]] + src + [self.src_vocab["<eos>"]]
        ), torch.tensor([self.tgt_vocab["<bos>"]] + tgt + [self.tgt_vocab["<eos>"]])


def collate_fn(batch):
    """
    用于 batch 填充
    """
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=0)  # 假设 <pad>=0
    tgt_batch = pad_sequence(tgt_batch, padding_value=0)
    return src_batch, tgt_batch
