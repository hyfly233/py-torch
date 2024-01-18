import spacy
import torch
from torch.nn.utils.rnn import pad_sequence

spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")


def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en(text)]


def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de(text)]


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, src_sents, tgt_sents, src_vocab, tgt_vocab):
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sents)

    def __getitem__(self, idx):
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
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=0)  # 假设 <pad>=0
    tgt_batch = pad_sequence(tgt_batch, padding_value=0)
    return src_batch, tgt_batch
