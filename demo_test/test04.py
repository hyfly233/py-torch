import os
import pprint
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, logging, DownloadConfig
from torch.utils.data import DataLoader

from dataset import SimpleDataset, collate_fn
from translation_model import Encoder, Decoder, Seq2Seq

# 用 WMT14 数据集
# 打开 INFO 日志，显示下载进度
logging.set_verbosity_info()

# 2. 自定义缓存目录（默认是 ~/.cache/huggingface/datasets）
# cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
download_config = DownloadConfig(
    # cache_dir=cache_dir,  # 改成你想要的路径
    resume_download=True,
)

raw_train = load_dataset(
    "wmt14",
    "de-en",
    split="train",  # 只要 train
    # cache_dir=download_config.cache_dir,
    download_config=download_config,
)

# 4. 查看缓存下的具体文件
pprint.pprint(raw_train.cache_files)

src_sentences = [ex["translation"]["en"] for ex in raw_train]
trg_sentences = [ex["translation"]["de"] for ex in raw_train]


# 特殊符号
# 2. 重写 build_vocab，支持 min_freq / max_size
special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]


# 构建词表
def build_vocab(sentences, tokenizer, min_freq=2, max_size=30000):
    counter = Counter()
    for sent in sentences:
        counter.update(tokenizer(sent))
    vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
    for token, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_size + len(special_tokens):
            break
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def simple_tokenizer(text):
    return text.lower().split()


# 真实数据集
# max_size=20000
max_size = 50
src_vocab = build_vocab(src_sentences, simple_tokenizer, min_freq=5, max_size=max_size)
trg_vocab = build_vocab(trg_sentences, simple_tokenizer, min_freq=5, max_size=max_size)

INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(trg_vocab)

print("src_vocab size =", INPUT_DIM, "trg_vocab size =", OUTPUT_DIM)


ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BATCH_SIZE = 10
N_EPOCHS = 10

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

print(f"Using device: {device}")

# 数据集和加载器
dataset = SimpleDataset(src_sentences, trg_sentences, src_vocab, trg_vocab)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# 模型
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=src_vocab["<pad>"])

print("Model initialized, starting training...")

# 训练循环
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0

    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")

# 保存模型
torch.save(src_vocab, "src_vocab.pth")
torch.save(trg_vocab, "trg_vocab.pth")
torch.save(model.state_dict(), "translation_model.pth")
