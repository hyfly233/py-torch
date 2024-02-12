import pprint
from collections import Counter

import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, logging, DownloadConfig
from torch.utils.data import DataLoader

from dataset import SimpleDataset, collate_fn
from translation_model import Encoder, Decoder, Seq2Seq

# 用 WMT 数据集（尝试 English ↔ Chinese）
# 打开 INFO 日志，显示下载进度
logging.set_verbosity_info()

# 2. 自定义缓存目录（默认是 ~/.cache/huggingface/datasets）
# cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
download_config = DownloadConfig(
    # cache_dir=cache_dir,  # 改成你想要的路径
    resume_download=True,
)

# 尝试加载英文->中文的数据集（不同 HF 数据集可能使用不同的配置名称）
# 假设我们希望 src=English, tgt=Chinese
try:
    # 说明：
    # - 如果你想使用真实的 HuggingFace 上的中英平行语料，请确认所选数据集与配置（config）支持 zh/en 对应项，
    #   并把 load_dataset 的第一个/第二个参数替换为正确的数据集名和配置（例如某些 wmt 版本使用 "zh-en" 或 "en-zh"）。
    # - dataset.collate_fn 使用了 batch_first=True，这里我们在训练前把 (batch, seq) -> (seq, batch)，以匹配 translation_model 的输入需求。
    raw_train = load_dataset(
        "wmt14",  # 这里尝试 wmt14 的 zh-en 配置；根据可用数据集调整（见下面的说明）
        "zh-en",
        split="train",
        download_config=download_config,
    )
    pprint.pprint(raw_train.cache_files)
    # 只取前1000条数据（示例）
    raw_train = raw_train.select(range(1000))

    src_sentences = [ex["translation"]["en"] for ex in raw_train]
    trg_sentences = [ex["translation"]["zh"] for ex in raw_train]

except Exception as e:
    print(
        "Warning: failed to load HF zh-en dataset (falling back to a small synthetic corpus):",
        e,
    )
    # 备选（离线/快速测试）: 构造一个小的中英对照语料用于 demo
    english_samples = [
        "Hello world",
        "How are you",
        "I love programming",
        "This is a test",
        "Good morning",
        "Thank you",
        "See you later",
        "I like pizza",
        "What is your name",
        "Have a nice day",
    ]
    chinese_samples = [
        "你好 世界",
        "你好吗",
        "我 爱 编程",
        "这是 一个 测试",
        "早上 好",
        "谢谢 你",
        "回头见",
        "我 喜欢 披萨",
        "你 叫什么 名字",
        "祝 你 有 美好 的 一天",
    ]
    # 将样本重复拼成 N 条
    N = 1000
    src_sentences = [english_samples[i % len(english_samples)] for i in range(N)]
    trg_sentences = [chinese_samples[i % len(chinese_samples)] for i in range(N)]


# 特殊符号
special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]


# 构建词表（同原逻辑）
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


# 英文分词（简单）
def simple_tokenizer(text):
    return text.lower().split()


# 中文分词（使用 jieba）
def chinese_tokenizer(text):
    # 如果句子里已经有空格分隔的词（如上面的 synthetic sample），jieba 也会处理得很好
    return jieba.lcut(text)


# 真实数据集（例：限制词表大小以便快速 demo）
max_size = 20000
# 对于英文（src）使用空格分词；对于中文（tgt）使用 jieba
src_vocab = build_vocab(src_sentences, simple_tokenizer, min_freq=1, max_size=max_size)
trg_vocab = build_vocab(trg_sentences, chinese_tokenizer, min_freq=1, max_size=max_size)

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
# 启用 shuffle（训练需要），并在 CUDA 时启用 pin_memory
pin_memory = True if str(device).startswith("cuda") else False

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=True,
    pin_memory=pin_memory,
)

# 模型
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters())
# 注意：loss 的 ignore_index 应该使用目标语言的 <pad>
criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab["<pad>"])

print("Model initialized, starting training...")

# 训练循环
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0

    for src, trg in dataloader:
        # collate_fn 返回 (batch, seq_len) 因为我们在 dataset.collate_fn 中使用了 batch_first=True
        # 但 translation model 期望 (seq_len, batch), 因此需要转置
        src = src.permute(1, 0).to(device)  # -> [src_len, batch_size]
        trg = trg.permute(1, 0).to(device)  # -> [trg_len, batch_size]

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        # 忽略第 0 个 timestep (<bos>)，并把预测与目标展开用于计算交叉熵
        # use reshape (or .contiguous().view) because slicing may produce non-contiguous tensors
        output = output[1:].reshape(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")

# 保存模型与词表
torch.save(src_vocab, "src_vocab.pth")
torch.save(trg_vocab, "trg_vocab.pth")
torch.save(model.state_dict(), "translation_model.pth")
