import json
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset, logging as hf_logging, DownloadConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SimpleDataset, collate_fn
from translation_model import Encoder, Decoder, Seq2Seq

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """配置类，集中管理所有超参数"""

    # 数据配置
    DATASET_NAME = "wmt14"
    LANGUAGE_PAIR = "de-en"
    MAX_VOCAB_SIZE = 30000
    MIN_FREQ = 2
    MAX_SEQUENCE_LENGTH = 100

    # 模型配置
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # 训练配置
    BATCH_SIZE = 32
    N_EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0

    # 其他配置
    SAVE_DIR = Path("checkpoints")
    CACHE_DIR = Path("cache")


class TranslationTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = self._get_device()
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

        # 创建必要目录
        self.config.SAVE_DIR.mkdir(exist_ok=True)
        self.config.CACHE_DIR.mkdir(exist_ok=True)

        logger.info(f"Using device: {self.device}")

    def _get_device(self) -> torch.device:
        """智能设备选择"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch, "mps") and torch.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def load_dataset(self) -> Tuple[List[str], List[str]]:
        """加载和预处理数据集"""
        logger.info("Loading dataset...")

        # 检查缓存
        cache_file = self.config.CACHE_DIR / "processed_data.pkl"
        if cache_file.exists():
            logger.info("Loading from cache...")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        # 设置HuggingFace日志
        hf_logging.set_verbosity_info()

        download_config = DownloadConfig(
            cache_dir=str(self.config.CACHE_DIR),
            resume_download=True,
        )

        # 加载数据集 - 使用较小的子集进行测试
        dataset = load_dataset(
            self.config.DATASET_NAME,
            self.config.LANGUAGE_PAIR,
            split="train[:10000]",  # 只使用前10000条数据进行快速测试
            cache_dir=str(self.config.CACHE_DIR),
            download_config=download_config,
        )

        src_sentences = [ex["translation"]["en"] for ex in dataset]
        trg_sentences = [ex["translation"]["de"] for ex in dataset]

        # 过滤过长的句子
        filtered_data = []
        for src, trg in zip(src_sentences, trg_sentences):
            if (
                len(src.split()) <= self.config.MAX_SEQUENCE_LENGTH
                and len(trg.split()) <= self.config.MAX_SEQUENCE_LENGTH
            ):
                filtered_data.append((src, trg))

        src_sentences, trg_sentences = zip(*filtered_data)
        src_sentences, trg_sentences = list(src_sentences), list(trg_sentences)

        logger.info(f"Loaded {len(src_sentences)} sentence pairs")

        # 缓存处理后的数据
        with open(cache_file, "wb") as f:
            pickle.dump((src_sentences, trg_sentences), f)

        return src_sentences, trg_sentences

    def simple_tokenizer(self, text: str) -> List[str]:
        """简单的分词器"""
        return text.lower().strip().split()

    def build_vocab(
        self, sentences: List[str], min_freq: int = None, max_size: int = None
    ) -> Dict[str, int]:
        """构建词表，支持缓存"""
        if min_freq is None:
            min_freq = self.config.MIN_FREQ
        if max_size is None:
            max_size = self.config.MAX_VOCAB_SIZE

        counter = Counter()
        for sent in tqdm(sentences, desc="Building vocab"):
            counter.update(self.simple_tokenizer(sent))

        vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}

        for token, freq in counter.most_common():
            if freq < min_freq or len(vocab) >= max_size + len(self.special_tokens):
                break
            if token not in vocab:
                vocab[token] = len(vocab)

        return vocab

    def create_model(self, src_vocab_size: int, trg_vocab_size: int) -> nn.Module:
        """创建模型"""
        enc = Encoder(
            src_vocab_size,
            self.config.ENC_EMB_DIM,
            self.config.HID_DIM,
            self.config.N_LAYERS,
            self.config.ENC_DROPOUT,
        )
        dec = Decoder(
            trg_vocab_size,
            self.config.DEC_EMB_DIM,
            self.config.HID_DIM,
            self.config.N_LAYERS,
            self.config.DEC_DROPOUT,
        )
        model = Seq2Seq(enc, dec, self.device).to(self.device)

        # 参数初始化
        def init_weights(m):
            if hasattr(m, "weight") and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        model.apply(init_weights)
        return model

    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """训练一个epoch"""
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(dataloader, desc="Training")
        for batch_idx, (src, trg) in enumerate(progress_bar):
            src, trg = src.to(self.device), trg.to(self.device)

            optimizer.zero_grad()
            output = model(src, trg)

            # 计算损失
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.GRADIENT_CLIP
            )

            optimizer.step()
            epoch_loss += loss.item()

            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return epoch_loss / len(dataloader)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        loss: float,
        src_vocab: Dict,
        trg_vocab: Dict,
    ):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": self.config.__dict__,
        }

        # 保存模型检查点
        checkpoint_path = self.config.SAVE_DIR / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # 保存词表
        with open(self.config.SAVE_DIR / "src_vocab.json", "w", encoding="utf-8") as f:
            json.dump(src_vocab, f, ensure_ascii=False, indent=2)

        with open(self.config.SAVE_DIR / "trg_vocab.json", "w", encoding="utf-8") as f:
            json.dump(trg_vocab, f, ensure_ascii=False, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def train(self):
        """主训练函数"""
        # 加载数据
        src_sentences, trg_sentences = self.load_dataset()

        # 构建词表
        logger.info("Building vocabularies...")
        src_vocab = self.build_vocab(src_sentences)
        trg_vocab = self.build_vocab(trg_sentences)

        logger.info(f"Source vocab size: {len(src_vocab)}")
        logger.info(f"Target vocab size: {len(trg_vocab)}")

        # 创建数据集和加载器
        dataset = SimpleDataset(src_sentences, trg_sentences, src_vocab, trg_vocab)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # 创建模型
        model = self.create_model(len(src_vocab), len(trg_vocab))
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # 优化器和损失函数
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        criterion = nn.CrossEntropyLoss(ignore_index=src_vocab["<pad>"])

        logger.info("Starting training...")
        best_loss = float("inf")

        # 训练循环
        for epoch in range(self.config.N_EPOCHS):
            logger.info(f"Epoch {epoch + 1}/{self.config.N_EPOCHS}")

            # 训练
            avg_loss = self.train_epoch(model, dataloader, optimizer, criterion)

            # 获取当前学习率
            current_lr = optimizer.param_groups[0]["lr"]

            # 学习率调度
            old_lr = current_lr
            scheduler.step(avg_loss)
            new_lr = optimizer.param_groups[0]["lr"]

            # 手动记录学习率变化
            if old_lr != new_lr:
                logger.info(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")

            logger.info(
                f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}, LR: {new_lr:.2e}"
            )

        logger.info("Training completed!")


def main():
    config = Config()
    trainer = TranslationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
