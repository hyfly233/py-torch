import torch

from translation_model import Encoder, Decoder, Seq2Seq

# 1. 加载词表（训练时用 torch.save(src_vocab, 'src_vocab.pth')、torch.save(trg_vocab, 'trg_vocab.pth') 保存）
src_vocab = torch.load("src_vocab.pth")
trg_vocab = torch.load("trg_vocab.pth")
inv_trg_vocab = {idx: tok for tok, idx in trg_vocab.items()}

# 2. 设备
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

# 3. 重建模型结构并加载权重
INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(trg_vocab)
enc = Encoder(INPUT_DIM, 256, 512, 2, 0.5)
dec = Decoder(OUTPUT_DIM, 256, 512, 2, 0.5)
model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load("translation_model.pth", map_location=device))
model.eval()


# 4. 定义翻译函数
def translate(sentence: str, max_len: int = 50) -> str:
    # 分词并转 id
    tokens = ["<bos>"] + sentence.lower().split() + ["<eos>"]
    src_ids = [src_vocab.get(tok, src_vocab["<unk>"]) for tok in tokens]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(1).to(device)  # [seq_len, 1]

    # 编码
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    # 解码
    trg_indexes = [trg_vocab["<bos>"]]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab["<eos>"]:
            break

    # 转回文本
    trg_tokens = [inv_trg_vocab[idx] for idx in trg_indexes]
    return " ".join(trg_tokens[1:-1])


# 5. 测试
print(translate("Hello"))
# print(translate("Hello, how are you?"))
