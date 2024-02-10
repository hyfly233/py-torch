import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

# 加载测试集，图像被 ToTensor() 转为形状 (C, H, W) 的张量
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        """
        先展平再通过线性栈得到 logits
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)  # 把模型移到选定设备
model.load_state_dict(
    torch.load("model.pth", weights_only=True)
)  # 从 model.pth 加载权重

# 类别列表
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()  # 切换到评估模式
x, y = (
    test_data[0][0],
    test_data[0][1],
)  # 取测试集的第一张图像和标签, x 形状为 (1, 28, 28)
with torch.no_grad():  # 评估时不需要计算梯度
    x = x.to(device)
    pred = model(x)  # 得到形状 (1,10) 的 logits
    predicted, actual = (
        classes[pred[0].argmax(0)],
        classes[y],
    )  # 取最大 logit 的索引映射为类别名
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
