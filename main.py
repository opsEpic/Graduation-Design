import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import cv2

import models.resnet50.model as resnet50

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 加载MNIST数据集
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# 加载数据集
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

modeller = resnet50.Modeller(None, device)


def train_epoch() -> None:
    for model_input, expect in train_loader:
        temp_input = []
        for pic in model_input:
            temp_pic = pic[0].numpy()
            temp_pic = cv2.resize(temp_pic, (224, 224), interpolation=cv2.INTER_NEAREST)
            temp_pic = temp_pic.tolist()

            temp_input.append([temp_pic, temp_pic, temp_pic])

        temp_expect = expect.tolist()
        modeller.train(torch.tensor(temp_input), torch.tensor(temp_expect))


def test_epoch() -> int:
    correct = 0
    for model_input, expect in test_loader:
        temp_input = []
        for pic in model_input:
            temp_pic = pic[0].numpy()
            temp_pic = cv2.resize(temp_pic, (224, 224), interpolation=cv2.INTER_NEAREST)
            temp_pic = temp_pic.tolist()

            temp_input.append([temp_pic, temp_pic, temp_pic])

        temp_expect = expect.tolist()
        correct += modeller.test(torch.tensor(temp_input), torch.tensor(temp_expect))

    return correct


train_epoch()
test_epoch()
