import torch
import torch.nn as nn

import models.resizer.model as resizer


class Residual(nn.Module):
    def __init__(self, input_dep: int, layer1: tuple, layer2: tuple, layer3: tuple, down_sample: bool, stride: int,
                 device):
        super().__init__()

        self.down_sample = down_sample
        if down_sample:
            self.conv2d0 = nn.Conv2d(input_dep, layer3[0], 1, stride=stride, padding=0, device=device)
            self.norm2d0 = nn.BatchNorm2d(layer3[0], device=device)

        self.conv2d1 = nn.Conv2d(input_dep, layer1[0], layer1[1], stride=stride if down_sample else 1, padding=(layer1[1] - 1) // 2, device=device)
        self.norm2d1 = nn.BatchNorm2d(layer1[0], device=device)
        self.Relu001 = nn.ReLU()

        self.conv2d2 = nn.Conv2d(layer1[0], layer2[0], layer2[1], padding=(layer2[1] - 1) // 2, device=device)
        self.norm2d2 = nn.BatchNorm2d(layer2[0], device=device)
        self.Relu002 = nn.ReLU()

        self.conv2d3 = nn.Conv2d(layer2[0], layer3[0], layer3[1], padding=(layer3[1] - 1) // 2, device=device)
        self.norm2d3 = nn.BatchNorm2d(layer3[0], device=device)
        self.Relu003 = nn.ReLU()

    def forward(self, model_input):
        residual = model_input
        if self.down_sample:
            residual = self.norm2d0(self.conv2d0(model_input))

        layer1 = self.Relu001(self.norm2d1(self.conv2d1(model_input)))
        layer2 = self.Relu002(self.norm2d2(self.conv2d2(layer1)))
        layer3 = self.Relu003(self.norm2d3(self.conv2d3(layer2)))

        return layer3 + residual


class Stage(nn.Module):
    def __init__(self, input_dep: int, layer1: tuple, layer2: tuple, layer3: tuple, residual_num: int, stride: int,
                 device):
        super().__init__()

        self.residuals = [
            Residual(input_dep if i == 0 else layer3[0], layer1, layer2, layer3, i == 0, stride, device=device)
            for i in range(residual_num)
        ]

    def forward(self, model_input):
        model_output = model_input
        for residual in self.residuals:
            model_output = residual(model_output)

        return model_output


class Resnet50(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.conv2d1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, device=device)
        self.maxP2d1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.stage1 = Stage(64, (64, 1), (64, 3), (256, 1), 3, 1, device=device)
        self.stage2 = Stage(256, (128, 1), (128, 3), (512, 1), 4, 2, device=device)
        self.stage3 = Stage(512, (256, 1), (256, 3), (1024, 1), 6, 2, device=device)
        self.stage4 = Stage(1024, (512, 1), (512, 3), (2048, 1), 3, 2, device=device)

        self.avgP2d = nn.AvgPool2d(7, 7)
        self.liner1 = nn.Linear(2048, 10, device=device)

    def forward(self, model_input):
        x = model_input

        x = self.conv2d1(x)
        x = self.maxP2d1(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgP2d(x)
        x = x.view(-1, 2048)
        x = self.liner1(x)

        return x


class Modeller:
    def __init__(self, device, model_path=None | str):
        if type(model_path) == str:
            self.model = torch.load(str(model_path))
        else:
            self.model = Resnet50(device=device)

        self.model = self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.device = device

        self.resizer = resizer.Modeller()

    def save(self, path) -> None:
        torch.save(self.model, path)

    def train(self, data) -> None:
        self.model.train()

        i = 1

        for batch_input, batch_expect in data:
            batch_input = self.resizer.run(batch_input.to(self.device))
            batch_expect = batch_expect.to(self.device)

            batch_output = self.model(batch_input)
            loss = self.criterion(batch_output, batch_expect)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print("train:\tstep {i}".format(i=i))
            i += 1

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        print("train:epoch done")

    def test(self, data):
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_input, batch_expect in data:
                batch_input = self.resizer.run(batch_input.to(self.device))
                batch_expect = batch_expect.to(self.device)

                batch_output = self.model(batch_input)
                _, answer = torch.max(batch_output.data, 1)

                batch_correct = (answer == batch_expect).sum().item()
                batch_total = len(batch_input)

                correct += batch_correct
                total += batch_total

                print("test:\tbatch_correct {batch_correct}, batch_total {batch_total}".format(batch_correct=batch_correct, batch_total=batch_total))

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        print("test:correct_rate {correct_rate}".format(correct_rate=correct/total))
