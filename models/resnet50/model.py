import torch
import torch.nn as nn

from torchviz import make_dot


class Residual(nn.Module):
    def __init__(self, input_dep: int, layer1: tuple, layer2: tuple, layer3: tuple, down_sample: bool, stride: int, device):
        super().__init__()
        self.relu = nn.ReLU()

        self.conv2d1 = nn.Conv2d(input_dep, layer1[0], layer1[1], stride=stride if down_sample else 1, padding=(layer1[1] - 1) // 2, device=device)
        self.norm2d1 = nn.BatchNorm2d(layer1[0], device=device)

        self.conv2d2 = nn.Conv2d(layer1[0], layer2[0], layer2[1], padding=(layer2[1] - 1) // 2, device=device)
        self.norm2d2 = nn.BatchNorm2d(layer2[0], device=device)

        self.conv2d3 = nn.Conv2d(layer2[0], layer3[0], layer3[1], padding=(layer3[1] - 1) // 2, device=device)
        self.norm2d3 = nn.BatchNorm2d(layer3[0], device=device)

        self.down_sample = down_sample
        if down_sample:
            self.conv2d0 = nn.Conv2d(input_dep, layer3[0], 1, stride=stride, padding=0, device=device)
            self.norm2d0 = nn.BatchNorm2d(layer3[0], device=device)

    def forward(self, model_input):
        x = self.relu(self.norm2d1(self.conv2d1(model_input)))
        x = self.relu(self.norm2d2(self.conv2d2(x)))
        x = self.norm2d3(self.conv2d3(x))

        if self.down_sample:
            return self.relu(x + self.norm2d0(self.conv2d0(model_input)))

        return self.relu(x + model_input)


class Stage(nn.Module):
    def __init__(self, input_dep: int, layer1: tuple, layer2: tuple, layer3: tuple, residual_num: int, stride: int, device):
        super().__init__()

        self.residuals = [
            Residual(input_dep if i == 0 else layer3[0], layer1, layer2, layer3, i == 0, stride, device=device)
            for i in range(residual_num)
        ]

    def forward(self, model_input):
        x = model_input
        for residual in self.residuals:
            x = residual(x)

        return x


class Resnet50(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.relu = nn.ReLU()

        self.conv2d1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, device=device)
        self.norm2d1 = nn.BatchNorm2d(64, device=device)

        self.maxP2d1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.stage1 = Stage(64, (64, 1), (64, 3), (256, 1), 3, 1, device=device)
        self.stage2 = Stage(256, (128, 1), (128, 3), (512, 1), 4, 2, device=device)
        self.stage3 = Stage(512, (256, 1), (256, 3), (1024, 1), 6, 2, device=device)
        self.stage4 = Stage(1024, (512, 1), (512, 3), (2048, 1), 3, 2, device=device)

        self.avgP2d1 = nn.AvgPool2d(7, 7)
        self.linear1 = nn.Linear(2048, 10, device=device)

    def forward(self, model_input):
        x = self.relu(self.norm2d1(self.conv2d1(model_input)))
        x = self.maxP2d1(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgP2d1(x)
        x = x.view(-1, 2048)
        x = self.linear1(x)

        return x


class ResizeResnet50(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.upSample = nn.UpsamplingNearest2d(224)
        self.resnet50 = Resnet50(device)

    def forward(self, model_input):
        x = self.upSample(model_input)
        x = x.view(-1, 224, 224)
        x = torch.stack((x, x, x), dim=1)
        x = self.resnet50(x)

        return x


class Modeller:
    def __init__(self, device, model_path=None | str):
        if type(model_path) == str:
            self.model = torch.load(str(model_path), device=device)
        else:
            self.model = ResizeResnet50(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.002)
        self.device = device

    def draw(self) -> None:
        x = torch.zeros([16, 1, 28, 28], device=self.device)
        y = self.model(x)
        output = make_dot(y.mean(), params=dict(self.model.named_parameters()))
        output.view()

    def save(self, path) -> None:
        torch.save(self.model, path)

    def train(self, data) -> None:
        self.model.train()

        i = 1

        for batch_input, batch_expect in data:
            batch_input = batch_input.to(self.device)
            batch_expect = batch_expect.to(self.device)

            batch_output = self.model(batch_input)
            loss = self.criterion(batch_output, batch_expect)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print("train:\tstep {i}, loss {loss}".format(i=i, loss=loss))
            i += 1

        print("train:epoch done")

    def test(self, data):
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_input, batch_expect in data:
                batch_input = batch_input.to(self.device)
                batch_expect = batch_expect.to(self.device)

                batch_output = self.model(batch_input)
                _, answer = torch.max(batch_output.data, 1)

                batch_correct = (answer == batch_expect).sum().item()
                batch_total = len(batch_input)

                correct += batch_correct
                total += batch_total

                print("test:\tbatch_correct {batch_correct}, batch_total {batch_total}".format(
                    batch_correct=batch_correct, batch_total=batch_total))

        print("test:correct_rate {correct_rate}".format(correct_rate=correct / total))
