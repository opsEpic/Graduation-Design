import torch
import torch.nn as nn


class _MyResnet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(in_features=3*224*224, out_features=10)

    def forward(self, model_input):
        model_input = model_input.view(-1, 3*224*224)

        x1 = self.linear1(model_input)
        return x1


class Modeller:
    def __init__(self, input_model=None | str, device='cpu'):
        self.model = input_model
        if type(self.model) == str:
            self.model = str(self.model)
            self.model = torch.load(self.model)
        else:
            self.model = _MyResnet50()

        self.model = self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.device = device

    def save(self, path) -> None:
        torch.save(self.model, path)

    def train(self, batch_input: torch.Tensor, batch_expect: torch.Tensor) -> None:
        self.model.train()

        batch_input = batch_input.to(self.device)
        batch_expect = batch_expect.to(self.device)

        batch_output = self.model(batch_input)
        loss = self.criterion(batch_output, batch_expect)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print("train:")

    def test(self, batch_input: torch.Tensor, batch_expect: torch.Tensor) -> int:
        self.model.eval()

        batch_input = batch_input.to(self.device)
        batch_expect = batch_expect.to(self.device)

        correct = 0

        with torch.no_grad():
            batch_output = self.model(batch_input)
            _, answer = torch.max(batch_output.data, 1)

            correct += (answer == batch_expect).sum().item()

        print("test:\tcorrect={correct}".format(correct=correct))
        return correct
