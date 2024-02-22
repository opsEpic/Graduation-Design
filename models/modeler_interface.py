import torch


class Modeler_interface:
    def __init__(self, speaker, device):
        self.preprocess = None
        self.model = None
        self.criterion = None
        self.optimizer = None

    def __load__(self, preprocess, model, criterion, optimizer, device):
        self.preprocess = preprocess.to(device)
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer

    def __train__(self, audio, label) -> float:
        self.model.train()

        batch_input = self.preprocess(audio)

        batch_output = self.model(batch_input)
        loss = self.criterion(batch_output, label)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def __infer__(self, audio1, audio2) -> int:
        self.model.eval()

        with torch.no_grad():
            vec1 = self.model(self.preprocess(audio1.view(1, -1)))
            vec2 = self.model(self.preprocess(audio2.view(1, -1)))

            score = torch.sum(vec1 * vec2).tolist()
            return 1 if score >= 0.5 else 0

    def __test__(self, audio1, audio2, label) -> (int, int, int, int):
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0

        if label == self.__infer__(audio1, audio2):
            if label:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if label:
                false_positive += 1
            else:
                false_negative += 1

        return true_positive, true_negative, false_positive, false_negative

    def model_train(self, train_dataset, test_dataset, epochs, repeats):
        eers = [[0. for _ in range(epochs)] for _ in range(repeats)]

        for i in range(repeats):
            for j in range(epochs):
                for audio, label in train_dataset:
                    self.__train__(audio, label)

                true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
                for audio1, audio2, label in test_dataset:
                    tp, tn, fp, fn = self.__test__(audio1, audio2, label)
                    true_positive += tp
                    true_negative += tn
                    false_positive += fp
                    false_negative += fn

                tpr = true_positive / (true_positive + false_negative)
                fpr = false_positive / (false_positive + true_negative)

                eers[i][j] = (fpr + 1. - tpr) / 2

        return eers
