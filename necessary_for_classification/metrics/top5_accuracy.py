import torch


class Top5Accuracy:
    def __init__(self):
        self.name = 'Top5Accuracy'

    def __call__(self, output, target):
        result = 0
        for o, t in zip(output, target):
            result += int(t in torch.argsort(o, descending=True)[:5])
        return result


OBJECT = Top5Accuracy
