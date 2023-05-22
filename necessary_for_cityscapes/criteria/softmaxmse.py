import torch


class SoftMaxMSELoss:
    def __init__(self):
        self.softmax = torch.nn.Softmax(dim=1)
        self.mse = torch.nn.MSELoss()
        self.name = 'SoftMaxMSELoss'

    def __call__(self, pred, target, num_classes=19):
        pred = self.softmax(pred)
        new_target = torch.stack([target == c for c in range(num_classes)], dim=1).to(target.device)
        return self.mse(pred, new_target.float())


OBJECT = SoftMaxMSELoss
