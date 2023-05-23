import torch
import torch.nn.functional as F


class SegSemCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.name = 'segsem_crossentropy'

    def forward(self, output, label):
        if output.shape[-1] != label.shape[-1] or output.shape[-2] != label.shape[-2]:
            output = F.interpolate(output, size=label.shape[-2:])
        return self.criterion(output, label)


OBJECT = SegSemCrossEntropyLoss
