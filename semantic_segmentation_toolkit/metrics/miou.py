import torch
import torch.nn.functional as F


class MIOU:
    @staticmethod
    def IoU(x, y, smooth=1):
        intersection = (x * y).abs().sum(dim=[1, 2])
        union = torch.sum(y.abs() + x.abs(), dim=[1, 2]) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou

    @staticmethod
    def get_mask(target, num_classes=19):
        mask = (target >= 0) & (target < num_classes)
        return mask.float()

    def __init__(self):
        self.name = 'miou'

    def __call__(self, output, target):
        if output.shape[-1] != target.shape[-1] or output.shape[-2] != target.shape[-2]:
            output = F.interpolate(
                output,
                size=(target.shape[-2],target.shape[-1]),
                mode='nearest')
        l = list()
        mask = self.get_mask(target)
        transformed_output = output.permute(0, 2, 3, 1).argmax(dim=3)
        for c in range(output.shape[1]):
            x = (transformed_output == c).float() * mask
            y = (target == c).float()
            l.append(self.IoU(x, y))
        return torch.sum(torch.mean(torch.stack(l).permute(1, 0), dim=1)).item()


OBJECT = MIOU
