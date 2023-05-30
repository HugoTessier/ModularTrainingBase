class Top1Accuracy:
    def __init__(self):
        self.name = 'Top1Accuracy'

    def __call__(self, output, target):
        pred = output.argmax(dim=1, keepdim=True)
        return pred.eq(target.view_as(pred)).sum().item()


OBJECT = Top1Accuracy
