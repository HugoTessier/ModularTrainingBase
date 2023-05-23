import torch
import math


def poly_schd(optimizer, epochs, poly_exp):
    def poly_schd_function(e):
        return math.pow(1 - e / epochs, poly_exp)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_schd_function)


OBJECT = poly_schd
