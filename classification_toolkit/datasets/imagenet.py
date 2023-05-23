import torch
from torchvision import datasets, transforms


def load_imagenet(path, train_batch_size, test_batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageNet(path, split='train', transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    test_dataset = datasets.ImageNet(path, split='val', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return {'train': train_loader, 'test': test_loader}


OBJECT = load_imagenet
