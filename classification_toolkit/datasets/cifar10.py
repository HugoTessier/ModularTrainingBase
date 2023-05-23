import torch
from torchvision import datasets, transforms


def load_cifar10(path, train_batch_size, test_batch_size):
    list_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

    transform_train = transforms.Compose(list_trans)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(path, train=True, download=True, transform=transform_train),
        batch_size=train_batch_size, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(path, train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=True, num_workers=4)

    return {'train': train_loader, 'test': test_loader}


OBJECT = load_cifar10
