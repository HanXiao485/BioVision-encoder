import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

class CIFAR10Manager:
    def __init__(self, root='./data', batch_size=64, num_workers=2):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_loader(self, train=True):
        dataset = torchvision.datasets.CIFAR10(
            root=self.root, 
            train=train, 
            download=True, 
            transform=self.transform
        )
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=train, 
            num_workers=self.num_workers
        )

class CIFAR100Manager:
    def __init__(self, root='/home/xiaohan/BioVision-encoder/data', batch_size=64, num_workers=2):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_loader(self, train=True):
        dataset = torchvision.datasets.CIFAR100(
            root=self.root, 
            train=train, 
            download=True, 
            transform=self.transform
        )
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=train, 
            num_workers=self.num_workers
        )