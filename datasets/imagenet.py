import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


class ImageNetManager:
    def __init__(
        self,
        root='/home/xiaohan/BioVision-encoder/data/imagenet',
        batch_size=64,
        num_workers=8,
        image_size=224
    ):
        """
        Args:
            root (str): ImageNet root directory
            batch_size (int): batch size
            num_workers (int): DataLoader worker
            image_size (int): image size default 224
        """
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # ImageNet standard mean / std
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def get_loader(self, train=True):
        """
        Args:
            train (bool): True -> train set, False -> validation set
        """
        split = 'train' if train else 'val'
        transform = self.train_transform if train else self.val_transform

        dataset = torchvision.datasets.ImageNet(
            root=self.root,
            split=split,
            transform=transform
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=True
        )

    @property
    def num_classes(self):
        return 1000
