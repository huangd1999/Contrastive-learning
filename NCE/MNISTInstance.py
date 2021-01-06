import torchvision.datasets as datasets
from PIL import Image


class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            data, targets = self.data[index], self.targets[index]
        else:
            data, targets = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return data, targets, index