import torchvision.datasets as datasets
from PIL import Image


class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            self.img, self.target = self.data[index], self.targets[index]
        else:
            self.img, self.target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        self.img = Image.fromarray(self.img)

        if self.transform is not None:
            self.img = self.transform(self.img)

        if self.target_transform is not None:
            target = self.target_transform(self.target)

        return self.img, self.target, index
