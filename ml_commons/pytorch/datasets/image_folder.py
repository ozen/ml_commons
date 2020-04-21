import os

from torchvision.datasets import ImageFolder


class ImageFolderWithPath(ImageFolder):
    """
    Extends ImageFolder dataset to return 3-tuples instead of 2-tuples
    by adding relative paths of the images as the third value.
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        relative_path = path.replace(self.root + os.sep, '')
        return sample, target, relative_path
