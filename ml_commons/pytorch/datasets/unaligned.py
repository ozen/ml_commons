import random
import h5py
from torch.utils.data import Dataset


class UnalignedDataset(Dataset):
    def __init__(self, path, split='train', transform=None):
        super().__init__()
        self.path = path
        self.split = split
        self.transform = transform
        self.file = None

        with h5py.File(self.path, 'r') as file:
            self.len_A = len(file[f'{split}_A'])
            self.len_B = len(file[f'{split}_A'])

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')

        index_A = index % self.len_A  # make sure index is within then range
        img_A = self.file[f'{self.split}_A'][index_A]
        index_B = random.randint(0, self.len_B - 1)  # randomize the index for B to avoid fixed pairs
        img_B = self.file[f'{self.split}_B'][index_B]

        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return img_A, img_B

    def __len__(self):
        return max(self.len_A, self.len_B)
