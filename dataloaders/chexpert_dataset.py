import pandas as pd
from torch.utils.data import Dataset


class ChexpertDataset(Dataset):
    def __init__(self, split, num_labels, data_file, img_root, transform, testing):
        self.split = split
        self.split_data = pd.read_csv(data_file)

        self.img_root = img_root
        self.transform = transform
        self.num_labels = num_labels
        self.known_labels = 0
        self.testing = testing
        self.epoch = 1

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, index):
        item = self.split_data[index]

        print(item)

        # TODO: implement this
        sample = {
            'image': None,
            'labels': None,
            'mask': None,
            'imageIDs': None
        }
        return sample
