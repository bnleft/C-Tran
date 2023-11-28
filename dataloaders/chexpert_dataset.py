import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataloaders.data_utils import image_loader, get_unk_mask_indices


class ChexpertDataset(Dataset):
    def __init__(self, split, num_labels, data_file, data_root, transform, testing):
        self.split = split
        self.split_data = pd.read_csv(data_file)

        self.data_root = data_root
        self.transform = transform
        self.num_labels = num_labels
        self.known_labels = 0
        self.testing = testing
        self.epoch = 1

    def __len__(self):
        return len(self.split_data)

    def __getitem__(self, index):
        item = self.split_data.iloc[index]

        image_path = os.path.join(self.data_root, item.loc['Path'])
        image = image_loader(image_path, self.transform)

        first_label_index = 5
        labels = item.iloc[first_label_index:first_label_index + self.num_labels - 1]
        labels = torch.Tensor(labels)

        unk_mask_indices = get_unk_mask_indices(image,self.testing, self.num_labels,self.known_labels)

        mask = labels.clone()
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        sample = {
            'image': image,
            'labels': labels,
            'mask': mask,
            'imageIDs': image_path
        }
        return sample
