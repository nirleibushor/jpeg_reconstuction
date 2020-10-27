from IPython import embed
import os
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader


class ImageToJpegDataset(Dataset):
    def __init__(self, images_names, images_root):
        self.images_names = load_images_paths(images_names)#[:10]
        self.images_names = [os.path.join(images_root, x) for x in self.images_names]

    def __getitem__(self, index):
        base_name = self.images_names[index]
        bmp = imread(base_name + '.bmp').astype('float32') / 255
        jpg = imread(base_name + '.jpg').astype('float32') / 255
        d = bmp - jpg  # bmp = jpg + d
        return np.expand_dims(bmp, 0), np.expand_dims(jpg, 0), np.expand_dims(d, 0), index

    def __len__(self):
        return len(self.images_names)

    # def show(self, index):
    #     data = self[index]
    #     img, jpg, d = data['image'], data['jpeg'], data['diff']
    #     img = (img * 255).astype(np.uint8)
    #     jpg = (jpg * 255).astype(np.uint8)
    #     display = np.vstack((img, jpg, d))
    #     imshow(display)

#
# def collate_fn(samples):
#     img, jpg, d = samples
#     img = img.unsqueeze(1)
#     jpg = jpg.unsqueeze(1)
#     d = d.unsqueeze(1)
#     return img, jpg, d


def load_images_paths(images_files_txt):
    with open(images_files_txt) as f:
        images = [x.strip() for x in f]
    return images


def get_series_images_names(series_index):
    return ['series_{:03d}_slice_{:03d}'.format(series_index, slice_index) for slice_index in range(100)]


def split_to_subsets(root_path, n_test_series=6, n_val_series=1, n_train_series=3):
    images = [x for series_idx in range(n_test_series) for x in get_series_images_names(series_idx)]
    with open(os.path.join(root_path, 'test_images.txt'), 'w') as f:
        f.writelines(x + '\n' for x in images)
    images = [x for series_idx in range(n_test_series, n_test_series + n_val_series) for x in get_series_images_names(series_idx)]
    with open(os.path.join(root_path, 'val_images.txt'), 'w') as f:
        f.writelines(x + '\n' for x in images)
    images = [x for series_idx in range(n_test_series + n_val_series, n_test_series + n_val_series + n_train_series) for x in get_series_images_names(series_idx)]
    with open(os.path.join(root_path, 'train_images.txt'), 'w') as f:
        f.writelines(x + '\n' for x in images)


if __name__ == '__main__':
    split_to_subsets('/home/nirl/Downloads/takehome/')
