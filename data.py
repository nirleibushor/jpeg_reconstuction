import os
import numpy as np
from scipy.misc import imread
from torch.utils.data import Dataset


class ImageToJpegDataset(Dataset):
    def __init__(self, images_names, images_root):
        self.images_names = load_images_paths(images_names)
        self.images_names = [os.path.join(images_root, x) for x in self.images_names]

    def __getitem__(self, index):
        base_name = self.images_names[index]
        bmp = imread(base_name + '.bmp').astype('float32') / 255
        jpg = imread(base_name + '.jpg').astype('float32') / 255
        d = bmp - jpg  # bmp = jpg + d
        return np.expand_dims(bmp, 0), np.expand_dims(jpg, 0), np.expand_dims(d, 0), index

    def __len__(self):
        return len(self.images_names)


def load_images_paths(images_files_txt):
    with open(images_files_txt) as f:
        images = [x.strip() for x in f]
    return images


def get_series_images_names(series_index):
    return ['series_{:03d}_slice_{:03d}'.format(series_index, slice_index) for slice_index in range(100)]


def split_to_subsets(root_path):
    # similar series: [1,7], [2,3,4] must be in the same split
    # 5, 6 are quite different than the rest => put in train?
    test = [0, 2, 3, 4, 5, 6]
    train = [1, 7, 8]
    val = [9]
    assert list(range(10)) == sorted(test+val+train)
    images = [x for series_idx in test for x in get_series_images_names(series_idx)]
    with open(os.path.join(root_path, 'test_images.txt'), 'w') as f:
        f.writelines(x + '\n' for x in images)
    images = [x for series_idx in val for x in get_series_images_names(series_idx)]
    with open(os.path.join(root_path, 'val_images.txt'), 'w') as f:
        f.writelines(x + '\n' for x in images)
    images = [x for series_idx in train for x in get_series_images_names(series_idx)]
    with open(os.path.join(root_path, 'train_images.txt'), 'w') as f:
        f.writelines(x + '\n' for x in images)


if __name__ == '__main__':
    split_to_subsets('/home/nirl/Downloads/takehome/')
