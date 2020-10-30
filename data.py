import os
import numpy as np
from scipy.misc import imread
from torch.utils.data import Dataset


def get_images_series(images_names):
    """
    :param images_names: a list of images names / paths
    :return: a list with the series numbers of given images
    """
    images_names = [os.path.basename(x) for x in images_names]
    return sorted(set(int(x[len('series_'): len('series_') + 3]) for x in images_names))


def get_series_images_names(series_index):
    """
    :param series_index: int
    :return: a list with images names for the given series
    """
    return ['series_{:03d}_slice_{:03d}'.format(series_index, slice_index) for slice_index in range(100)]


def split_to_subsets(output_path):
    """
    split images for series 0 to 9 to train, val and test, and saves txt files in the given output path
    :param output_path: str
    :return:
    """
    # similar series: [1,7], [2,3,4] must be in the same split
    # 5, 6 are quite different than the rest => put in train?
    test = [0, 1, 2, 3, 4, 7]
    train = [5, 6, 8]
    val = [9]
    assert list(range(10)) == sorted(test+val+train)
    images = [x for series_idx in test for x in get_series_images_names(series_idx)]
    with open(os.path.join(output_path, 'test_images.txt'), 'w') as f:
        f.writelines(x + '\n' for x in images)
    images = [x for series_idx in val for x in get_series_images_names(series_idx)]
    with open(os.path.join(output_path, 'val_images.txt'), 'w') as f:
        f.writelines(x + '\n' for x in images)
    images = [x for series_idx in train for x in get_series_images_names(series_idx)]
    with open(os.path.join(output_path, 'train_images.txt'), 'w') as f:
        f.writelines(x + '\n' for x in images)


def load_images_paths(images_files_txt):
    with open(images_files_txt) as f:
        images = [x.strip() for x in f]
    return images


class ImageToJpegDataset(Dataset):
    """
    a torch data set that returns samples which of the following structure:
    bmp image, jpg image, between the two diff
    jpg image is the model input
    bmp image is the target for model's output
    (diff can also be used as target in a different setting)
    """
    def __init__(self, images_names, images_root):
        """
        :param images_names: path to txt file with paths to images
        :param images_root: path where images are saved
        """
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
