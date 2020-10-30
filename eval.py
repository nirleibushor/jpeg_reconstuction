import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from data import ImageToJpegDataset, get_images_series
from network import DnCNN
from train import load_model


def rmse(x, y):
    """
    RMSE for numpy images
    :param x:
    :param y:
    :return:
    """
    n = x.shape[0]
    x = x.reshape(n, -1)
    y = y.reshape(n, -1)
    return np.sqrt(np.square(x - y).mean(axis=1)).mean()


def np_uint8(x, clip=False):
    """
    :param x: torch float tensor with values scale 0,1
    :param clip: if true, clips output values not in 0,255
    :return: numpy uint8 array in scale 0,255
    """
    x = x.cpu().numpy() * 255
    if clip:
        x = np.clip(x, 0, 255)
    return x.astype(np.uint8)


def display(bmp, jpg, bmp_pred, image_name):
    #    TODO maybe change to new model in colab, submit
    """
    display an example of jpeg reconstruction
    :param bmp: np array target bmp
    :param jpg: np array source jpg
    :param bmp_pred: np array bmp prediction
    :param image_name: name of the presented bmp/jpg
    :return:
    """
    row1 = np.hstack((bmp[50:150, 50:150], jpg[50:150, 50:150]))
    row2 = np.hstack((bmp[50:150, 50:150], bmp_pred[50:150, 50:150]))
    img = np.vstack((row1, row2))
    plt.imshow(img, cmap='gray')
    plt.title(f'{image_name}\nup: bmp,before\ndown: bmp,after')
    plt.show()


def load_test_data(images_root, test_images_path, batch_size, num_workers, limit=None):
    """
    load the test data set
    :param images_root: path to images
    :param test_images_path: path to txt file with the test images paths
    :param batch_size: int, batch size for testing
    :param num_workers: int, num processed torch uses to craete batches
    :param limit: int, limit the number of images to test
    :return:
    """
    ds = ImageToJpegDataset(test_images_path, images_root)
    if limit is not None:
        ds.images_names = ds.images_names[::len(ds.images_names) // limit]
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)
    print(f'testing with series: {get_images_series(ds.images_names)} ({len(ds.images_names)} images)')
    return dl


def run_on_test_set(model, dataloader):
    """
    run the given model on the loaded data
    :param model: DnCNN
    :param dataloader: torch.utils.data.Dataloader
    :return: test_loss: rmse between jpgs and bmp predcited by the model, test_loss_jpgs rmse between the bmps and original jpegs
    """
    test_loss_jpgs = 0.
    test_loss = 0.
    for batch in dataloader:
        bmps, jpgs, y, indices = batch
        if torch.cuda.is_available():
            bmps, jpgs = bmps.cuda(), jpgs.cuda()
        bmps_pred = model(jpgs)

        bmps = np_uint8(bmps)
        jpgs = np_uint8(jpgs)
        bmps_pred = np_uint8(bmps_pred, clip=True)

        loss = rmse(bmps, bmps_pred)
        loss_jpgs = rmse(bmps, jpgs)

        w = jpgs.shape[0] / float(len(dataloader.dataset))
        test_loss += loss * w
        test_loss_jpgs += loss_jpgs * w
    return test_loss, test_loss_jpgs


def evaluate(images_root,
             test_images,
             checkpoint_path,
             batch_size,
             num_workers,
             do_display,
             limit):
    """
    evaluate the model on test set
    :param images_root: path to images
    :param test_images: path to txt file with test images paths
    :param checkpoint_path: path to model checkpoint
    :param batch_size: int, batch size for testing
    :param num_workers: int, num processed torch uses to create batches
    :param do_display: bool, if true, displays an example
    :param limit: int, limit the number of test images
    :return:
    """
    model = DnCNN()
    epoch, step = load_model(model, checkpoint_path)
    print(f'loaded model: {checkpoint_path} (epoch: {epoch} step: {step})')
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    dl = load_test_data(images_root, test_images, batch_size, num_workers, limit)

    with torch.no_grad():
        test_loss, test_loss_jpgs = run_on_test_set(model, dl)

    stats_line = f'### epoch {epoch}) test loss: {test_loss} (jpgs loss: {test_loss_jpgs}) ###\n'
    print(stats_line)

    if do_display:
        with torch.no_grad():
            for batch in dl:
                bmps, jpgs, y, indices = batch
                if torch.cuda.is_available():
                    bmps, jpgs = bmps.cuda(), jpgs.cuda()
                bmps_pred = model(jpgs)
                bmps = np_uint8(bmps)
                jpgs = np_uint8(jpgs)
                bmps_pred = np_uint8(bmps_pred, clip=True)
                print('displaying a random example.')
                i = np.random.randint(bmps.shape[0])
                display(bmps[i, 0, :, :], jpgs[i, 0, :, :], bmps_pred[i, 0, :, :], indices[i])
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_root', '-ir', default='images', help='path to images')
    parser.add_argument('--test_images', '-ti', default='data_splits/test_images.txt', help='path to a txt file with test images')
    parser.add_argument('--checkpoint_path', '-cp', default='model.pkl', help='path to model checkpoint')
    parser.add_argument('--batch_size', '-bs', default=2, type=int, help='batch size for testing for test mode')
    parser.add_argument('--num_workers', '-nw', default=1, type=int, help='number of processes torch uses to create batches for test mode')
    parser.add_argument('--do_display', '-d', default=False, action='store_true', help='display examples from test set for test mode')
    parser.add_argument('--limit', '-l', default=None, type=int, help='run on a limited number of images')
    args = parser.parse_args()

    evaluate(**vars(args))
