from torch.utils.data import DataLoader
from data import ImageToJpegDataset
from IPython import embed
import numpy as np
import torch
from network import DnCNN
import matplotlib.pyplot as plt
from train import load_model


if __name__ == '__main__':
    size = 64
    ckpt_epoch = 999
    batch_size = 32
    num_workers = 1

    model_path = '/home/nirl/Downloads/takehome/checkpoints_{}x{}/model_epoch_{:05d}.pkl'.format(size, size, ckpt_epoch)

    model = DnCNN()
    epoch, step = load_model(model, model_path)
    model.eval()

    size = 128
    ds = ImageToJpegDataset('/home/nirl/Downloads/takehome/val_images.txt',
                            f'/home/nirl/Downloads/takehome/images_{size}x{size}/')
    # ds = ImageToJpegDataset('/home/nirl/Downloads/takehome/val_images.txt',
    #                         f'/home/nirl/Downloads/takehome/')
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, drop_last=False)

    with torch.no_grad():
        test_loss_jpgs = 0.
        test_loss = 0.
        for batch in dl:
            bmps, jpgs, y, indices = batch
            bmps_pred = model(jpgs)

            bmps = (bmps.numpy() * 255).astype(np.uint8)
            jpgs = (jpgs.numpy() * 255).astype(np.uint8)
            bmps_pred = np.clip(bmps_pred.numpy() * 255, 0, 255).astype(np.uint8)

            b = bmps.shape[0]
            bmps = bmps.reshape(b, -1)
            jpgs = jpgs.reshape(b, -1)
            bmps_pred = bmps_pred.reshape(b, -1)
            loss_jpgs = np.sqrt(np.square(bmps - jpgs).mean(axis=1)).mean()
            loss = np.sqrt(np.square(bmps - bmps_pred).mean(axis=1)).mean()

            # loss = rmse(bmps, bmps_pred)
            # loss_jpgs = rmse(bmps, jpgs)

            w = jpgs.shape[0] / float(len(ds))
            test_loss += loss * w
            test_loss_jpgs += loss_jpgs * w

        stats_line = f'### EPOCH {epoch}) test loss: {loss} (loss jpgs: {loss_jpgs} ###\n'
        print(stats_line)

        for batch in dl:
            bmps, jpgs, y, indices = batch
            bmps_pred = model(jpgs)

        bmps = (bmps[:5].squeeze(1).numpy() * 255).astype(np.uint8)
        jpgs = (jpgs[:5].squeeze(1).numpy() * 255).astype(np.uint8)
        bmps_pred = np.clip(bmps_pred[:5].squeeze(1).numpy() * 255, 0, 255).astype(np.uint8)

        for i, bmp, jpg, pred in zip(indices, bmps, jpgs, bmps_pred):
            row1 = np.hstack((bmp, jpg))
            row2 = np.hstack((bmp, pred))
            # row1 = np.hstack((bmp[:20,:20], jpg[:20,:20]))
            # row2 = np.hstack((bmp[:20,:20], pred[:20,:20]))
            img = np.vstack((row1, row2))

            plt.imshow(img, cmap='gray')
            plt.title(f'{ds.images_names[i]}\nup: bmp,before\ndown: bmp,after')
            plt.show()

