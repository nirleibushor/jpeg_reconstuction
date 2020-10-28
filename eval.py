from IPython import embed
import numpy as np
import torch
from network import DnCNN
from scipy.misc import imread
import matplotlib.pyplot as plt
from train import rmse, load_model


if __name__ == '__main__':
    size = 64
    ckpt_epoch = 44
    model_path = '/home/nirl/Downloads/takehome/checkpoints_{}x{}_resid/model_epoch_{:05d}.pkl'.format(size, size, ckpt_epoch)
    # size = 128
    image_base_path = f'/home/nirl/Downloads/takehome/images_{size}x{size}/series_007_slice_000'  # train
    # image_base_path = f'/home/nirl/Downloads/takehome/images_{size}x{size}/series_006_slice_000'  # val
    # image_base_path = f'/home/nirl/Downloads/takehome/images_{size}x{size}/series_000_slice_000'  # test

    model = DnCNN()
    epoch, step = load_model(model, model_path)
    model.eval()

    bmp = imread(image_base_path + '.bmp')
    jpg = imread(image_base_path + '.jpg')

    with torch.no_grad():
        x = torch.from_numpy(jpg.astype('float32') / 255).unsqueeze(0).unsqueeze(0)
        pred_residual = model(x)
        pred_residual = np.round(pred_residual.numpy()[0][0] * 255).astype(np.int64)
        pred_bmp = np.clip(jpg.astype(np.int64) + pred_residual, 0, 255).astype(np.uint8)

        y = torch.from_numpy(bmp.astype('float32') / 255).unsqueeze(0).unsqueeze(0)
        y_pred = torch.from_numpy(pred_bmp.astype('float32') / 255).unsqueeze(0).unsqueeze(0)
        loss_jpg = rmse(y, x)
        loss_pred = rmse(y, y_pred)

        print('jpg', loss_jpg)
        print('pred', loss_pred)
        print('GOOD!' if loss_jpg > loss_pred else 'BAD!')

        row1 = np.hstack((bmp[:30,:30], jpg[:30,:30]))
        row2 = np.hstack((bmp[:30,:30], pred_bmp[:30,:30]))
        # row1 = np.hstack((bmp, jpg))
        # row2 = np.hstack((bmp, pred_bmp))
        img = np.vstack((row1, row2))

        plt.imshow(img, cmap='gray')
        plt.title('up: bmp,before\ndown: bmp,after')
        plt.show()
