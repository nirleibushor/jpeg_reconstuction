from IPython import embed
import numpy as np
import torch
from network import DnCNN
from scipy.misc import imread
import matplotlib.pyplot as plt
from train import rmse


if __name__ == '__main__':
    size = 64
    ckpt_epoch = 999
    model_path = '/home/nirl/Downloads/takehome/checkpoints_{}x{}_bkp/model_epoch_{:04d}.pkl'.format(size, size, ckpt_epoch)
    size = 128
    image_base_path = f'/home/nirl/Downloads/takehome/images_{size}x{size}/series_007_slice_000'

    model = DnCNN()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    bmp = imread(image_base_path + '.bmp')
    jpg = imread(image_base_path + '.jpg')

    x = torch.from_numpy(jpg.astype('float32') / 255).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        bmp_pred = model(x).numpy()[0][0]
    bmp_tmp = torch.from_numpy(bmp.astype('float32') / 255).unsqueeze(0).unsqueeze(0)

    rmse_jpg = rmse(bmp_tmp, x)
    rmse_pred = rmse(bmp_tmp, bmp_pred)
    print('jpg', rmse_jpg)
    print('pred', rmse_pred)
    print('GOOD!' if rmse_jpg > rmse_pred else 'BAD!')
    print()

    bmp_pred = (np.clip(bmp_pred, 0, 1) * 255).astype(np.uint8)

    row1 = np.hstack((bmp, jpg))
    row2 = np.hstack((bmp, bmp_pred))
    img = np.vstack((row1, row2))

    plt.imshow(img, cmap='gray')
    plt.title('up: bmp,before\ndown: bmp,after')
    plt.show()
