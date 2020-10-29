from torch.utils.tensorboard import SummaryWriter
from IPython import embed
from train import rmse
import torch
import os
from network import DnCNN
from train import load_model
from data import ImageToJpegDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    checkpoints_path = '/home/nirl/Downloads/takehome/checkpoints_64x64/'
    size = 128
    num_workers = 1
    batch_size = 32
    start_epoch = 800

    checkpoints = sorted([x for x in os.listdir(checkpoints_path) if x.startswith('model_')])
    checkpoints = checkpoints[start_epoch:]

    val = '/home/nirl/Downloads/takehome/val_images.txt'
    images_path = f'/home/nirl/Downloads/takehome/images_{size}x{size}/'
    ds = ImageToJpegDataset(val, images_path)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)

    model = DnCNN()

    # writer = SummaryWriter(checkpoints_path)

    for ckpt in checkpoints:
        epoch, step = load_model(model, os.path.join(checkpoints_path, ckpt))

        model.eval()
        with torch.no_grad():
            val_loss_jpgs = 0.
            val_loss = 0.
            for batch in dl:
                bmps, jpgs, y, indices = batch
                bmps_pred = model(jpgs)
                loss = rmse(bmps, bmps_pred)
                loss_jpgs = rmse(bmps, jpgs)
                w = jpgs.shape[0] / float(len(ds))
                val_loss += loss * w
                val_loss_jpgs += loss_jpgs * w
        writer.add_scalar('validation_loss', val_loss, step)
        writer.add_scalar('validation_loss_jpgs', val_loss_jpgs, step)

        stats_line = f'### EPOCH {epoch}) validation loss: {val_loss} (loss jpgs: {val_loss_jpgs} ###\n'
        print(stats_line)
