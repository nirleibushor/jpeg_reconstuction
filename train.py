from torch.utils.tensorboard import SummaryWriter
from time import time
import os
import torch
from IPython import embed
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from data import load_images_paths, ImageToJpegDataset
from network import DnCNN


def rmse(targets, preds): # todo replace with torch's loss
    b = targets.shape[0]
    loss_per_sample = torch.sqrt(torch.pow(targets - preds, 2).view(b, -1).mean(dim=1))
    return loss_per_sample.mean()


def save_model(model, epoch, step, output_path):
    checkpoint_name = 'model_epoch_{:05d}.pkl'.format(epoch)
    torch.save({'state_dict': model.state_dict(),
                'epoch': epoch,
                'step': step},
               os.path.join(output_path, checkpoint_name))


def parse_epoch(checkpoint_path):
    return int(os.path.splitext(checkpoint_path[checkpoint_path.index('model_epoch_') + len('model_epoch_'):])[0])


def load_model(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['state_dict'])
    return ckpt['epoch'], ckpt['step']


if __name__ == '__main__':
    size = 64
    images_path = f'/home/nirl/Downloads/takehome/images_{size}x{size}/'
    val = '/home/nirl/Downloads/takehome/val_images.txt'
    train = '/home/nirl/Downloads/takehome/train_images.txt'
    checkpoints_path = f'/home/nirl/Downloads/takehome/checkpoints_{size}x{size}/'
    batch_size = 64
    num_workers = 2
    num_epochs = 1000
    print_step = 1
    save_step = 10
    learning_rate = 1e-3
    lr_milestones = [20, 60, 120]
    lr_gamma = 0.1

    # resume_from = checkpoints_path + 'model_epoch_00010.pkl'
    resume_from = None

    ds_val = ImageToJpegDataset(val, images_path)
    ds_train = ImageToJpegDataset(train, images_path)

    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers)
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers)

    print(f'loaded {len(ds_train)} train images and {len(ds_val)} validation images.')

    model = DnCNN()
    last_epoch, step = -1, 0
    if resume_from:
        last_epoch, step = load_model(model, resume_from)
        print(f'loaded checkpoint: {resume_from}')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma, last_epoch=last_epoch)

    writer = SummaryWriter(checkpoints_path)

    model.train()
    for epoch in range(last_epoch + 1, num_epochs):
        t = time()
        for batch in dl_train:
            optimizer.zero_grad()
            bmps, jpgs, y, indices = batch
            bmps_pred = model(jpgs)
            loss = rmse(bmps, bmps_pred)
            loss.backward()  # todo create a run_a_batch function
            optimizer.step()
            if step % print_step == 0:
                print('epoch {}, step {}) loss: {:.8f} learning rate: {:.8f}'.format(epoch, step, loss, optimizer.param_groups[0]['lr']))
            writer.add_scalar('train_loss', loss, step)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step)
            step += 1
        epoch_time = time() - t
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss_jpgs = 0.
            val_loss = 0.
            for batch in dl_val:
                bmps, jpgs, y, indices = batch
                bmps_pred = model(jpgs)
                loss = rmse(bmps, bmps_pred)
                loss_jpgs = rmse(bmps, jpgs)
                w = jpgs.shape[0] / float(len(ds_val))
                val_loss += loss * w
                val_loss_jpgs += loss_jpgs * w
        model.train()
        writer.add_scalar('validation_loss', val_loss, step)
        writer.add_scalar('validation_loss_jpgs', val_loss_jpgs, step)

        stats_line = f'### EPOCH {epoch}) validation loss: {val_loss}  validation loss jpgs: {val_loss_jpgs} ({epoch_time}secs) ###\n'
        if val_loss_jpgs > val_loss:
            stats_line += '! SOLVED !\n'
        print(stats_line)

        if epoch % save_step == 0:
            save_model(model, epoch, step, checkpoints_path)

        epoch += 1
