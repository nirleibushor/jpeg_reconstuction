import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import ImageToJpegDataset, get_images_series
from network import DnCNN


def save_model(model, epoch, step, output_path):
    """
    save models checkpoint
    :param model: DnCNN
    :param epoch: int
    :param step: int
    :param output_path: str, path to save the checkpoint
    :return:
    """
    checkpoint_name = 'model_epoch_{:05d}.pkl'.format(epoch)
    torch.save({'state_dict': model.state_dict(),
                'epoch': epoch,
                'step': step},
               os.path.join(output_path, checkpoint_name))


def load_model(model, checkpoint_path):
    """
    load the torch model checkpoint and returns the epoch and step on which it was saved
    :param model: DnCNN
    :param checkpoint_path: str, path to checkpoint saved with save_model
    :return: epoch, step
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    return ckpt['epoch'], ckpt['step']


def load_train_validation_data(images_root, train_images_path, val_images_path, batch_size, num_workers):
    """
    load training and validation dataloaders
    :param images_root: path to where .bmp and .jpg images are saved
    :param train_images_path: path to txt file with all images used for training
    :param val_images_path: path to txt file with all images used for validation
    :param batch_size: int
    :param num_workers: int, number of processed torch uses to create batches
    :return: train data loader, validation data loader
    """
    ds_train = ImageToJpegDataset(train_images_path, images_root)
    ds_val = ImageToJpegDataset(val_images_path, images_root)
    print(f'training with series: {get_images_series(ds_train.images_names)} ({len(ds_train.images_names)} images)')
    print(f'validating with series: {get_images_series(ds_val.images_names)} ({len(ds_val.images_names)} images)')
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)
    return dl_train, dl_val


def reduce_learning_rate(optimizer, gamma):
    """
    sets learning rate to learning rate * gamma
    :param optimizer: torch optimizer initialized with model parameters
    :param gamma: float
    :return:
    """
    assert 0 < gamma < 1, 'gamma value must be between 0 and 1'
    old_lr = optimizer.param_groups[0]['lr']
    new_lr = old_lr * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def reduce_learning_rate_on_milestones(loss, optimizer, lr_gamma, lr_milestones):
    """
    used to multiply learning rate by gamma (<1) every time loss crosses some value.
    for examples if milestones is [1., 0.5] then potentially learning rate will be decreased
    when loss goes below 1.0, and then again when it goes below 0.5.
    :param loss: float
    :param optimizer: torch optimizer
    :param lr_gamma: float
    :param lr_milestones: a list of floats
    :return:
    """
    if lr_milestones and loss < lr_milestones[0]:
        reduce_learning_rate(optimizer, lr_gamma)
        lr_milestones.pop(0)


def l2_loss(targets, preds):
    """
    l2 norm loss. used instead of RMSE just because lr milestones were already tuned for this scale
    :param targets: torch tensor
    :param preds: torch tensor
    :return:
    """
    loss_per_sample = torch.sqrt(torch.pow(targets - preds, 2).view(targets.shape[0], -1).sum(dim=1))
    return loss_per_sample.mean()


def validate(model, val_dataloader):
    """
    run model on validation set and return mean loss
    :param model: DnCNN
    :param val_dataloader: torch.utils.data.Dataloader
    :return:
    """
    val_loss_jpgs = 0.
    val_loss = 0.
    for batch in val_dataloader:
        bmps, jpgs, y, indices = batch
        if torch.cuda.is_available():
            bmps, jpgs = bmps.cuda(), jpgs.cuda()
        bmps_pred = model(jpgs)
        loss = l2_loss(bmps, bmps_pred)
        loss_jpgs = l2_loss(bmps, jpgs)
        w = jpgs.shape[0] / float(len(val_dataloader.dataset))
        val_loss += loss * w
        val_loss_jpgs += loss_jpgs * w
    return val_loss, val_loss_jpgs


def train(images_root,
          train_images,
          val_images,
          checkpoints_path,
          learning_rate,
          learning_rate_milestones,
          learning_rate_gamma,
          num_layers,
          num_epochs,
          batch_size,
          num_workers,
          print_step,
          val_step,
          resume_from):
    """
    A function to train DnCNN model on the supplied data.
    :param images_root: path to images
    :param train_images: path to a txt file with training images
    :param val_images: path to a txt file with validation images
    :param checkpoints_path: path to save / load model checkpoints
    :param learning_rate: learning rate for optimizing network parameters
    :param learning_rate_milestones: milestones at which learning rate will be reduced. see  reduce_learning_rate_on_milestones docs
    :param learning_rate_gamma: when reduces, learning rate will be multiplied by this value
    :param num_layers: number of layers DnCNN models have
    :param num_epochs: number of epochs to train model
    :param batch_size: batch size for training and validation
    :param num_workers: number of processes torch uses to create batches
    :param print_step: statistics will be print every print_step steps (batches)
    :param val_step: model will be validated and saved every val_step epochs
    :param resume_from: a checkpoint to resume training from. epoch and step will be started at the respected values, so keep in mind which learning_rate_milestones to give
    :return:
    """
    dl_train, dl_val = load_train_validation_data(images_root, train_images, val_images, batch_size, num_workers)

    model = DnCNN(num_layers)
    if torch.cuda.is_available():
        print('running on GPU')
        model.cuda()
    else:
        print('running on CPU')
    last_epoch, step = -1, 0
    if resume_from:
        last_epoch, step = load_model(model, resume_from)
        print(f'loaded checkpoint: {resume_from}')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(checkpoints_path)  # write statistics for tensorboard

    # training
    model.train()
    for epoch in range(last_epoch + 1, num_epochs):
        for batch in dl_train:
            optimizer.zero_grad()
            bmps, jpgs, y, indices = batch
            if torch.cuda.is_available():
                bmps, jpgs = bmps.cuda(), jpgs.cuda()
            bmps_pred = model(jpgs)
            loss = l2_loss(bmps, bmps_pred)
            reduce_learning_rate_on_milestones(loss, optimizer, learning_rate_gamma, learning_rate_milestones)
            loss.backward()
            optimizer.step()
            if step % print_step == 0:
                print('epoch {}, step {}) loss: {:.8f} learning rate: {:.8f}'.format(epoch,
                                                                                     step,
                                                                                     loss,
                                                                                     optimizer.param_groups[0]['lr']))
            writer.add_scalar('train_loss', loss, step)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step)
            step += 1

        if epoch % val_step == 0:
            # validation
            model.eval()
            with torch.no_grad():
                val_loss, val_loss_jpgs = validate(model, dl_val)
            model.train()
            writer.add_scalar('validation_loss', val_loss, step)
            writer.add_scalar('validation_loss_jpgs', val_loss_jpgs, step)
            print(f'##### epoch {epoch}) validation loss: {val_loss} (loss jpgs: {val_loss_jpgs} #####\n')

            # save checkpoint
            save_model(model, epoch, step, checkpoints_path)

        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_root', '-ir', default='images', help='path to images')
    parser.add_argument('--train_images', '-ti', default='data_splits/train_images.txt', help='path to a txt file with training images')
    parser.add_argument('--val_images', '-vi', default='data_splits/val_images.txt', help='path to a txt file with validation images')
    parser.add_argument('--checkpoints_path', '-cp', default='checkpoints', help='path to save / load model checkpoints')
    parser.add_argument('--resume_from', '-rf', default=None, help='a checkpoint to resume training from. epoch and step will be started at the respected values, so keep in mind which learning_rate_milestones to give')
    parser.add_argument('--learning_rate', '-lr', default=0.001, help='learning rate for optimizing network parameters')
    parser.add_argument('--learning_rate_milestones', '-lrm', default=[4., 3., 3.3], type=float, nargs='+', help='milestones at which learning rate will be reduced. see  reduce_learning_rate_on_milestones docs')
    parser.add_argument('--learning_rate_gamma', '-lrg', default=0.1, type=float, help='when reduces, learning rate will be multiplied by this value')
    parser.add_argument('--num_layers', '-nl', default=3, type=int, help='number of layers DnCNN models have')
    parser.add_argument('--num_epochs', '-ne', default=1000, type=int, help='number of epochs to train model')
    parser.add_argument('--batch_size', '-bs', default=32, type=int, help='batch size for training and validation')
    parser.add_argument('--num_workers', '-nw', default=1, type=int, help='number of processes torch uses to create batches')
    parser.add_argument('--print_step', '-ps', default=1, type=int, help='statistics will be print every print_step steps (batches)')
    parser.add_argument('--val_step', '-vs', default=10, type=int, help='model will be validated and saved every val_step epochs')
    args = parser.parse_args()

    train(**vars(args))
