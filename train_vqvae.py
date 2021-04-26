import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from vqvae import VQVAE
from dataset import MvtecDataset
from torch.utils.tensorboard import SummaryWriter


def train(epoch, dataloader, model, optimizer, scheduler, device):
    pbar = tqdm(dataloader)

    criterion = nn.MSELoss()
    latent_loss_weight = 0.2
    sample_size = 8

    for i, (img, label, img0) in enumerate(pbar):
        model.zero_grad()

        img = img.to(device)
        img0 = img0.to(device)

        out, latent_loss, _, _ = model(img)
        # recon_loss = criterion(out, img)
        recon_loss = criterion(out, img0)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_description((
            f"epoch: {epoch}; mse: {recon_loss.item():.5f}; "
            f"latent: {latent_loss.item():.3f}; "
            f"lr: {lr:.5f}"))

        if i % 200 == 0:
            model.eval()
            sample = img[:sample_size]

            with torch.no_grad():
                out, _, _, _ = model(sample)
            if not os.path.exists("sample"):
                os.makedirs('sample')
            utils.save_image(
                torch.cat([sample, out], 0),
                f"sample/{str(epoch).zfill(5)}_{str(i).zfill(5)}.png",
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            model.train()


def main(args):

    transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = MvtecDataset(args.dataroot, transform=transform, classes=args.classes)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             drop_last=True,
                                             pin_memory=True)

    model = VQVAE().to(args.device)
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model,
                                      device_ids=args.gpu_ids,
                                      output_device=args.gpu_ids[0])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         args.milestones,
                                                         gamma=0.1)
    elif args.sched == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=0.95)
    elif args.sched == 'cycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            args.lr,
            total_steps=len(dataset) // args.batch_size * args.epochs,
            pct_start=0.1,
            div_factor=100,
            final_div_factor=50)

    for i in range(1, args.epochs + 1):
        train(i, dataloader, model, optimizer, scheduler, args.device)
        if (i % args.save_step) == 0:
            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"vqvae_{str(i).zfill(3)}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',
                        default='/mnt/qiuzheng/data/mvtec',
                        help='path to dataset')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='input batch size')
    parser.add_argument('--num_workers',
                        type=int,
                        help='number of data loading workers',
                        default=8)
    parser.add_argument('--input_size',
                        type=int,
                        default=640,
                        help='input image size.')
    parser.add_argument('--classes',
                        type=list,
                        default=['capsule'],
                        help='classes.')
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='0',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--epochs',
                        type=int,
                        default=640,
                        help='number of epochs')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='initial learning rate for adam')
    parser.add_argument('--save_step',
                        type=int,
                        default=20,
                        help='frequency of saving model')
    parser.add_argument('--ckpt_dir',
                        type=str,
                        default='./checkpoint/capsule',
                        help='ckpt dir')
    parser.add_argument("--sched", type=str, default='cycle')

    args = parser.parse_args()
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        int_id = int(str_id)
        if int_id >= 0:
            args.gpu_ids.append(int_id)
    args.device = torch.device("cuda:{}".format(args.gpu_ids[0]) if torch.cuda.
                               is_available() else 'cpu')
    print(args)

    main(args)
