import argparse
import os
import cv2
import numpy as np
import torch
import pytorch_ssim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from vqvae import VQVAE
from torch.autograd import Variable

def load_model(checkpoint, device):
    ckpt = torch.load(os.path.join('checkpoint', checkpoint))

    model = VQVAE()
    model = model.to(device)
    # model = torch.nn.DataParallel(model) 
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    
    model.eval()

    return model


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--weights', type=str, default='capsule/vqvae_500.pt')
    parser.add_argument('--data_path', type=str, default='/mnt/qiuzheng/data/mvtec/capsule/test')
    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    model = load_model(args.weights, device)


    transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    print(dataset.class_to_idx)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch,
                                            shuffle=False,
                                            num_workers=4)
    for i, (img, label) in enumerate(dataloader):
        img = img.to(device)
        out, _, min_dist_t, min_dist_b = model(img)
        # save_image(torch.cat([img, out], 0), os.path.join(args.save_path, str(i) + '.png'), normalize=True, range=(-1, 1))
        draw_img = torch.squeeze(img).mul(0.5).add_(0.5).mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy().copy()
        draw_out = torch.squeeze(out).mul(0.5).add_(0.5).mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy().copy()

        cv2.imwrite('./results/' + str(i) + '_ori.png', draw_img)
        cv2.imwrite('./results/' + str(i) + '_out.png', draw_out)
        img1 = cv2.imread('./results/' + str(i) + '_ori.png')
        img2 = cv2.imread('./results/' + str(i) + '_out.png')

        img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0).cuda()/255.0
        img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0).cuda()/255.0

        print(pytorch_ssim.ssim(img1, img2, size_average=False))

        ssim_loss = pytorch_ssim.SSIM(window_size=11)

        print(ssim_loss(img1, img2))
        if i == 3:
            break
        #draw = np.resize(draw, [768, 768])
        # for i in range(1, 32):
        #     out[i * 8, :, :] = 0
        # for j in range(1, 32):
        #     out[:, j * 8, :] = 0
        min_dist_b = torch.squeeze(min_dist_b).detach().cpu().numpy()

        heatmapshow = None
        heatmapshow = cv2.normalize(min_dist_b, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        

        # cv2.imwrite('1.jpg', draw)
        # for y in range(32):
        #     for x in  range(32):
        #         text = str(min_dist_t[y][x])
        #         cv2.putText(draw, text, (x * 24 + 2, y * 24 + 2), cv2.FONT_HERSHEY_COMPLEX, 0.1, (100, 200, 200), 1)
        cv2.imwrite('./results/' + str(i) + '_heat.png', heatmapshow)
        cv2.imwrite('./results/' + str(i) + '_diff.png', draw_img - draw_out)
        # cv2.imwrite('./res/' + str(i) + '.png', draw)
        

