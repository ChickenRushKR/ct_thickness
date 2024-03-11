import argparse
import logging
import os

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from Unet_Segmentation.utils.data_loading import BasicDataset
from Unet_Segmentation.unet import UNet
from Unet_Segmentation.utils.utils import plot_img_and_mask
from datetime import datetime as dt

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--inputdir', '-id', metavar='INPUT', help='Filenames of input images dir')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(in_files):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return list(map(_generate_name, in_files))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    # in_files = args.input
    in_files = os.listdir(args.inputdir)
    out_files = get_output_filenames(in_files)
    
    now = dt.now()
    time_now = now.strftime("%m%d%H%M")
    print(time_now)
    out_dir = './result/' + time_now
    try:
        os.mkdir(out_dir)
        print(out_dir, 'was created.')
    except:
        print(out_dir, 'is already existed')
        exit()
    out_pred_dir = './result/' + time_now + '/pred/'
    try:
        os.mkdir(out_pred_dir)
        print(out_pred_dir, 'was created.')
    except:
        print(out_pred_dir, 'is already existed')
        exit()
    out_cat_dir = './result/' + time_now + '/cat/'
    try:
        os.mkdir(out_cat_dir)
        print(out_cat_dir, 'was created.')
    except:
        print(out_cat_dir, 'is already existed')
        exit()
    out_data_dir = './result/' + time_now + '/data/'
    try:
        os.mkdir(out_data_dir)
        print(out_data_dir, 'was created.')
    except:
        print(out_data_dir, 'is already existed')
        exit()
    
    
    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')
    print('Model loaded!')
    excel = pd.read_csv('Dataset/points_647.csv')
    for i, filename in enumerate(in_files):
        
        logging.info(f'\nPredicting image {args.inputdir + filename} ...')
        # print(f'Predicting image {filename} ...')
        img = Image.open(args.inputdir + filename)
        img_gray = Image.open(args.inputdir + filename).convert("L")
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)


        # print(excel[excel['Name']==filename])
        pt1x=int(excel[excel['Name']==filename]['pt1x'])
        pt1y=int(excel[excel['Name']==filename]['pt1y'])
        pt2x=int(excel[excel['Name']==filename]['pt2x'])
        pt2y=int(excel[excel['Name']==filename]['pt2y'])
        res = np.asarray(mask_to_image(mask))
        # print(pt1x,pt1y,pt2x,pt2y)
        # print(res[pt2y:pt1y,:])
        # exit()
        count1 = np.count_nonzero(res[pt2y:pt1y,:])
        avgwidth = count1 / abs(pt1y-pt2y)
        print(filename, avgwidth)
        # print('avgwidth=', avgwidth)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_pred_dir + filename)
            img.save(out_data_dir + filename)
            logging.info(f'Mask saved to {out_pred_dir + filename}')


        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)