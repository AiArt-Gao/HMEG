

import gs

import os
from PIL import Image
import numpy as np
from tqdm import tqdm

from skimage.metrics import structural_similarity as ssim
import cv2


def _load_imgs(img_dir, size):
    imgs_np = []
    for img in tqdm(os.listdir(img_dir)):
        img_np = np.asarray(Image.open(os.path.join(img_dir, img)).resize(size).convert('L')).copy()
        img_np[img_np > 127] = 255
        img_np[img_np <= 127] = 0
        imgs_np.append(img_np)
    return imgs_np


def compute_gs(imgs_np):
    imgs_flatten = []
    for img_np in imgs_np:
        img_np = (255 - img_np)
        img_np = img_np.reshape(1, -1)
        imgs_flatten.append(img_np)
    src = np.concatenate(imgs_flatten, axis=0)
    imgs_rlt = gs.rlts(src, gamma=1.0/8, n=20, i_max=3, L_0=32)
    imgs_rlt = np.mean(imgs_rlt, axis=0)
    return imgs_rlt


def gs_metric(src_dir, tars_dir, size=(64, 64)):
    src_rlt = compute_gs(_load_imgs(src_dir, size))
    geom_scores = []
    for tar_dir in tars_dir:
        tar_rlt = compute_gs(_load_imgs(tar_dir, size))
        score = gs.geom_score(src_rlt, tar_rlt)
        print(score)
        geom_scores.append(score)
    return geom_scores


def compute_ssim(src_dir, tar_dir):
    sims = []
    for img in tqdm(FILES):
        # tar_img = img.split('.')[0] + '_fake_B' + '.png'
        tar_img = img
        if os.path.exists(os.path.join(tar_dir, tar_img)):
            src = np.asarray(Image.open(os.path.join(src_dir, img)).resize((128, 128)).convert('L')).copy()
            tar = np.asarray(Image.open(os.path.join(tar_dir, tar_img)).resize((128, 128)).convert('L')).copy()
            sims.append(ssim(src, tar))
    sims = np.mean(sims)
    return sims


if __name__ == '__main__':
    src_dir = '../../datasets/results/原始数据/test'

    tars_dir = [
        '../../datasets/results_test_src/test_npy_sub',
        '../../datasets/results_test_src/test_npy_nosub',
        '../../datasets/results_test_src/test_npy_grid_sub',
        '../../datasets/results_test_src/test_npy_grid_nosub',
        '../../datasets/results_test_src/test_nomask_nosub',
    ]
    FILES = os.listdir('../../datasets/results_test/test_npy_sub')
    # print(gs_metric(src_dir, tars_dir))

    print(compute_ssim(src_dir, tars_dir[0]))
    print(compute_ssim(src_dir, tars_dir[1]))
    print(compute_ssim(src_dir, tars_dir[2]))
    print(compute_ssim(src_dir, tars_dir[3]))
    print(compute_ssim(src_dir, tars_dir[4]))
