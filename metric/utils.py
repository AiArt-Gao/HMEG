

import os
import shutil
from PIL import Image
import numpy as np
import cv2

from tqdm import tqdm





def adjust_shape(src_dir, tar_dir):

    for file in os.listdir(tar_dir):
        W, H = Image.open(os.path.join(src_dir, file.split('.')[0][:-7] + '.png')).size
        img = Image.open(os.path.join(tar_dir, file)).convert('L').resize((W, H)).save(os.path.join(tar_dir, file))

        # img_array = np.asarray(img)
        # ret, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        # img_binary = 255 - binary
        # # kernel = np.ones((3, 3), np.uint8)
        # # erosion = cv2.erode(img_binary, kernel)
        # y, x = np.where(img_binary == 255)
        # img = Image.fromarray(img_array[y.min(): y.max(), x.min(): x.max()]).resize((W, H))
        # L = max(W, H)
        # canvas = Image.new('L', (L, L), (255,))
        # canvas.paste(img, (round((L - W) / 2), round((L - H) / 2)))
        # canvas = canvas.resize((256, 256))
        # canvas.save(os.path.join(tar_dir, file))




def to_binary(data_dir):
    for file in tqdm(os.listdir(data_dir)):
        img = Image.open(os.path.join(data_dir, file)).convert('L')
        img_array = np.asarray(img)
        ret, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        img = Image.fromarray(255 - binary).resize((256, 256))
        img.save(os.path.join(data_dir, file))


if __name__ == '__main__':

    # tar_dir = '../../datasets/results/cycle_gan'
    # for img in os.listdir(tar_dir):
    #     img_new = img.split('.')[0][:-7] + '.png'
    #     shutil.move(os.path.join(tar_dir, img), os.path.join(tar_dir, img_new))

    # adjust_shape('../../datasets/results/原始数据/trainA',
    #              '../../datasets/results/cycle_gan')
    to_binary('../../datasets/results_test/test_npy_sub')
    to_binary('../../datasets/results_test/test_npy_nosub')
    to_binary('../../datasets/results_test/test_npy_grid_sub')
    to_binary('../../datasets/results_test/test_npy_grid_nosub')
    # to_binary('../../datasets/results_test/test_nomask_nosub')
    pass