import os
import cv2
import gym
import time
import tool
import utils
import numpy as np
from skimage.measure import compare_psnr, compare_ssim

agent = tool.Agent()

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path) 
        
        
def restore_path(imgpath, savepath):
    makedirs(savepath)
    for f in os.listdir(imgpath):
        if utils.is_image_file(f):
            img = utils.imread_uint(imgpath + '/' + f, 3)
            print(f)
            if 'JPEG' in f:
                img_E = agent.restore_image(img, path_length=3, isJPEG=True, isBLUR=False)
            elif 'BLUR' in f:
                img_E = agent.restore_image(img, path_length=3, isJPEG=False, isBLUR=True)
            else:
                img_E = agent.restore_image(img, path_length=3, isJPEG=False, isBLUR=False)
            utils.imsave(img_E, savepath + '/' + f)
            

if __name__ == '__main__':
    restore_path("./test_images/", "./results/")