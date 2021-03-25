import os
import cv2
import time
import image_restoration_tools.tool as tool
import image_restoration_tools.utils as utils
import numpy as np
from skimage.measure import compare_psnr, compare_ssim

agent = tool.Agent()

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path) 
        
        
def restore_dir(imgdir, savedir, path_length=3, isJPEG=False, isBLUR=False):
    makedirs(savedir)
    for f in os.listdir(imgdir):
        if utils.is_image_file(f):
            img = utils.imread_uint(imgdir + '/' + f, 3)
            img_E = agent.restore_image(img, path_length, isJPEG, isBLUR)
            utils.imsave(img_E, savedir + '/' + f)
            
            
def restore_path(imgpath, savepath, path_length=3, isJPEG=False, isBLUR=False):
    img = utils.imread_uint(imgpath, 3)
    img_E = agent.restore_image(img, path_length, isJPEG, isBLUR)
    utils.imsave(img_E, savepath)
    
    
def restore_image(img, savepath, path_length=3, isJPEG=False, isBLUR=False):
    img_E = agent.restore_image(img, path_length, isJPEG, isBLUR)
    utils.imsave(img_E, savepath)
