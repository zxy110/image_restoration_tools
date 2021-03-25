# -*- coding: utf-8 -*-
import os
import cv2
import math
import random
import image_restoration_tools.utils as utils
import torch
import numpy as np
from numpy import uint8
from numpy import float32
from numpy import hstack

from PIL import Image
from io import BytesIO
from skimage import filters
from scipy.stats import gamma
from scipy.ndimage import correlate
from skimage.util import view_as_windows

from image_restoration_tools.network import Net

import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self):
        self.e = Estimate()
        self.d = Degrading()
        self.t = Tool()
        
    # 计算失真分数
    def calscore(self, x):
        n,b = self.e.score(x)
        return [n,b]
    
    # 对图像执行type类型的工具
    def step(self, img, type):
        return self.t.step(img, type)
    
    
    def restore_image(self, img, path_length=6, isJPEG=False, isBLUR=False):
        # 预处理
        if isJPEG:
            img = self.step(img, 2) 
            path_length = 1
        if isBLUR:
            img = self.step(img, 3) 
            path_length = 1

        # 迭代复原
        steps = 0
        tool0 = tool1 = True
        for _ in range(path_length):                
            obs = self.calscore(img)
            if obs[0] > 2 and tool0:
                tool = 0
            elif tool1:
                tool = 1

            img_E = self.step(img, tool) 
            next_obs = self.calscore(img_E)

            # print("step:{} obs:[{},{}]->[{},{}] take tool:{} {} {}".format(steps, obs[0], obs[1], next_obs[0], next_obs[1], tool, tool0, tool1))

            if tool==0 and (obs[tool]-next_obs[tool]<0.3): 
                tool0 = False
                continue
            elif tool==1 and (next_obs[tool]>obs[tool]): 
                tool1 = False
                continue
                
            if not (tool0 or tool1):  break
            if next_obs[0]-obs[0]>5: break
            if obs[0]<1 and obs[1]<0.5: break
            
            steps += 1
            img = img_E
            tool0 = tool1 = True

        return img

    

class Degrading:   
    def noise(self, img, d):
        np.random.seed(seed=0)
        img = utils.uint2single(img)
        img_noise = img + np.random.normal(0, d / 255., img.shape)
        img_noise = utils.single2uint(img_noise)
        return img_noise

    def jpeg(self, img, qf):
        CompressBuffer = BytesIO()
        tmp = Image.fromarray(img)
        tmp.save(CompressBuffer, "JPEG", quality=qf)
        img_jpeg = np.asarray(Image.open(CompressBuffer))
        return img_jpeg

    def blur(self, img, b):
        img_blur = cv2.GaussianBlur(img, (21, 21), b)
        return img_blur


class Tool:
    def __init__(self):
        self.denoise_net = self.net('gaussian_noise.pth').to(device)
        self.dejpeg_net = self.net('jpeg.pth').to(device)
        self.deblur_net = self.net('gaussian_blur.pth').to(device)
            
    def net(self, ck):
        model = Net()
        model.load_state_dict(torch.load("image_restoration_tools/models/" + ck,
                                         map_location=torch.device(device)), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        return model

    
    def sharpen(self, img ,radius=5):
        h,w,chan = img.shape

        GaussBlue1 = np.zeros(img.shape,dtype = uint8)
        GaussBlue2 = np.zeros(img.shape, dtype=uint8)
        GaussBlue3 = np.zeros(img.shape, dtype=uint8)
        Dest_float_img = np.zeros(img.shape, dtype=float32)
        Dest_img = np.zeros(img.shape, dtype=uint8)

        w1 = 0.5
        w2 = 0.5
        w3 = 0.25

        GaussBlue1 = cv2.GaussianBlur(img,(radius,radius),1)
        GaussBlue2 = cv2.GaussianBlur(img,(radius*2-1,radius*2-1),2)
        GaussBlue3 = cv2.GaussianBlur(img,(radius*4-1,radius*4-1),4)

        for i in range(0,h):
            for j in range(0,w):
                for k in range(0,chan):
                    Src = img.item(i,j,k)
                    D1 = Src-GaussBlue1.item(i,j,k)
                    D2 = GaussBlue1.item(i,j,k) - GaussBlue2.item(i,j,k)
                    D3 = GaussBlue2.item(i,j,k) - GaussBlue3.item(i,j,k)
                    if(D1 > 0):
                        sig = 1
                    else:
                        sig = -1
                    Dest_float_img.itemset((i,j,k),(1-w1*sig)*D1+w2*D2+w3*D3+Src)

        Dest_img = cv2.convertScaleAbs(Dest_float_img)
        return Dest_img

    
    def denoise(self, img, type="bilater"):
        if type == "mean":
            # 均值滤波
            img = cv2.blur(img, (5,5))
        elif type == "gaussian":
            # 高斯滤波，高斯矩阵的长与宽都是5，标准差取0
            img = cv2.GaussianBlur(img,(5,5),0)
        elif type == "median":
            # 中值滤波
            img = cv2.medianBlur(img, 3)
        elif type == "bilater":
            # 双边滤波
            img = cv2.bilateralFilter(img,3,5,5)

        return img

    
    # toolId = {0:'DENOISE', 1:'SHARPEN', 2:'DEJPEG', 3:'DEBLUR'}
    def step(self, img, type):
        img_E = img
        # img_E = self.denoise(img_E, 'bilater') # 'mean', 'gaussian', 'median', 'bilater'
        if type == 0: 
            img_E = utils.uint2tensor4(img_E).to(device)
            img_E = self.denoise_net(img_E)
            img_E = utils.tensor2uint(img_E)
        elif type == 1:
            img_E = self.sharpen(img_E, 5)
        elif type == 2: 
            img_E = utils.uint2tensor4(img_E).to(device)
            img_E = self.dejpeg_net(img_E)
            img_E = utils.tensor2uint(img_E)
        elif type == 3:
            img_E = utils.uint2tensor4(img_E).to(device)
            img_E = self.deblur_net(img_E)
            img_E = utils.tensor2uint(img_E)
        
 
        return img_E
    
    
    
class Estimate:
    def score(self, x):
        w, h, _ = x.shape
        if w < 1000 and h < 1000:
            l1,l2,r1,r2 = 0,w,0,h
        else:
            l1,l2,r1,r2 = w//2-128,w//2+128,h//2-128,h//2+128
        
        n = self.noise_score(utils.uint2single(x[:64,:64,0]))["nlevel"][0] * 1000 / 4
        b = self.jpeg_score(x[l1:l2,r1:r2,0])
        n,b = round(n,2),round(b,2)
        
        return n,b

    
    def jpeg_score(self, img):
        m, n = img.shape
        if m < 16 or n < 16:
            return -2.0

        arr = img.astype(np.float)

        d_h = arr[:, 1:] - arr[:, :n-1]
        d_arr = np.zeros((m, int(n/8)-1), dtype=np.float)
        for i in range(1,int(n/8)-1):
            d_arr[:, i] = d_h[:, i*8-1]

        B_h = (np.abs(d_arr)).mean()
        A_h = (8.0*(np.abs(d_h)).mean()-B_h) / 7.0

        sign_h = np.zeros(d_h.shape)
        for i in range(d_h.shape[0]):
            for j in range(d_h.shape[1]):
                if d_h[i,j] > 0:
                    sign_h[i, j]= 1
                elif d_h[i,j] == 0:
                    sign_h[i,j]= 0
                else:
                    sign_h[i,j] =-1

        left_sig = sign_h[:, :n-2]
        right_sig = sign_h[:, 1:]
        Z_h = (left_sig*right_sig<0).mean()

        d_v = arr[1:,:] - arr[:m-1, :]
        d_arr = np.zeros((int(m/8)-1, n), dtype=np.float)
        for i in range(1, int(m/8)-1):
            d_arr[i, :] = d_v[i*8-1,:]

        B_v = (np.abs(d_arr)).mean()
        A_v = (8.0*(np.abs(d_v)).mean()-B_v) / 7.0

        sign_v = np.zeros(d_v.shape)
        for i in range(d_v.shape[0]):
            for j in range(d_v.shape[1]):
                if d_v[i,j]>0:
                    sign_v.itemset((i,j), 1)
                elif d_v[i,j]==0:
                    sign_v.itemset((i,j), 0)
                else:
                    sign_v.itemset((i,j), -1)

        up_sig = sign_v[:m-2, :]
        down_sig = sign_v[1:, :]
        Z_v = (up_sig*down_sig<0).mean()

        B = (B_h + B_v) /2.0
        A = (A_h + A_v) /2.0
        Z = (Z_h + Z_v) /2.0
        
        
        alpha = -927.4240
        beta = 850.8986
        gamma1 = 0.02354451
        gamma2 = 0.01287548
        gamma3 = -0.03414790
        if B == 0.0: B = 1.0
        if A == 0.0: A = 1.0
        if Z == 0.0: Z = 1.0
        
        '''
        score = alpha + beta * (complex(B) ** gamma1) * (complex(A) ** gamma2) * (complex(Z) ** gamma3)
        return Block.real, score.real 
        '''
        if A==0 or Z==0: 
            AZ = 0
        else:
            AZ = (complex(A) * complex(Z)) ** (-1)
        return AZ.real
        
    
    def Tengrad(self, gray): # 灰度方差乘积
        tmp = filters.sobel(gray)
        source=np.sum(tmp**2)
        source=np.sqrt(source)

        return source
    

    def noise_score(self, img, patchsize=7, decim=0, confidence=1 - 1e-6, iterations=3):
        def noiselevel(img, patchsize, decim, confidence, iterations):
            if len(img.shape) < 3:
                img = np.expand_dims(img, 2)

            nlevel = np.ndarray(img.shape[2])
            thresh = np.ndarray(img.shape[2])
            num = np.ndarray(img.shape[2])

            kh = np.expand_dims(np.expand_dims(np.array([-0.5, 0, 0.5]), 0), 2)
            imgh = correlate(img, kh, mode="nearest")
            imgh = imgh[:, 1 : imgh.shape[1] - 1, :]
            imgh = imgh * imgh

            kv = np.expand_dims(np.vstack(np.array([-0.5, 0, 0.5])), 2)
            imgv = correlate(img, kv, mode="nearest")
            imgv = imgv[1 : imgv.shape[0] - 1, :, :]
            imgv = imgv * imgv

            Dh = conv2d_matrix(np.squeeze(kh, 2), patchsize, patchsize)
            Dv = conv2d_matrix(np.squeeze(kv, 2), patchsize, patchsize)

            DD = np.transpose(Dh) @ Dh + np.transpose(Dv) @ Dv

            r = np.double(np.linalg.matrix_rank(DD))
            Dtr = np.trace(DD)

            tau0 = gamma.ppf(confidence, r / 2, scale=(2 * Dtr / r))

            for cha in range(img.shape[2]):
                X = view_as_windows(img[:, :, cha], (patchsize, patchsize))
                X = X.reshape(np.int(X.size / patchsize ** 2), patchsize ** 2, order="F").transpose()

                Xh = view_as_windows(imgh[:, :, cha], (patchsize, patchsize - 2))
                Xh = Xh.reshape(
                    np.int(Xh.size / ((patchsize - 2) * patchsize)), ((patchsize - 2) * patchsize), order="F"
                ).transpose()

                Xv = view_as_windows(imgv[:, :, cha], (patchsize - 2, patchsize))
                Xv = Xv.reshape(
                    np.int(Xv.size / ((patchsize - 2) * patchsize)), ((patchsize - 2) * patchsize), order="F"
                ).transpose()

                Xtr = np.expand_dims(np.sum(np.concatenate((Xh, Xv), axis=0), axis=0), 0)

                if decim > 0:
                    XtrX = np.transpose(np.concatenate((Xtr, X), axis=0))
                    XtrX = np.transpose(
                        XtrX[
                            XtrX[:, 0].argsort(),
                        ]
                    )
                    p = np.floor(XtrX.shape[1] / (decim + 1))
                    p = np.expand_dims(np.arange(0, p) * (decim + 1), 0)
                    Xtr = XtrX[0, p.astype("int")]
                    X = np.squeeze(XtrX[1 : XtrX.shape[1], p.astype("int")])

                # noise level estimation
                tau = np.inf

                if X.shape[1] < X.shape[0]:
                    sig2 = 0
                else:
                    cov = (X @ np.transpose(X)) / (X.shape[1] - 1)
                    d = np.flip(np.linalg.eig(cov)[0], axis=0)
                    sig2 = d[0]

                for _ in range(1, iterations):
                    # weak texture selection
                    tau = sig2 * tau0
                    p = Xtr < tau
                    Xtr = Xtr[p]
                    X = X[:, np.squeeze(p)]

                    # noise level estimation
                    if X.shape[1] < X.shape[0]:
                        break

                    cov = (X @ np.transpose(X)) / (X.shape[1] - 1)
                    d = np.flip(np.linalg.eig(cov)[0], axis=0)
                    sig2 = d[0]

                nlevel[cha] = np.sqrt(sig2)
                thresh[cha] = tau
                num[cha] = X.shape[1]

            # clean up
            img = np.squeeze(img)

            return nlevel, thresh, num


        def conv2d_matrix(H, rows, columns):
            s = np.shape(H)
            rows = int(rows)
            columns = int(columns)

            matr_row = rows - s[0] + 1
            matr_column = columns - s[1] + 1

            T = np.zeros([matr_row * matr_column, rows * columns])

            k = 0
            for i in range(matr_row):
                for j in range(matr_column):
                    for p in range(s[0]):
                        start = (i + p) * columns + j
                        T[k, start : start + s[1]] = H[p, :]

                    k += 1
            return T


        def weak_texture_mask(img, patchsize, thresh):
            if img.ndim < 3:
                img = np.expand_dims(img, 2)

            kh = np.expand_dims(np.transpose(np.vstack(np.array([-0.5, 0, 0.5]))), 2)
            imgh = correlate(img, kh, mode="nearest")
            imgh = imgh[:, 1 : imgh.shape[1] - 1, :]
            imgh = imgh * imgh

            kv = np.expand_dims(np.vstack(np.array([-0.5, 0, 0.5])), 1)
            imgv = correlate(img, kv, mode="nearest")
            imgv = imgv[1 : imgv.shape[0] - 1, :, :]
            imgv = imgv * imgv

            s = img.shape
            msk = np.zeros_like(img)

            for cha in range(s[2]):
                m = view_as_windows(img[:, :, cha], (patchsize, patchsize))
                m = np.zeros_like(m.reshape(np.int(m.size / patchsize ** 2), patchsize ** 2, order="F").transpose())

                Xh = view_as_windows(imgh[:, :, cha], (patchsize, patchsize - 2))
                Xh = Xh.reshape(
                    np.int(Xh.size / ((patchsize - 2) * patchsize)), ((patchsize - 2) * patchsize), order="F"
                ).transpose()

                Xv = view_as_windows(imgv[:, :, cha], (patchsize - 2, patchsize))
                Xv = Xv.reshape(
                    np.int(Xv.size / ((patchsize - 2) * patchsize)), ((patchsize - 2) * patchsize), order="F"
                ).transpose()

                Xtr = np.expand_dims(np.sum(np.concatenate((Xh, Xv), axis=0), axis=0), 0)

                p = Xtr < thresh[cha]
                ind = 0

                for col in range(0, s[1] - patchsize + 1):
                    for row in range(0, s[0] - patchsize + 1):
                        if p[:, ind]:
                            msk[row : row + patchsize - 1, col : col + patchsize - 1, cha] = 1
                        ind = ind + 1

            # clean up
            img = np.squeeze(img)
            return np.squeeze(msk)

        try:
            img = np.array(img)
        except:
            raise TypeError("Input image should be a NumPy ndarray")

        try:
            patchsize = int(patchsize)
        except ValueError:
            raise TypeError("patchsize must be an integer, or int-compatible, variable")

        try:
            decim = int(decim)
        except ValueError:
            raise TypeError("decim must be an integer, or int-compatible, variable")

        try:
            confidence = float(confidence)
        except ValueError:
            raise TypeError("confidence must be a float, or float-compatible, value between 0 and 1")

        if not (confidence >= 0 and confidence <= 1):
            raise ValueError("confidence must be defined in the interval 0 <= confidence <= 1")

        try:
            iterations = int(iterations)
        except ValueError:
            raise TypeError("iterations must be an integer, or int-compatible, variable")

        output = {}
        nlevel, thresh, num = noiselevel(img, patchsize, decim, confidence, iterations)
        mask = weak_texture_mask(img, patchsize, thresh)

        output["nlevel"] = nlevel
        output["thresh"] = thresh
        output["num"] = num
        output["mask"] = mask

        return output


