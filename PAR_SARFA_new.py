from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)
from scipy.ndimage.filters import gaussian_filter
import torch
from torch.autograd import Variable
import torch.nn.functional as F

# from scipy.misc.pilutil import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

# Extra imports for f-metric
from scipy.special import softmax
from scipy.stats import entropy
import cv2


import math
import random
import numpy as np
import os
from collections import deque
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import model as m
from atari_wrappers import wrap_deepmind, make_atari
import gym
import matplotlib.pyplot as plt
import time
import torchgeometry as tgm

# fmetric_saliency = score_frame_fmetric(policy_net, my_image, occlude)
# plt.imshow(fmetric_saliency)
# plt.show()
# frame_f = saliency_on_atari_frame(fmetric_saliency, my_image[3], fudge_factor=3000, channel=2)

class SARFA():

    def __init__(self, model, all_mask_pic):
        self.model = model
        self.cnt = 289
        # 转成tensor上GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.all_mask_pic = torch.from_numpy(all_mask_pic).unsqueeze(0).to(self.device)
        self.gauss = tgm.image.GaussianBlur((13, 13), (3, 3))
        # self.batch_size_list = [i for i in range(2)]
        # self.mask_size_list = [i for i in range(289)]


    # 计算KL散度，这里用了交叉熵是一个意思
    def cross_entropy(self, L_policy, l_policy, L_idx):
        p = np.delete(L_policy, L_idx, axis=2)
        new_p = np.delete(l_policy, L_idx, axis=2)

        KL = entropy(p, new_p, axis=2)
        K = 1. / (1. + KL)
        return K

    # prepro用于裁剪图像
    prepro = lambda picture: cv2.resize(cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY), (84, 84),
                                        interpolation=cv2.INTER_LINEAR).reshape(1, 84, 84) / 255.
    # 下面两个变量都是用于扰动图像的
    searchlight = lambda I, mask: I * mask + gaussian_filter(I, sigma=3) * (1 - mask)  # choose an area NOT to blur
    # occlude = lambda I, mask: I * (1 - mask) + gaussian_filter(I, sigma=3) * mask  # choose an area to blur

    def get_all_mask_pic(self, d=5, r=5):
        all_mask_pic = np.zeros((int((np.ceil(84/d)) ** 2), 4, 84, 84))
        for i in range(0, 84, d):
            for j in range(0, 84, d):
                mask = self.get_mask(center=[i, j], size=[84, 84], r=r)[np.newaxis, :]
                mask = np.concatenate([mask, mask, mask, mask])
                all_mask_pic[self.cnt, :, :] = mask
                self.cnt += 1

        return all_mask_pic


    # 进行mask操作
    def get_mask(self, center, size, r):
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        keep = x * x + y * y <= 1
        mask = np.zeros(size)
        mask[keep] = 1  # select a circle of pixels
        mask = gaussian_filter(mask, sigma=r)  # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
        return mask / mask.max()


    def score_frame_fmetric(self, batch_state):
        all_attention_map = np.zeros((batch_state.shape[0], 84, 84))
        d = 5
        occlude = lambda I, mask: I * (1 - mask) + self.gauss(I.squeeze(1).float()).unsqueeze(1) * mask
        L = self.run_through_model(self.model, batch_state.to(self.device), interp_func=occlude) #()
        L_policy = softmax(L.cpu().data.numpy(), axis=1) #(batch_size, 6)
        L_idx = np.argmax(L_policy, axis=1)   #(batch_size, 1)
        L_idx = np.repeat(L_idx.reshape(-1, 1), 289, axis=1) #(batch_size, 289)
        # print(L_idx)
        l = self.run_through_model(self.model, batch_state.to(self.device), interp_func=occlude, mask=self.all_mask_pic)
        l_policy = softmax(l.cpu().data.numpy(), axis=2) #(batch_size, 289, 6)

        L_policy = np.repeat(L_policy[:, np.newaxis, :], 289, axis=1)   #（batch_size, 289, 6）
        ij = np.indices(L_policy.shape[:-1])
        dP = np.max(L_policy, axis=2) -  l_policy[(*ij, L_idx)]#(batch_size, 289)
        # dP = L_policy[self.batch_size_list, self.mask_size_list, L_idx] - l_policy[self.batch_size_list, self.mask_size_list, L_idx]
        K = self.cross_entropy(L_policy, l_policy, L_idx) #(batch_size, 289)
        tmp_score = (2 * dP * K / (K + dP))[:,:,np.newaxis]
        tmp_score[np.where(dP <= 0)] = 0
        scores = tmp_score.reshape((batch_state.shape[0], int(84 / d) + 1, int(84 / d) + 1)) #(batch_size, 17, 17)
        #TODO 把scores按第1个维度展开，取这个维度的矩阵当中的最大值，即pmax的维度是（batch_size, 1）
        count = 0
        for pic in scores:
            pmax = pic.max()
            pic = cv2.resize(pic, (84, 84),
                                interpolation=cv2.INTER_LINEAR).astype(np.float32)
            attention_map = pmax * pic / pic.max()
            # print(attention_map)
            all_attention_map[count] = attention_map
            count += 1

        return  all_attention_map








    # 更换这个函数就可以换不同的mask方式
    def run_through_model(self, model, history, interp_func=None, mask=None, way='all'):
        # im是一个1*84*84的图像，是下面的第一张
        # print('history:', history.shape)
        if mask is None:
            return model(history)
        else:
            if way == 'all':
                # TODO 使用batch_size加速的时候需要把下面这行注释打开
                history = history.unsqueeze(1)
                # 这里的mask是（289， 4， 84， 84）
                # 这里的history是(state_batch, 4, 84, 84)，所以首先需要把history扩成(state_batch, 289, 4, 84, 84)
                # history = np.repeat(history[np.newaxis, :], self.cnt, axis=0)
                # print('mask.shape:', mask.shape)
                # print('history.shape:', history.shape)
                # im = np.vstack([interp_func(history.cpu().data.numpy(), mask)])
                t1 = time.time()
                im = interp_func(history, mask)
                t2 = time.time()
                # print('function_time:', t2-t1)
                # print(im.shape)

                # 输入的tens_state需要是一个4*84*84的向量
                # tens_state = torch.from_numpy(im)
                tens_state = im.reshape(history.shape[0]*289, 4, 84, 84)


                return model(tens_state).reshape(history.shape[0], 289, 6)


# def get_mask(center, size, r):
#     y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
#     keep = x * x + y * y <= 1
#     mask = np.zeros(size)
#     mask[keep] = 1  # select a circle of pixels
#     mask = gaussian_filter(mask, sigma=r)  # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
#     return mask / mask.max()
#
# def get_all_mask_pic(d=5, r=5):
#     cnt = 0
#     all_mask_pic = np.zeros((int((np.ceil(84 / d)) ** 2), 4, 84, 84))
#     for i in range(0, 84, d):
#         for j in range(0, 84, d):
#             mask = get_mask(center=[i, j], size=[84, 84], r=r)[np.newaxis, :]
#             mask = np.concatenate([mask, mask, mask, mask])
#             all_mask_pic[cnt, :, :] = mask
#             cnt += 1
#
#     return all_mask_pic
#
# all_mask_pic = get_all_mask_pic()
#
#
# env_name = 'SpaceInvaders'
# env_raw = make_atari('{}NoFrameskip-v4'.format(env_name))
# env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)
#
#
# my_state_1 = np.load('my_pic.npy')[np.newaxis, :] * 255.
# my_state_2 = np.load('my_pic_2.npy')[np.newaxis, :] * 255.
# my_state = torch.tensor(np.vstack([my_state_1, my_state_2]))
#
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if gpu is to be used
# model = m.DQN(84, 84, 6, device).to(device)
# model_file = 'SpaceInvaders_best/model_in_32600000.pth'
# model.load_state_dict(torch.load('./{}'.format(model_file)))
# sarfa = SARFA(model, all_mask_pic)
# # my_state = torch.tensor(my_state).unsqueeze(0)
# aa = sarfa.score_frame_fmetric(my_state)
# #
# # import pickle
# plt.imshow(aa[0])
# plt.show()
# pickle.dump(file=open('./try_pic_torch.pkl','wb'), obj=aa)


# model_file = 'SpaceInvaders_best/model_in_32600000.pth'
# image_file = 'spaceinvaders/pic-750.jpg'
#
#
# sarfa = SARFA(env, model_file, device)
# aa = sarfa.score_frame_fmetric()
# plt.imshow(aa)
# plt.show()