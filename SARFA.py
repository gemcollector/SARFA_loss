from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)
from scipy.ndimage.filters import gaussian_filter
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
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


# fmetric_saliency = score_frame_fmetric(policy_net, my_image, occlude)
# plt.imshow(fmetric_saliency)
# plt.show()
# frame_f = saliency_on_atari_frame(fmetric_saliency, my_image[3], fudge_factor=3000, channel=2)

class SARFA():

    def __init__(self, env, model_file, device):
        self.model = m.DQN(84, 84, env.action_space.n, device).to(device)
        self.model.load_state_dict(torch.load('./{}'.format(model_file)))


    # 计算KL散度，这里用了交叉熵是一个意思
    def cross_entropy(self, L_policy, l_policy, L_idx):
        p = L_policy[:L_idx]
        p = np.append(p, L_policy[L_idx + 1:])
        new_p = l_policy[:L_idx]
        new_p = np.append(new_p, l_policy[L_idx + 1:])
        KL = entropy(p, new_p)
        K = 1. / (1. + KL)
        return K

    # prepro用于裁剪图像
    prepro = lambda picture: cv2.resize(cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY), (84, 84),
                                        interpolation=cv2.INTER_LINEAR).reshape(1, 84, 84) / 255.
    # 下面两个变量都是用于扰动图像的
    searchlight = lambda I, mask: I * mask + gaussian_filter(I, sigma=3) * (1 - mask)  # choose an area NOT to blur
    # occlude = lambda I, mask: I * (1 - mask) + gaussian_filter(I, sigma=3) * mask  # choose an area to blur

    # 进行mask操作
    def get_mask(self, center, size, r):
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        keep = x * x + y * y <= 1
        mask = np.zeros(size)
        mask[keep] = 1  # select a circle of pixels
        mask = gaussian_filter(mask, sigma=r)  # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
        return mask / mask.max()

    def score_frame_fmetric(self, state):
        occlude = lambda I, mask: I * (1 - mask) + gaussian_filter(I, sigma=3) * mask  # choose an area to blur
        L =self.run_through_model(self.model, state, interp_func=occlude)
        L_policy = softmax(L.cpu().data.numpy())
        L_idx = np.argmax(L_policy)
        d = 5
        r = 5
        scores = np.zeros((int(84 / d) + 1, int(84 / d) + 1))  # saliency scores S(t,i,j)
        for i in range(0, 84, d):
            for j in range(0, 84, d):
                mask = self.get_mask(center=[i, j], size=[84, 84], r=r)
                l = self.run_through_model(self.model, state, interp_func=occlude, mask=mask)
                # Our f-metric
                l_policy = softmax(l.cpu().data.numpy())
                dP = L_policy[L_idx] - l_policy[L_idx]
                if (dP > 0):
                    K = self.cross_entropy(L_policy, l_policy, L_idx)
                    scores[int(i / d), int(j / d)] = 2 * dP * K / (K + dP)
        pmax = scores.max()
        scores = cv2.resize(scores, (84, 84),
                            interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return pmax * scores / scores.max()


    # 更换这个函数就可以换不同的mask方式
    def run_through_model(self, model, history, interp_func=None, mask=None):
        # im是一个1*84*84的图像，是下面的第一张

        prepro = lambda picture: picture.reshape(1, 84, 84)

        if mask is None:
            im = prepro(history[0])
        else:
            assert (interp_func is not None, "interp func cannot be none")
            im = prepro(history[0])  # perturb input I -> I'

        for i in range(1, 3):
            tmp_im = prepro(history[i])
            im = np.vstack((im, tmp_im))

        if mask is None:
            tmp_im = prepro(history[3])
        else:
            tmp_im = interp_func(prepro(history[3]).squeeze(), mask).reshape(1, 84, 84)
        im = np.vstack((im, tmp_im))

        # 输入的tens_state需要是一个4*84*84的向量
        tens_state = torch.Tensor([im])

        return model(tens_state)[0]

# model_file = 'SpaceInvaders_best/model_in_32600000.pth'
# image_file = 'spaceinvaders/pic-750.jpg'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if gpu is to be used
# env_name = 'SpaceInvadersNoFrameskip-v4'
# env = gym.make(env_name)
#
#
# sarfa = SARFA(env, model_file, device)
# aa = sarfa.score_frame_fmetric()
# plt.imshow(aa)
# plt.show()