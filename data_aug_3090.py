import os
import torch.nn as nn
import gym
import torch
import cv2
import numpy as np
import argparse
from a2c_ppo_acktr.interrupt_envs import make_vec_envs
import matplotlib.pyplot as plt
import copy
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.model import Policy
from PPO_PAR_SARFA_new import SARFA
from scipy.ndimage.filters import gaussian_filter
from mydataset import OriginDataset
import torch.optim as optim


def main():
    device_num = 2
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # if gpu is to be used
    param_path = './model/origin_model/origin_1.pth'
    envs = make_vec_envs('BreakoutNoFrameskip-v4', 1, 1, 0.99, None, device, False)
    actor_critic = Policy(envs.observation_space.shape,
                          envs.action_space,
                          base_kwargs={'recurrent': False})
    actor_critic.load_state_dict(torch.load(param_path, map_location='cpu')[0].state_dict())
    actor_critic.to(device)


    actor_critic_playerB = Policy(envs.observation_space.shape,
                                  envs.action_space,
                                  base_kwargs={'recurrent': False})
    actor_critic_playerB.load_state_dict(torch.load(param_path, map_location='cpu')[0].state_dict())
    actor_critic_playerB.to(device)

    # 创建SARFA对象
    def get_mask(center, size, r):
        y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
        keep = x * x + y * y <= 1
        mask = np.zeros(size)
        mask[keep] = 1  # select a circle of pixels
        mask = gaussian_filter(mask, sigma=r)  # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
        return mask / mask.max()

    def get_all_mask_pic(d=5, r=5):
        cnt = 0
        all_mask_pic = np.zeros((int((np.ceil(84 / d)) ** 2), 4, 84, 84))
        for i in range(0, 84, d):
            for j in range(0, 84, d):
                mask = get_mask(center=[i, j], size=[84, 84], r=r)[np.newaxis, :]
                mask = np.concatenate([mask, mask, mask, mask])
                all_mask_pic[cnt, :, :] = mask
                cnt += 1
        return all_mask_pic

    all_mask_pic = get_all_mask_pic()
    aim_sarfa = SARFA(actor_critic, all_mask_pic, device_num)

    BATCH_SIZE = 128
    origindataset = OriginDataset()
    origin_data_loader = torch.utils.data.DataLoader(dataset=origindataset,
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True)
    # 目前最佳adam lr=5e-4 eps=1e-5 训练一大轮没什么问题
    optimizer = optim.Adam(actor_critic_playerB.parameters(), lr=5e-4, eps=1e-5)
    # optimizer = optim.RMSprop(actor_critic_playerB.parameters(), lr=6e-4, alpha=0.95)
    max_grad_norm = 0.5

    def frame_process(frame, variant):
        # random_num = np.random.randint(50)
        if variant == 'variant-rectangle':
            width = 2
            random_width = np.random.randint(1, 5)
            width = width + random_width
            random_num = np.random.randint(-30, 30)
            random_num2 = np.random.randint(-30, 30)
            frame[:, :, 40 + random_num:40 + random_num + width, :][:, :, :,35 + random_num2:35 + random_num2 + width] = 40 + random_num2
        elif variant == 'noise':
            noise = np.random.normal(0, 100 ** 0.5, (BATCH_SIZE, 4, 84, 84))
            frame = frame + noise
        return frame

    best_loss = 1
    for epoch in range(5):
        print('epoch:.{}'.format(epoch))
        for i, image in enumerate(origin_data_loader):
            if image.shape[0] != BATCH_SIZE:
                continue
            image = image.float()
            with torch.no_grad():
                aim_attention, _ = aim_sarfa.score_frame(image, 0, 0)

            eval_sarfa = SARFA(actor_critic_playerB, all_mask_pic, device_num)

            noise_image = frame_process(image, 'variant-rectangle').float()

            eval_attention, _ = eval_sarfa.score_frame(noise_image, 0, 0)

            optimizer.zero_grad()

            re_loss = (aim_attention - eval_attention).pow(2).mean() * 100

            re_loss.backward()

            nn.utils.clip_grad_norm_(actor_critic_playerB.parameters(),
                                     max_grad_norm)

            optimizer.step()

            if re_loss.item() < best_loss:
                best_loss = re_loss
                torch.save([actor_critic_playerB],
                           './model/offline_model/offline_guide_lr=5e-4_x100_adam_noise=100_best.pth')

            if i % 50 == 0:
                print('this time loss {}'.format(re_loss))
                # plt.imshow(aim_attention[0])
                # plt.show()
                # plt.imshow(eval_attention[0])
                # plt.show()

        torch.save([actor_critic_playerB], './model/offline_model/offline_guide_lr=5e-4_{}x100_adam_noise=100.pth'.format(epoch))


if __name__ == "__main__":
    main()