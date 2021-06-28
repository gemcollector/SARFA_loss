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
import argparse
from PAR_SARFA import SARFA

import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

# 1. GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if gpu is to be used

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='SpaceInvaders')
parser.add_argument('--func', type=str)
# parser.add_argument('--gpu', type=int)
parser.add_argument('--param_transfer', type=bool, default=False)
parser.add_argument('--AAE_transfer', type=bool, default=False)
parser.add_argument('--frozen', type=bool, default=False)
parser.add_argument('--regular', type=float, default=0.1)
args = parser.parse_args()


if not os.path.exists('{}_{}'.format(args.name, args.func)):
    os.mkdir('{}_{}'.format(args.name, args.func))


env_name = args.name
# 3. e
#env_name = 'Breakout'
#env_name = 'SpaceInvaders'
#env_name = 'Riverraid'
#env_name = 'Seaquest'
#env_name = 'MontezumaRevenge'
#env_name = 'DemonAttack'
#env_name = 'Phoenix'
#env_name = 'Pong'
env_raw = make_atari('{}NoFrameskip-v4'.format(env_name))
env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)

c,h,w = m.fp(env.reset()).shape
n_actions = env.action_space.n
print(n_actions)
print('regular:', args.regular)

# 4. Network reset
policy_net = m.DQN(h, w, n_actions, device).to(device)
target_net = m.DQN(h, w, n_actions, device).to(device)
policy_net.apply(policy_net.init_weights)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

aim_model = m.DQN(h, w, n_actions, device).to(device)
model_file = 'SpaceInvaders_best/model_in_32600000.pth'
aim_model.load_state_dict(torch.load('./{}'.format(model_file)))
aim_sarfa = SARFA(aim_model)

# SpaceInvaders_net = m.DQN(h, w, 6, device).to(device)
# SpaceInvaders_net.load_state_dict(torch.load('./SpaceInvaders_best/model_in_32600000.pth'))

class AUTOENCODER(nn.Module):

    def __init__(self, device):
        super(AUTOENCODER, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        # self.fc2 = nn.Linear(512, outputs)
        self.device = device

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.0)

        if type(m) == nn.Conv2d:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # m.bias.data.fill_(0.1)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # return x.view(x.size(0), -1) # 最后出来的应该是64*7*7
        return x     #或者最后出来的是特征图，直接让这个特征图进Decoder

print('args.AAE_transfer：', args.AAE_transfer, type(args.AAE_transfer))
if args.AAE_transfer:
    GPU = 1
    DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")
    Q = AUTOENCODER(DEVICE).cuda()
    Q.load_state_dict(torch.load('Breakout_encoder_weights.pth'))

    for target_param, local_param in zip(policy_net.parameters(), Q.parameters()):
        print('aaa')
        if len(target_param.shape) != 4:
            break
        target_param.data.copy_(local_param.data)

    for target_param, local_param in zip(target_net.parameters(), Q.parameters()):
        if len(target_param.shape) != 4:
            break
        target_param.data.copy_(local_param.data)






print('args.param_transfer:', args.param_transfer)
if args.param_transfer:
    param_net = m.DQN(h, w, 9, device).to(device)
    param_net.load_state_dict(torch.load('./MsP_best/Msp_best.pth'))
    for target_param, local_param in zip(policy_net.parameters(), param_net.parameters()):
        print('aaaa')
        if len(target_param.shape) != 4:
            break
        target_param.data.copy_(local_param.data)

print('args.frozen:', args.frozen)
if args.frozen:
    for name, param in policy_net.named_parameters():
        if "fc1" in name:
            param.requires_grad = True
            print('name:{}'.format(name), param.requires_grad)
        elif "fc2" in name:
            param.requires_grad = True
            print('name:{}'.format(name), param.requires_grad)
        else:
            param.requires_grad = False
            print('name:{}'.format(name), param.requires_grad)


# 5. DQN hyperparameters
ATTENTION_UPDATE = 50
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE = 10000
NUM_STEPS = 50000000
M_SIZE = 1000000
POLICY_UPDATE = 4
EVALUATE_FREQ = 200000
if not args.frozen:
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0000625, eps=1.5e-4)
else:
    optimizer = optim.Adam(filter(lambda p :p.requires_grad, policy_net.parameters()), lr=0.0000625, eps=1.5e-4)

# replay memory and action selector
memory = m.ReplayMemory(M_SIZE, [5,h,w], n_actions, device)
sa = m.ActionSelector(EPS_START, EPS_END, policy_net, EPS_DECAY, n_actions, device)

steps_done = 0

def optimize_model(train):
    if not train:
        return
    state_batch, action_batch, reward_batch, n_state_batch, done_batch = memory.sample(BATCH_SIZE)

    # Add attention map to optimizing process
    # with torch.no_grad():
    #     aim_attention = aim_sarfa.score_frame_fmetric(state_batch)
    # #TODO to calculate the aim_attention correctly!!
    #
    # eval_sarfa = SARFA(policy_net)
    # eval_attention = eval_sarfa.score_frame_fmetric(state_batch)
    # re_loss = (torch.from_numpy((aim_attention - eval_attention))).pow(2).mean()

    q = policy_net(state_batch).gather(1, action_batch)
    nq = target_net(n_state_batch).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (nq * GAMMA)*(1.-done_batch[:,0]) + reward_batch[:,0]

    # Compute Huber loss
    loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.grad == None:
            continue
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def attention_optimize_model(train, regular):
    if not train:
        return
    for i in range(1):
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = memory.sample(BATCH_SIZE)

        # Add attention map to optimizing process
        with torch.no_grad():
            aim_attention = aim_sarfa.score_frame_fmetric(state_batch)
        # TODO to calculate the aim_attention correctly!!

        eval_sarfa = SARFA(policy_net)
        eval_attention = eval_sarfa.score_frame_fmetric(state_batch)
        re_loss = (torch.from_numpy((aim_attention - eval_attention))).pow(2).mean()

        q = policy_net(state_batch).gather(1, action_batch)
        nq = target_net(n_state_batch).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (nq * GAMMA)*(1.-done_batch[:,0]) + reward_batch[:,0]

        # Compute Huber loss
        loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1)) + regular * re_loss

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            if param.grad == None:
                continue
            param.grad.data.clamp_(-1, 1)
        optimizer.step()



def evaluate(step, policy_net, device, env, n_actions, eps=0.05, num_episode=5):
    env = wrap_deepmind(env)
    sa = m.ActionSelector(eps, eps, policy_net, EPS_DECAY, n_actions, device)
    e_rewards = []
    q = deque(maxlen=5)
    count = 0
    for i in range(num_episode):
        env.reset()
        e_reward = 0
        for _ in range(10): # no-op
            n_frame, _, done, _ = env.step(0)
            n_frame = m.fp(n_frame)
            q.append(n_frame)

        while not done:
            count += 1
            state = torch.cat(list(q))[1:].unsqueeze(0)
            action, eps = sa.select_action(state, train)
            n_frame, reward, done, info = env.step(action)
            n_frame = m.fp(n_frame)
            q.append(n_frame)
            
            e_reward += reward
        e_rewards.append(e_reward)

    f = open("{}_{}/file.txt".format(args.name, args.func),'a')
    f.write("%f, %d, %d\n" % (float(sum(e_rewards))/float(num_episode), step, num_episode))
    f.close()

q = deque(maxlen=5)
done = True
eps = 0
episode_len = 0

progressive = tqdm(range(NUM_STEPS), total=NUM_STEPS, ncols=50, leave=False, unit='b')
for step in progressive:
    if done: # life reset !!!
        env.reset()
        sum_reward = 0
        episode_len = 0
        img, _, _, _ = env.step(1) # BREAKOUT specific !!!
        for i in range(10): # no-op
            n_frame, _, _, _ = env.step(0)
            n_frame = m.fp(n_frame)
            q.append(n_frame.numpy())
        
    train = len(memory) > 50000
    # Select and perform an action
    state = torch.from_numpy(np.concatenate(list(q))[1:][np.newaxis, :])
    action, eps = sa.select_action(state, train)
    n_frame, reward, done, info = env.step(action)
    n_frame = m.fp(n_frame)

    # 5 frame as memory
    q.append(n_frame.numpy())
    memory.push(np.concatenate(list(q))[np.newaxis, :], action, reward, done) # here the n_frame means next frame from the previous time step


    episode_len += 1

    # Perform one step of the optimization (on the target network)
    if step % POLICY_UPDATE == 0:
        optimize_model(train)
    
    # Update the target network, copying all weights and biases in DQN
    if step % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if step % ATTENTION_UPDATE == 0:
        attention_optimize_model(train, args.regular)

    
    if (step) % EVALUATE_FREQ == 0:
        evaluate(step, policy_net, device, env_raw, n_actions, eps=0.05, num_episode=15)
        policy_net.to('cpu')
        torch.save(policy_net.state_dict(), '{}_{}/model_in_{}.pth'.format(args.name, args.func, step))
        policy_net.to(device)
        print('yes!!!!!!!')


