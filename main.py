from multiprocessing import Process
import time
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym
import torch
import numpy as np
from torch import nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from model import Model
import math
from pathlib import Path

path = Path(f'./weights2')
n_process = 8
n_steps = 128
n_epochs = 4
batch_size = 256
gru_sz = 256
img_sz = (1,80,80)
n_interacts = 4000

def T(x, cuda=True):
    if x.dtype in (np.int8, np.int16, np.int32, np.int64):
        x = torch.from_numpy(x.astype(np.int64))
    elif x.dtype in (np.float32, np.float64):
        x = torch.from_numpy(x.astype(np.float32))
    if cuda:
        x = x.pin_memory().cuda(non_blocking=True)
    return x

def discount_r(reward, next_value):
    res = T(np.zeros([n_steps,n_process]))
    mask = (reward == 0).float()
    res[-1] = next_value*mask[-1] + reward[-1]*(~mask[-1].byte()).float()
    for step in reversed(range(n_steps-1)):
        res[step] = res[step+1]*mask[step]*0.99 + reward[step]
    return res

def prepro(img):
    if len(img.shape) == 3:
        img = img.reshape([1, *img.shape])
    img = img[:, 35:195]
    img = img[:, ::2, ::2, 0]
    img = np.expand_dims(img, 1)
    return img.astype(np.float)

def make_env(seed):
    env = gym.make("Pong-v0")
    env.seed(seed)
    return env
    
def p_log_p(p):
    return p*torch.log(p)

def main():
    start = time.time()
    epsilon = 0.1
    lr = 2.5e-4
    n_updates = n_interacts*n_epochs*math.ceil(n_process*n_steps/batch_size)
    epsilons = np.linspace(epsilon, 0, num=n_updates, endpoint=False)
    lrs = np.linspace(lr, 0, num=n_updates, endpoint=False)
    global_step = 0
    reward_sum = 0
    envs = SubprocVecEnv([lambda:make_env(i) for i in range(n_process)])
    model = Model(gru_sz).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    storage_sz = (n_steps,n_process)
    storage = {
        'reward':T(np.zeros(storage_sz)),
        'p':T(np.zeros(storage_sz)),
        'value':T(np.zeros(storage_sz)),
        'x':T(np.zeros([*storage_sz,*img_sz])),
        'action':T(np.zeros(storage_sz)),
        'state':T(np.zeros([n_steps+1,n_process, gru_sz]))
    }
    sampler = BatchSampler(SubsetRandomSampler(range(n_process*n_steps)),batch_size,drop_last=False)
    x = T(prepro(envs.reset()))
    with torch.no_grad():
        p, value, state = model(x, storage['state'][0])
    for interact in range(n_interacts):
        storage['state'][0] = storage['state'][-1]
        for step in range(n_steps):
            storage['x'][step] = x
            action = (torch.rand(n_process, device='cuda') > p) + 2
            x, reward, done, info = envs.step(action.cpu().numpy())
            x = T(prepro(x))
            storage['action'][step] = action
            storage['p'][step] = (action==2).float()*p + (action==3).float()*(1-p)
            storage['value'][step] = value
            storage['reward'][step] = T(reward)
            for i in range(n_process):
                if reward[i] != 0:
                    state[i] = T(np.zeros(gru_sz))
            storage['state'][step+1] = state
            with torch.no_grad():
                p, value, state = model(x, state)
        reward_ = discount_r(storage['reward'], value)
        adv = reward_ - storage['value']
        action_, p_old, reward_, adv, state_, x_ = storage['action'].view(-1), storage['p'].view(-1),\
         reward_.view(-1), adv.view(-1), storage['state'].view(-1, gru_sz), storage['x'].view(-1, *img_sz)
        for epoch in range(n_epochs):
            for idxs in sampler:
                epsilon = epsilons[global_step]
                for pg in optimizer.param_groups:
                    pg['lr'] = lrs[global_step]
                p_new, value_new, _ = model(x_[idxs], state_[idxs])
                p_new = (action_[idxs]==2).float()*p_new + (action_[idxs]==3).float()*(1-p_new)
                ratio = p_new/p_old[idxs]
                policy_loss = -torch.sum(torch.min(ratio*adv[idxs],
                 torch.clamp(ratio, 1-epsilon, 1+epsilon)*adv[idxs]))
                value_loss = torch.sum((value_new-reward_[idxs])**2)
                #entropy_loss = -torch.sum(p_log_p(p_new), p_log_p(1-p_new))
                loss = policy_loss+value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                global_step += 1
        reward_sum += storage['reward'].sum().item()
        if (interact+1) % 100 == 0:
            torch.save(model.state_dict(), path/f'3/{interact+1}')
        if (interact+1) % 10 == 0:
            end = time.time()
            print(f'reward_sum:{reward_sum}, time:{start-end}')
            reward_sum = 0
            start = end
            
if __name__ == '__main__':
    main()