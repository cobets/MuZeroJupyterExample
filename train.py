# Training of neural net

import torch
import torch.nn.functional as F
import numpy as np
import config as cfg

# Generate inputs and targets for training
def gen_target(ep, k):
    # path, reward, observation, action, policy
    turn_idx = np.random.randint(len(ep[0]))
    ps, vs, ax = [], [], []
    for t in range(turn_idx, turn_idx + k + 1):
        if t < len(ep[0]):
            p = ep[4][t]
            a = ep[3][t]
        else:  # state after finishing game
            # p is 0 (loss is 0)
            p = np.zeros_like(ep[4][-1])
            # random action selection
            a = np.zeros(np.prod(ep[3][-1].shape), dtype=np.float32)
            a[np.random.randint(len(a))] = 1
            a = a.reshape(ep[3][-1].shape)
        vs.append([ep[1] if t % 2 == 0 else -ep[1]])
        ps.append(p)
        ax.append(a)

    return ep[2][turn_idx], ax, ps, vs


def train(episodes, net, opt):
    '''Train neural net'''
    p_loss_sum, v_loss_sum = 0, 0
    net.train()
    k = 4
    for _ in range(cfg.num_steps):
        x, ax, p_target, v_target = zip(
            *[gen_target(episodes[np.random.randint(len(episodes))], k) for j in range(cfg.batch_size)])
        x = torch.from_numpy(np.array(x))
        ax = torch.from_numpy(np.array(ax))
        p_target = torch.from_numpy(np.array(p_target))
        v_target = torch.FloatTensor(np.array(v_target))

        # Change the order of axis as [time step, batch, ...]
        ax = torch.transpose(ax, 0, 1)
        p_target = torch.transpose(p_target, 0, 1)
        v_target = torch.transpose(v_target, 0, 1)

        # Compute losses for k (+ current) steps
        p_loss, v_loss = 0, 0
        for t in range(k + 1):
            rp = net.representation(x) if t == 0 else net.dynamics(rp, ax[t - 1])
            p, v = net.prediction(rp)
            p_loss += F.kl_div(torch.log(p), p_target[t], reduction='sum')
            v_loss += torch.sum(((v_target[t] - v) ** 2) / 2)

        p_loss_sum += p_loss.item()
        v_loss_sum += v_loss.item()

        opt.zero_grad()
        (p_loss + v_loss).backward()
        opt.step()

    num_train_datum = cfg.num_steps * cfg.batch_size
    print('p_loss %f v_loss %f' % (p_loss_sum / num_train_datum, v_loss_sum / num_train_datum))
    return net
