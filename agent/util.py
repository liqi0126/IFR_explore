import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# Hyperparameters
h_out_dim = 32


def sample_action(prob, adj, pred_thres, uncertainty_thres, policy, early_stop):
    if policy == 'no_interaction':
        a = len(prob) - 1
    elif policy == 'random':
        if not early_stop:
            prob = prob[:-1]
        a = np.random.randint(len(prob))
    elif policy == 'most_promising':
        adj[adj == 1] = -1
        adj_max = adj.max(-1)[0]
        idx = adj_max.argmax()
        if adj_max[idx] > pred_thres or not early_stop:
            a = idx
        else:
            a = len(prob) - 1
    elif policy == 'most_uncertain':
        adj = adj.numpy()
        uncertainty = np.min([adj, 1 - adj], 0).max(-1)
        if uncertainty.max() > uncertainty_thres or not early_stop:
            a = np.argmax(uncertainty)
        else:
            a = len(prob) - 1
    elif policy == 'rl':
        if not early_stop:
            prob = prob[:-1]
        m = Categorical(prob)
        a = m.sample().item()
    else:
        raise NotImplementedError
    return a


def explore(env, rl, pred_thres=0, uncertainty_thres=0):
    def exploration_action(prob, adj, pred_thres, uncertainty_thres, early_stop, env):
        ran = random.random()
        if ran < rl.epsilon:
            a = sample_action(prob, adj, pred_thres, uncertainty_thres, 'random', early_stop)
        else:
            a = sample_action(prob, adj, pred_thres, uncertainty_thres, 'rl', early_stop)
        return a

    h_out = (torch.zeros([1, 1, h_out_dim], dtype=torch.float).cuda(), torch.zeros([1, 1, h_out_dim], dtype=torch.float).cuda())
    s, pn_feats, obj_pos, obj_rot, obj_size, obj_attr, adj = env.reset()
    done = False

    data = []
    total_r = 0
    while not done:
        h_in = h_out
        _, h_out, logits = rl.pi(pn_feats, obj_pos, obj_rot, obj_size, obj_attr, adj, h_in)
        logits = logits.squeeze()
        logits[:len(env.intervened)][env.intervened] = -100
        prob = F.softmax(logits, dim=-1)

        a = exploration_action(prob, adj, pred_thres, uncertainty_thres, True, env)
        s_prime, r, done, info = env.step(a)
        idx = 0
        while info['repeat']:
            a = exploration_action(prob, adj, pred_thres, uncertainty_thres, True, env)
            s_prime, r, done, info = env.step(a)
            idx += 1
            if idx > 20:
                a = len(prob) - 1
                break

        total_r += r
        adj_prime = info['graph']

        data.append((pn_feats, obj_pos, obj_rot, obj_size, obj_attr, adj, adj_prime, s, a, r, s_prime, prob[a].item(), (h_in[0].detach(), h_in[1].detach()), (h_out[0].detach(), h_out[1].detach()), done))
        adj = adj_prime
        s = s_prime

        if done:
            rl.batch_data.append(data)
            break

    rl.train_net()
    return total_r, info['precision'], info["recall"], info["f1"], info["step"]
