import os
import copy
import time
import pickle
import numpy as np
import json
import random
import csv
from tqdm import tqdm
from collections import defaultdict

import torch
from torch import nn

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNFEMnist
from utils import get_dataset, average_weights, exp_details
from agent import Agent


def fed_heuristic(heuristic):
    torch.manual_seed(0)

    class arg:
        def __init__(self):
            self.epochs = 200
            self.freq_in = 300
            self.freq_out = 1
            self.num_users = 3
            self.par_users = 1
            self.local_ep = 300
            self.local_bs = 2
            self.lr = 0.01
            self.mu = 1
            self.dataset = 'femnist'
            self.VR = False
            self.model = 'cnn'
            self.optimizer = 'PSGD'
            self.gpu = True

    args = arg()
    args.gpu = 'cuda:0'
    # This cell is to load FeMNIST data
    path_project = os.path.abspath('..')

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    if args.model == 'cnn':
        if args.dataset == 'femnist':
            global_model = CNNFEMnist(args=args)
    elif args.model == 'mlp':
        global_model = MLP(60, 32, 10)
    else:
        exit('Error: unrecognized model')

    global_model.train()

    agent_list = []
    for i in range(args.num_users):
        agent_list.append(Agent(global_model, args, i, nn.NLLLoss().to(device)))

    # copy weights
    global_weights = global_model.state_dict()

    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    random.seed(10)
    global_dict = {}
    for epoch in tqdm(range(args.epochs)):
        local_weights = []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        m = args.num_users
        if epoch % args.freq_out == 0:
            compute_full = True
            update_model = True
        else:
            compute_full = False
            update_model = False
        global_model.train()

        if epoch == 0:
            for idx in range(args.num_users):
                global_dict[str(idx)] = global_model.state_dict()

        if heuristic:
            if (epoch + 1) % 4 == 0:
                S = ['1']
            else:
                S = random.sample(['0', '2'], 1)
        else:
            S = [str(i) for i in random.sample(range(args.num_users), args.par_users)]

        for idx in S:
            w = agent_list[int(idx)].train_(global_model.state_dict(), args.freq_in, train_dataset, user_groups,
                                            update_model, compute_full)
            local_weights.append(copy.deepcopy(w))
            global_dict[idx] = local_weights[-1]

        if S:
            w_avg = copy.deepcopy(local_weights[0])
            w_avg = {key: w_avg[key] - w_avg[key] for key in w_avg.keys()}
            for key in w_avg.keys():
                for i in range(args.num_users):
                    w_avg[key].add_(global_dict[str(i)][key])
                w_avg[key].div_(args.num_users)

            # update global weights
            global_weights = w_avg
            global_model.load_state_dict(global_weights)

        # Calculate avg training accuracy over all users at every epoch
        global_model.eval()
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        train_accuracy.append(test_acc)
        train_loss.append(test_loss)

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print(f'negative train loss : {200 - train_loss[-1]}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]), flush=True)

    return train_accuracy, train_loss

# Save results to a csv file
# filename = args.optimizer + '_' + args.dataset + '_' + str(args.par_users) + 'of' + str(args.num_users)
# with open(filename+'.csv', 'w') as f:
#     write = csv.writer(f)
#     write.writerow(train_accuracy)
#     write.writerow(train_loss)
