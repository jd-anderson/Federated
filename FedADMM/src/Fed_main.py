
# This code is developed based on the code base available with FedPD paper at https://github.com/564612540/FedPD

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

torch.manual_seed(0) # Set the seed for the repeatability of the experiments.

args = args_parser() # parsing the arguments passed form the terminal

# This cell is to load FeMNIST data
path_project = os.path.abspath('..')

# Set the GPU according to the input
if args.gpu:
    torch.cuda.set_device(args.gpu)
device = 'cuda' if args.gpu else 'cpu'

# load dataset and user groups
train_dataset, test_dataset, user_groups = get_dataset(args)

# Load the dataset specific model
if args.model == 'cnn':
    if args.dataset == 'femnist':
        global_model = CNNFEMnist(args=args)
    elif args.dataset == 'cifar':
        global_model = CNNCifar(args=args)
elif args.model == 'mlp':
    global_model = MLP(60, 32, 10)
else:
    exit('Error: unrecognized model')

global_model.train()

# Copy the server model to all the clients
agent_list = []
for i in range(args.num_users):
    agent_list.append(Agent(global_model, args, i, nn.NLLLoss().to(device)))

# copy weights
global_weights = global_model.state_dict()

# Declare variables to be used in the code
train_loss, train_accuracy = [], []
val_acc_list, net_list = [], []
cv_loss, cv_acc = [], []
print_every = 5
val_loss_pre, counter = 0, 0

random.seed(10) # setting random seed for repeatability as it would be used in client selection
global_dict = {} # This is the dictionary that keeps track of the model weights of all the clients
for epoch in tqdm(range(args.epochs)):
    local_weights = []
    print(f'\n | Global Training Round : {epoch+1} |\n')

    m = args.num_users
    if epoch % args.freq_out == 0:
        compute_full = True
        update_model = True
    else:
        compute_full = False
        update_model = False
    global_model.train()

# for the first epoch copy global model weights to all the clients.
    if epoch == 0:
        for idx in range(args.num_users):
            global_dict[str(idx)] = global_model.state_dict()
# Select the clients that will be participating in this epoch randomly
    S = [str(i) for i in random.sample(range(args.num_users), args.par_users)]
# Train all the client models and collect their output
    for idx in S:
        w = agent_list[int(idx)].train_(global_model.state_dict(), args.freq_in, train_dataset, user_groups,
                                        update_model, compute_full)
        local_weights.append(copy.deepcopy(w))
        global_dict[idx] = local_weights[-1]
# Averaging all the client model weights
    if S:
        w_avg = copy.deepcopy(local_weights[0])
        w_avg = {key: w_avg[key]-w_avg[key] for key in w_avg.keys()}
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
    if (epoch+1) % print_every == 0:
        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        print(f'Training Loss : {np.mean(np.array(train_loss))}')
        print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]), flush=True)

# Save results to a csv file
filename = args.optimizer + '_' + args.dataset + '_' + 'lr_' + str(args.lr) + '_' + \
           str(args.par_users) + 'of' + str(args.num_users)
if args.dataset == 'synthetic':
    filename = args.optimizer + '_' + args.dataset + '_' + args.syn_alpha_beta + '_' + 'lr_' + str(args.lr) + '_' + \
               str(args.par_users) + 'of' + str(args.num_users)

with open(filename+'.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(train_accuracy)
    write.writerow(train_loss)



