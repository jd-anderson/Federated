# FedADMM

This is the repo that could be used to generate the results in the paper 
"FedADMM: A Federated Primal-Dual Algorithm Allowing Partial Participation"
URL: https://arxiv.org/abs/2203.15104

FedADMM Folder:
1. data folder has the Femnist dataset and the links to produce the synthetic dataset.
2. src folder has all the functions required to produce results in the paper.

There is a script file showing the general arguments that are used to produce the results.
Batch script in src folder can be used to generate results in parallel. It assumes you have 3 GPUs which is the hardware we used for our experiments. Change it according to the hardware you are using.

Reference: The Python FedADMM codebase is developed over codebase provided with
FedPD paper "https://arxiv.org/abs/2005.11418"
