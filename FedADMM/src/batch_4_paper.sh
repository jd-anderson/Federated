#!/bin/bash

:: Femnist dataset simulations

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 20 --par_users 10 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset femnist --VR False --model cnn --optimizer FedADMM --gpu cuda:0 &

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 20 --par_users 10 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset femnist --VR False --model cnn --optimizer FedDR --gpu cuda:1 &

:: Synthetic dataset simulations

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 30 --par_users 10 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset synthetic --syn_alpha_beta 0 --VR False --model mlp --optimizer FedADMM --gpu cuda:2

wait

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 30 --par_users 10 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset synthetic --syn_alpha_beta 0.5 --VR False --model mlp --optimizer FedADMM --gpu cuda:0 &

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 30 --par_users 10 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset synthetic --syn_alpha_beta 1 --VR False --model mlp --optimizer FedADMM --gpu cuda:1 &

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 30 --par_users 10 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset synthetic --syn_alpha_beta 0 --VR False --model mlp --optimizer FedDR --gpu cuda:2

wait

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 30 --par_users 10 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset synthetic --syn_alpha_beta 0.5 --VR False --model mlp --optimizer FedDR --gpu cuda:0 &

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 30 --par_users 10 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset synthetic --syn_alpha_beta 1 --VR False --model mlp --optimizer FedDR --gpu cuda:1 &

:: Femnist partial participation performance comparison simulations

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 30 --par_users 5 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset femnist --VR False --model cnn --optimizer FedADMM --gpu cuda:2

wait

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 30 --par_users 10 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset femnist --VR False --model cnn --optimizer FedADMM --gpu cuda:0 &

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 30 --par_users 15 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset femnist --VR False --model cnn --optimizer FedADMM --gpu cuda:1 &

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 30 --par_users 20 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset femnist --VR False --model cnn --optimizer FedADMM --gpu cuda:2

wait

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 30 --par_users 25 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset femnist --VR False --model cnn --optimizer FedADMM --gpu cuda:0 &

python3 ./Fed_main.py --epochs 200 --freq_in 300 --freq_out 1 --num_users 30 --par_users 30 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset femnist --VR False --model cnn --optimizer FedADMM --gpu cuda:1


