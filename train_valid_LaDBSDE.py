import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random as python_random
import equation as eq
import os
import pandas as pd
import LaDBSDE 

d = 1
T = 1.0
N = 16

example_type = "BlackScholes"
#example_type = "BurgersType"

LR_type = 1

nr_runs = 1
neurons = 10
batch = 128

flag_test = True
#flag_test = False

#flag_verbose = False
flag_verbose = True

if LR_type == 1:    
    net_config = {"L":4, "hidden_neurons": [d+neurons, d+neurons, d+neurons, d+neurons], "k": 30000,
                  "lr_values": [1e-3, 1e-3], "lr_boundaries": [5000],
                  "k_disp": 100, "verbose": flag_verbose, "M_valid": 4096, "M_test": 4096, "batch_size": batch}

elif LR_type == 2:
    net_config = {"L":4, "hidden_neurons": [d+neurons, d+neurons, d+neurons, d+neurons], "k": 60000,
                  "lr_values": [1e-3, 3e-4, 1e-4, 3e-5, 1e-5], "lr_boundaries": [20000, 30000, 40000, 50000],
                  "k_disp": 100, "verbose": flag_verbose, "M_valid": 4096, "M_test": 4096, "batch_size": batch}

eqn_config = {"d":d, "T": T, "N":N}

if example_type == "Simplelinear":
    bsde = eq.Simplelinear(eqn_config)
    bsde_params = "X0_"+str(np.mean(bsde.X0))
    
elif example_type == "BlackScholes":
    bsde = eq.BlackScholes(eqn_config)
    bsde_params = "X0_"+str(np.mean(bsde.X0))+ "_K_" + str(bsde.K) + "_r_" + str(bsde.r) + "_alpha_" + str(bsde.alpha) + "_delta_" + str(bsde.delta) + "_mu_"+str(bsde.mu)+"_sigma_"+str(bsde.sigma)

elif example_type == "BurgersType":
    bsde = eq.BurgersType(eqn_config)
    bsde_params = "X0_"+str(np.mean(bsde.X0))+"_sigma_"+str(bsde.sigma)
    
if flag_test == True:    
    method_path = "C:/Users/amna/Documents/PhD/Papers/3 Third Paper/UQ/LaDBSDE/Testing/Results_test/"
else:
    method_path = "C:/Users/amna/Documents/PhD/Papers/3 Third Paper/UQ/LaDBSDE/Testing/Results_final/"

net_path = method_path + example_type + "/"+ bsde_params + "d_"+str(eqn_config["d"]) + "_T_"+str(eqn_config["T"])+"_N_"+str(eqn_config["N"]) + "/" +  "LR_type_" + str(LR_type) +  "_neurons_" + str(d+neurons) + "_batch_" + str(batch) + "/"

for run in range(nr_runs):
    # Fix seed to get reproducable results
    print("############################# Results for run: %d ##########################"%(run+1))
    np.random.seed(run+1)
    python_random.seed(run+1)
    tf.random.set_seed(run+1)
        
    run_path_r = net_path +  "run_" + str(run+1)+"/"
    isExist = os.path.exists(run_path_r)
    if not isExist:
      # Create a new directory because it does not exist 
        os.makedirs(run_path_r)
        print("The new directory is created:")
        print(run_path_r)
    bsde_solver = LaDBSDE.BSDESolver(eqn_config, net_config, bsde)
    info_model = bsde_solver.train()
    bsde_solver.model.save(run_path_r + "LaDBSDE_model", save_format="tf") #save model
    
    M = len(info_model[0])
    loss_history = np.array(info_model[0]).reshape([M, 1])
    step  = np.reshape(np.arange(M) * net_config["k_disp"], [M, 1])
    Yhat_0_history = np.array(info_model[1]).reshape([M, 1])
    Zhat_0_history = np.array(info_model[2]).reshape([M, d])
    time_history = np.array(info_model[3]).reshape([M, 1])
    Y0 = info_model[4]
    Z0 = info_model[5]
    
    table = {} 
    table_prim = {} 
    for i in range(M):
        table_prim['step'] = step[i][0]
        table_prim['loss'] = loss_history[i][0]
        table_prim['Y0'] = Y0            
        table_prim['Yhat_0'] = Yhat_0_history[i][0]
        table_prim['Z0'] = np.reshape(Z0, (1, d))                        
        table_prim['Zhat_0'] = Zhat_0_history[i, :].reshape(1, d)
        table_prim['time'] = time_history[i][0]

        for k, v in table_prim.items():
            table.setdefault(k, []).append(v)
    pd_table = pd.DataFrame.from_dict(table)
    pd_table.to_csv(run_path_r+'Valid_data.csv', mode='w', header=True)