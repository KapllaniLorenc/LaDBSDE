import numpy as np
import tensorflow as tf
import random as python_random
import os
import pandas as pd
import DBSDE, LDBSDE, LDBSDE_RNN, LDBSDE_LSTM, LaDBSDE 
import initvar as iv
import sys

Q = 2   
d = 1
T = 0.5
N = 32

method = str(sys.argv[-1]) # "DBSDE", "LDBSDE", "LDBSDE_RNN", "LDBSDE_LSTM", "LaDBSDE"

example_type = "SB"
#example_type = "QZ"
#example_type = "BSBLog"
#example_type = "DIRLog"

#example_type = "BSLog"
#example_type = "DIROneDimLog"

flag_test = True
#flag_test = False

#flag_verbose = False
flag_verbose = True

eqn_config, net_config, bsde, bsde_params = iv.int_var(example_type, flag_verbose, method, d, T, N)            

########################### NOTE: Update based on your directory ####################################
base_path = "C:/Users/amna/Documents/PhD/Dissertation/"

if flag_test == True:    
    method_path = base_path+"Results_test/DeepL/" + method + "/"
else:
    method_path = base_path+"/Results_final/DeepL/" + method + "/"

net_path = method_path + example_type + "/"+ bsde_params + "d_"+str(eqn_config["d"]) + "_T_"+str(eqn_config["T"])+"_N_"+str(eqn_config["N"]) + "/"+ "L_" + str(net_config["L"]) + "_B_" + str(net_config["B"])  + "_eta_" + str(net_config["eta"][0])+"_Kf_" + str(net_config["Kf"])+"/"      
for q in range(Q):
    # Fix seed to get reproducable results
    print("############################# Results for run: %d ##########################"%(q+1))
    np.random.seed(q+1)
    python_random.seed(q+1)
    tf.random.set_seed(q+1)
        
    run_path_r = net_path +  "run_" + str(q+1)+"/"
    isExist = os.path.exists(run_path_r)
    if not isExist:
      # Create a new directory because it does not exist 
        os.makedirs(run_path_r)
        print("The new directory is created:")
        print(run_path_r)
        
    if method == "DBSDE":
        bsde_solver = DBSDE.BSDESolver(eqn_config, net_config, bsde)
        training_history = bsde_solver.train()
        bsde_solver.model.save(run_path_r + "DBSDE_model", save_format="tf") 
    elif method == "LDBSDE":
        bsde_solver = LDBSDE.BSDESolver(eqn_config, net_config, bsde)
        training_history = bsde_solver.train()
        bsde_solver.model.save(run_path_r + "LDBSDE_model", save_format="tf") 
    elif method == "LDBSDE_RNN":
        bsde_solver = LDBSDE_RNN.BSDESolver(eqn_config, net_config, bsde)
        training_history = bsde_solver.train()
        bsde_solver.model.save(run_path_r + "LDBSDE_RNN_model", save_format="tf") 
    elif method == "LDBSDE_LSTM":
        bsde_solver = LDBSDE_LSTM.BSDESolver(eqn_config, net_config, bsde)
        training_history = bsde_solver.train()
        bsde_solver.model.save(run_path_r + "LDBSDE_LSTM_model", save_format="tf") 
    elif method == "LaDBSDE":
        bsde_solver = LaDBSDE.BSDESolver(eqn_config, net_config, bsde)
        training_history = bsde_solver.train()
        bsde_solver.model.save(run_path_r + "LaDBSDE_model", save_format="tf")

    if eqn_config["flag_exact_solution"] == True:
        training_history_df = pd.DataFrame(data=training_history, columns=["kappa", "L_train", "L_valid", "epsy_0", "epsz_0", "tau"])       
    else:
        training_history_df = pd.DataFrame(data=training_history, columns=["kappa", "L_train", "L_valid", "epsy_0", "tau"])       
        
    with open(run_path_r+'Train_Valid_data'+'.csv', 'w') as f:
        training_history_df.to_csv(f, index=False, line_terminator='\n')