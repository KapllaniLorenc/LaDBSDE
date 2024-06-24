import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random as python_random
import initvar as iv
import os
import sys
params = {'legend.fontsize': 22,
          'figure.figsize': (12, 10),
         'axes.labelsize': 22,
         'axes.titlesize':22,
         'xtick.labelsize':22,
         'ytick.labelsize':22}
plt.rcParams.update(params)

Q = 2   
d = 1
T = 0.5
N = 32

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

eqn_config, net_config_DBSDE, bsde, bsde_params = iv.int_var(example_type, flag_verbose, "DBSDE", d, T, N)
_, net_config_other, _, _ = iv.int_var(example_type, flag_verbose, "LDBSDE", d, T, N)

# Update based on your files
base_path = "C:/Users/amna/Documents/PhD/Dissertation/"

if flag_test == True:    
    DBSDE_path = base_path+"Results_test/DeepL/DBSDE/"
    LDBSDE_path = base_path+"Results_test/DeepL/LDBSDE/"
    LaDBSDE_path = base_path+"Results_test/DeepL/LaDBSDE/"
    if example_type == "SB" and d == 1:
        LDBSDE_RNN_path = base_path+"Results_test/DeepL/LDBSDE_RNN/"
        LDBSDE_LSTM_path = base_path+"Results_test/DeepL/LDBSDE_LSTM/"
        
else:
    DBSDE_path = base_path+"Results_final/DeepL/DBSDE/"
    LDBSDE_path = base_path+"Results_final/DeepL/LDBSDE/"
    LaDBSDE_path = base_path+"Results_final/DeepL/LaDBSDE/"
    if example_type == "SB" and d == 1:
        LDBSDE_RNN_path = base_path+"Results_final/DeepL/LDBSDE_RNN/"
        LDBSDE_LSTM_path = base_path+"Results_final/DeepL/LDBSDE_LSTM/"

DBSDE_net_path = DBSDE_path + example_type + "/"+ bsde_params + "d_"+str(eqn_config["d"]) + "_T_"+str(eqn_config["T"])+"_N_"+str(eqn_config["N"]) + "/"+ "L_" + str(net_config_DBSDE["L"]) + "_B_" + str(net_config_DBSDE["B"])  + "_eta_" + str(net_config_DBSDE["eta"][0])+"_Kf_" + str(net_config_DBSDE["Kf"])+"/"      

LDBSDE_net_path = LDBSDE_path + example_type + "/"+ bsde_params + "d_"+str(eqn_config["d"]) + "_T_"+str(eqn_config["T"])+"_N_"+str(eqn_config["N"]) + "/"+ "L_" + str(net_config_other["L"]) + "_B_" + str(net_config_other["B"])  + "_eta_" + str(net_config_other["eta"][0])+"_Kf_" + str(net_config_other["Kf"])+"/"      

LaDBSDE_net_path = LaDBSDE_path + example_type + "/"+ bsde_params + "d_"+str(eqn_config["d"]) + "_T_"+str(eqn_config["T"])+"_N_"+str(eqn_config["N"]) + "/"+ "L_" + str(net_config_other["L"]) + "_B_" + str(net_config_other["B"])  + "_eta_" + str(net_config_other["eta"][0])+"_Kf_" + str(net_config_other["Kf"])+"/"      

if example_type == "SB" and d == 1:
    LDBSDE_RNN_net_path = LDBSDE_RNN_path + example_type + "/"+ bsde_params + "d_"+str(eqn_config["d"]) + "_T_"+str(eqn_config["T"])+"_N_"+str(eqn_config["N"]) + "/"+ "L_" + str(net_config_other["L"]) + "_B_" + str(net_config_other["B"])  + "_eta_" + str(net_config_other["eta"][0])+"_Kf_" + str(net_config_other["Kf"])+"/"      
    LDBSDE_LSTM_net_path = LDBSDE_LSTM_path + example_type + "/"+ bsde_params + "d_"+str(eqn_config["d"]) + "_T_"+str(eqn_config["T"])+"_N_"+str(eqn_config["N"]) + "/"+ "L_" + str(net_config_other["L"]) + "_B_" + str(net_config_other["B"])  + "_eta_" + str(net_config_other["eta"][0])+"_Kf_" + str(net_config_other["Kf"])+"/"      

    
def epsy(y_n, yhat_n): # y_n E R^B, yhat_n E R^B
    return np.mean(np.power(y_n-yhat_n, 2))

def epsy_r(y_n, yhat_n): # y_n E R^B, yhat_n E R^B
    return np.mean(np.power(y_n-yhat_n, 2)/np.power(y_n, 2))

def epsz(z_n, zhat_n): # y_n E R^(Bxd), zhat_n E R^(Bxd)
    return np.mean( np.sum(np.power(z_n-zhat_n, 2), axis = 1))

def epsz_r(z_n, zhat_n): # y_n E R^(Bxd), zhat_n E R^(Bxd)
    return np.mean( np.sum(np.power(z_n-zhat_n, 2), axis = 1)/np.sum(np.power(z_n, 2), axis = 1))

def moments_f(vec, ax):
    return np.mean(vec, axis=ax),  np.std(vec, axis=ax)

def save_dat(path, name, x, y_DBSDE_mean, y_DBSDE_std, y_LDBSDE_mean, y_LDBSDE_std, y_LaDBSDE_mean, y_LaDBSDE_std, format_x, format_y):
    M = np.shape(x)[0]
    np.savetxt(path+name+".dat", np.concatenate([x.reshape(M,1), y_DBSDE_mean.reshape(M,1), (y_DBSDE_mean + y_DBSDE_std).reshape(M,1), np.maximum(y_DBSDE_mean - y_DBSDE_std, 1e-16).reshape(M,1), y_LDBSDE_mean.reshape(M,1), (y_LDBSDE_mean + y_LDBSDE_std).reshape(M,1), np.maximum(y_LDBSDE_mean - y_LDBSDE_std, 1e-16).reshape(M,1), y_LaDBSDE_mean.reshape(M,1), (y_LaDBSDE_mean + y_LaDBSDE_std).reshape(M,1), np.maximum(y_LaDBSDE_mean - y_LaDBSDE_std, 1e-16).reshape(M,1)], axis = 1), fmt=[format_x]+[format_y]*9)

def plot_fig(path, name, x, y_DBSDE_mean, y_DBSDE_std, y_LDBSDE_mean, y_LDBSDE_std, y_LaDBSDE_mean, y_LaDBSDE_std, xlab, ylab, leg1, leg2, leg3):
    fig = plt.figure()
    plt.semilogy(x, y_DBSDE_mean, c = 'b', linestyle = '--')
    plt.semilogy(x, y_LDBSDE_mean, c = 'orange', linestyle = ':')
    plt.semilogy(x, y_LaDBSDE_mean, c = 'g', linestyle = '-')
    plt.fill_between(x, y_DBSDE_mean-y_DBSDE_std, y_DBSDE_mean+y_DBSDE_std, alpha = 0.5, color = 'b')
    plt.fill_between(x, y_LDBSDE_mean-y_LDBSDE_std, y_LDBSDE_mean+y_LDBSDE_std, alpha = 0.5, color = 'orange')
    plt.fill_between(x, y_LaDBSDE_mean-y_LaDBSDE_std, y_LaDBSDE_mean+y_LaDBSDE_std, alpha = 0.5, color = 'g')
    plt.xlabel(xlab)        
    plt.ylabel(ylab)
    plt.legend([leg1, leg2, leg3], loc = 'best', prop={'size': 16})
    fig.savefig(path+name+'.png')
    #plt.show()
    
    
if eqn_config["flag_exact_solution"] == True:
    epsy_test_DBSDE = np.zeros([N, Q])
    epsz_test_DBSDE = np.zeros([N, Q])
    epsy_test_LDBSDE = np.zeros([N, Q])
    epsz_test_LDBSDE = np.zeros([N, Q])
    epsy_test_LaDBSDE = np.zeros([N, Q])
    epsz_test_LaDBSDE = np.zeros([N, Q])
            
    epsy_r_0_DBSDE = np.zeros([Q,])
    epsz_r_0_DBSDE = np.zeros([Q,])
    epsy_r_0_LDBSDE = np.zeros([Q,])
    epsz_r_0_LDBSDE = np.zeros([Q,])
    epsy_r_0_LaDBSDE = np.zeros([Q,])
    epsz_r_0_LaDBSDE = np.zeros([Q,])

    if example_type == "SB" and d == 1:
        epsy_r_0_LDBSDE_RNN = np.zeros([Q,])
        epsz_r_0_LDBSDE_RNN = np.zeros([Q,])
        tau_LDBSDE_RNN = np.zeros([Q,])

        epsy_r_0_LDBSDE_LSTM = np.zeros([Q,])
        epsz_r_0_LDBSDE_LSTM = np.zeros([Q,])
        tau_LDBSDE_LSTM = np.zeros([Q,])
        
    kk = int(net_config_other["Kf"]/net_config_other["k_disp"])+1
    L_valid_DBSDE = np.zeros([kk,Q])
    epsy_0_valid_DBSDE = np.zeros([kk,Q])
    epsz_0_valid_DBSDE = np.zeros([kk,Q])
    tau_DBSDE = np.zeros([Q,])

    L_valid_LDBSDE = np.zeros([kk,Q])
    epsy_0_valid_LDBSDE = np.zeros([kk,Q])
    epsz_0_valid_LDBSDE = np.zeros([kk,Q])
    tau_LDBSDE = np.zeros([Q,])

    L_valid_LaDBSDE = np.zeros([kk,Q])
    epsy_0_valid_LaDBSDE = np.zeros([kk,Q])
    epsz_0_valid_LaDBSDE = np.zeros([kk,Q])
    tau_LaDBSDE = np.zeros([Q,])
    
    for q in range(Q):      
        print("############################# Results for run: %d ##########################"%(q+1))    
        # Fix seed to get reproducable results
        np.random.seed(q+1)
        python_random.seed(q+1)
        tf.random.set_seed(q+1)
        DBSDE_run_path_r = DBSDE_net_path +  "run_" + str(q+1)+"/"
        LDBSDE_run_path_r = LDBSDE_net_path +  "run_" + str(q+1)+"/"
        LaDBSDE_run_path_r = LaDBSDE_net_path +  "run_" + str(q+1)+"/"
        
        if example_type == "SB" and d == 1:
            LDBSDE_RNN_run_path_r = LDBSDE_RNN_net_path +  "run_" + str(q+1)+"/"
            LDBSDE_LSTM_run_path_r = LDBSDE_LSTM_net_path +  "run_" + str(q+1)+"/"
        
        dW_test, X_test, exp_X_test = bsde.sample(net_config_other["B_test"])

        model_DBSDE = tf.keras.models.load_model(DBSDE_run_path_r+ "DBSDE_model", compile=False)    
        Yhat_test_DBSDE, Zhat_test_DBSDE = model_DBSDE((X_test, dW_test), training=False)

        model_LDBSDE = tf.keras.models.load_model(LDBSDE_run_path_r+ "LDBSDE_model", compile=False)    
        Yhat_test_LDBSDE, Zhat_test_LDBSDE = model_LDBSDE((bsde.t, X_test, dW_test))

        model_LaDBSDE = tf.keras.models.load_model(LaDBSDE_run_path_r+ "LaDBSDE_model", compile=False)    
        Yhat_test_LaDBSDE, Zhat_test_LaDBSDE = model_LaDBSDE((bsde.t, X_test, dW_test))
        
        if example_type == "SB" and d == 1:
            model_LDBSDE_RNN = tf.keras.models.load_model(LDBSDE_RNN_run_path_r+ "LDBSDE_RNN_model", compile=False)    
            Yhat_test_LDBSDE_RNN, Zhat_test_LDBSDE_RNN = model_LDBSDE_RNN((bsde.t, X_test, dW_test))

            model_LDBSDE_LSTM = tf.keras.models.load_model(LDBSDE_LSTM_run_path_r+ "LDBSDE_LSTM_model", compile=False)    
            Yhat_test_LDBSDE_LSTM, Zhat_test_LDBSDE_LSTM = model_LDBSDE_LSTM((bsde.t, X_test, dW_test))
            
        pd_data_DBSDE = pd.read_csv(DBSDE_run_path_r+'Train_Valid_data.csv')  
        pd_data_LDBSDE = pd.read_csv(LDBSDE_run_path_r+'Train_Valid_data.csv')  
        pd_data_LaDBSDE = pd.read_csv(LaDBSDE_run_path_r+'Train_Valid_data.csv')  
        
        data_DBSDE = pd_data_DBSDE.to_numpy()
        data_LDBSDE = pd_data_LDBSDE.to_numpy()
        data_LaDBSDE = pd_data_LaDBSDE.to_numpy()
        
        kappa = data_DBSDE[:, 0]        
        L_valid_DBSDE[:, q] = data_DBSDE[:, 2]        
        epsy_0_valid_DBSDE[:, q] = data_DBSDE[:, 3]
        epsz_0_valid_DBSDE[:, q] = data_DBSDE[:, 4]
        tau_DBSDE[q] = data_DBSDE[-1, -1]

        L_valid_LDBSDE[:, q] = data_LDBSDE[:, 2]        
        epsy_0_valid_LDBSDE[:, q] = data_LDBSDE[:, 3]
        epsz_0_valid_LDBSDE[:, q] = data_LDBSDE[:, 4]
        tau_LDBSDE[q] = data_LDBSDE[-1, -1]        
        
        L_valid_LaDBSDE[:, q] = data_LaDBSDE[:, 2]        
        epsy_0_valid_LaDBSDE[:, q] = data_LaDBSDE[:, 3]
        epsz_0_valid_LaDBSDE[:, q] = data_LaDBSDE[:, 4]
        tau_LaDBSDE[q] = data_LaDBSDE[-1, -1] 
        
        if example_type == "SB" and d == 1:
            pd_data_LDBSDE_RNN = pd.read_csv(LDBSDE_RNN_run_path_r+'Train_Valid_data.csv')  
            pd_data_LDBSDE_LSTM = pd.read_csv(LDBSDE_LSTM_run_path_r+'Train_Valid_data.csv')  
            
            data_LDBSDE_RNN = pd_data_LDBSDE_RNN.to_numpy()
            data_LDBSDE_LSTM = pd_data_LDBSDE_LSTM.to_numpy()
        
            tau_LDBSDE_RNN[q] = data_LDBSDE_RNN[-1, -1]        
            tau_LDBSDE_LSTM[q] = data_LDBSDE_LSTM[-1, -1]        
        
        for n in range(0, N):
            Y_n_test = bsde.Y_tf(bsde.t[n], exp_X_test[:, :, n]).numpy()
            Z_n_test = bsde.Z_tf(bsde.t[n], exp_X_test[:, :, n]).numpy()                          
            
            # Calculation on testing data
            epsy_test_DBSDE[n, q] = epsy(Y_n_test, Yhat_test_DBSDE[n])
            epsz_test_DBSDE[n, q] = epsz(Z_n_test, Zhat_test_DBSDE[n])

            epsy_test_LDBSDE[n, q] = epsy(Y_n_test, Yhat_test_LDBSDE[n])
            epsz_test_LDBSDE[n, q] = epsz(Z_n_test, Zhat_test_LDBSDE[n])

            epsy_test_LaDBSDE[n, q] = epsy(Y_n_test, Yhat_test_LaDBSDE[n])
            epsz_test_LaDBSDE[n, q] = epsz(Z_n_test, Zhat_test_LaDBSDE[n])
            
            if n == 0:
                epsy_r_0_DBSDE[q] = epsy_r(Y_n_test, Yhat_test_DBSDE[n])
                epsy_r_0_LDBSDE[q] = epsy_r(Y_n_test, Yhat_test_LDBSDE[n])
                epsy_r_0_LaDBSDE[q] = epsy_r(Y_n_test, Yhat_test_LaDBSDE[n])
                
                if example_type == "QZ": # Z_0 = np.zeros(1, d), take absolute error                    
                    epsz_r_0_DBSDE[q] = epsz_test_DBSDE[n, q]
                    epsz_r_0_LDBSDE[q] = epsz_test_LDBSDE[n, q]
                    epsz_r_0_LaDBSDE[q] = epsz_test_LaDBSDE[n, q]
                else:
                    epsz_r_0_DBSDE[q] = epsz_r(Z_n_test, Zhat_test_DBSDE[n])
                    epsz_r_0_LDBSDE[q] = epsz_r(Z_n_test, Zhat_test_LDBSDE[n])
                    epsz_r_0_LaDBSDE[q] = epsz_r(Z_n_test, Zhat_test_LaDBSDE[n])
                    
                if example_type == "SB" and d == 1:
                    epsy_r_0_LDBSDE_RNN[q] = epsy_r(Y_n_test, Yhat_test_LDBSDE_RNN[n])
                    epsy_r_0_LDBSDE_LSTM[q] = epsy_r(Y_n_test, Yhat_test_LDBSDE_LSTM[n])

                    epsz_r_0_LDBSDE_RNN[q] = epsz_r(Z_n_test, Zhat_test_LDBSDE_RNN[n])
                    epsz_r_0_LDBSDE_LSTM[q] = epsz_r(Z_n_test, Zhat_test_LDBSDE_LSTM[n])                    

                    
    # Moments of performance metrics
    epsy_test_DBSDE_mean, epsy_test_DBSDE_std = moments_f(epsy_test_DBSDE, 1)
    epsz_test_DBSDE_mean, epsz_test_DBSDE_std = moments_f(epsz_test_DBSDE, 1)

    epsy_test_LDBSDE_mean, epsy_test_LDBSDE_std = moments_f(epsy_test_LDBSDE, 1)
    epsz_test_LDBSDE_mean, epsz_test_LDBSDE_std = moments_f(epsz_test_LDBSDE, 1)

    epsy_test_LaDBSDE_mean, epsy_test_LaDBSDE_std = moments_f(epsy_test_LaDBSDE, 1)
    epsz_test_LaDBSDE_mean, epsz_test_LaDBSDE_std = moments_f(epsz_test_LaDBSDE, 1)
    
    save_dat(LaDBSDE_net_path, "vareps_Y", bsde.t[0:-1], epsy_test_DBSDE_mean, epsy_test_DBSDE_std, epsy_test_LDBSDE_mean, epsy_test_LDBSDE_std, epsy_test_LaDBSDE_mean, epsy_test_LaDBSDE_std, '%.4f', '%.8e')
    
    save_dat(LaDBSDE_net_path, "vareps_Z", bsde.t[0:-1], epsz_test_DBSDE_mean, epsz_test_DBSDE_std, epsz_test_LDBSDE_mean, epsz_test_LDBSDE_std, epsz_test_LaDBSDE_mean, epsz_test_LaDBSDE_std, '%.4f', '%.8e')

    # Error at each discrete time
    plot_fig(LaDBSDE_net_path, "vareps_Y", bsde.t[0:-1], epsy_test_DBSDE_mean, epsy_test_DBSDE_std, epsy_test_LDBSDE_mean, epsy_test_LDBSDE_std, epsy_test_LaDBSDE_mean, epsy_test_LaDBSDE_std, r'$t_n$', r'$\tilde{\varepsilon}^{y}_n$', r'$\tilde{\varepsilon}^{y}_n$' + "DBSDE", r'$\tilde{\varepsilon}^{y}_n$' + "LDBSDE", r'$\tilde{\varepsilon}^{y}_n$' + "LaDBSDE")

    plot_fig(LaDBSDE_net_path, "vareps_Z", bsde.t[0:-1], epsz_test_DBSDE_mean, epsz_test_DBSDE_std, epsz_test_LDBSDE_mean, epsz_test_LDBSDE_std, epsz_test_LaDBSDE_mean, epsz_test_LaDBSDE_std, r'$t_n$', r'$\tilde{\varepsilon}^{z}_n$', r'$\tilde{\varepsilon}^{z}_n$' + "DBSDE", r'$\tilde{\varepsilon}^{z}_n$' + "LDBSDE", r'$\tilde{\varepsilon}^{z}_n$' + "LaDBSDE")
    
    epsy_r_0_DBSDE_mean, epsy_r_0_DBSDE_std = moments_f(epsy_r_0_DBSDE, 0)
    epsz_r_0_DBSDE_mean, epsz_r_0_DBSDE_std = moments_f(epsz_r_0_DBSDE, 0)
    
    epsy_r_0_LDBSDE_mean, epsy_r_0_LDBSDE_std = moments_f(epsy_r_0_LDBSDE, 0)
    epsz_r_0_LDBSDE_mean, epsz_r_0_LDBSDE_std = moments_f(epsz_r_0_LDBSDE, 0)

    epsy_r_0_LaDBSDE_mean, epsy_r_0_LaDBSDE_std = moments_f(epsy_r_0_LaDBSDE, 0)
    epsz_r_0_LaDBSDE_mean, epsz_r_0_LaDBSDE_std = moments_f(epsz_r_0_LaDBSDE, 0)
    
    if example_type == "SB" and d == 1:

        epsy_r_0_LDBSDE_RNN_mean, epsy_r_0_LDBSDE_RNN_std = moments_f(epsy_r_0_LDBSDE_RNN, 0)
        epsz_r_0_LDBSDE_RNN_mean, epsz_r_0_LDBSDE_RNN_std = moments_f(epsz_r_0_LDBSDE_RNN, 0)

        epsy_r_0_LDBSDE_LSTM_mean, epsy_r_0_LDBSDE_LSTM_std = moments_f(epsy_r_0_LDBSDE_LSTM, 0)
        epsz_r_0_LDBSDE_LSTM_mean, epsz_r_0_LDBSDE_LSTM_std = moments_f(epsz_r_0_LDBSDE_LSTM, 0)
        
        pd_eps_r_mom_t0 = pd.DataFrame(
            {'Scheme': ["DBSDE", "LDBSDE", "LDBSDE_RNN", "LDBSDE_LSTM", "LaDBSDE"],             
             'vareps_y_r_0_mean': [epsy_r_0_DBSDE_mean, epsy_r_0_LDBSDE_mean, epsy_r_0_LDBSDE_RNN_mean, epsy_r_0_LDBSDE_LSTM_mean, epsy_r_0_LaDBSDE_mean],
             'vareps_y_r_0_std': [epsy_r_0_DBSDE_std, epsy_r_0_LDBSDE_std, epsy_r_0_LDBSDE_RNN_std, epsy_r_0_LDBSDE_LSTM_std, epsy_r_0_LaDBSDE_std],
             'vareps_z_r_0_mean': [epsz_r_0_DBSDE_mean, epsz_r_0_LDBSDE_mean, epsz_r_0_LDBSDE_RNN_mean, epsz_r_0_LDBSDE_LSTM_mean, epsz_r_0_LaDBSDE_mean],
             'vareps_z_r_0_std': [epsz_r_0_DBSDE_std, epsz_r_0_LDBSDE_std, epsz_r_0_LDBSDE_RNN_std, epsz_r_0_LDBSDE_LSTM_std, epsz_r_0_LaDBSDE_std]
            })
        
    else:        
        pd_eps_r_mom_t0 = pd.DataFrame(
            {'Scheme': ["DBSDE", "LDBSDE", "LaDBSDE"],
             'vareps_y_r_0_mean': [epsy_r_0_DBSDE_mean, epsy_r_0_LDBSDE_mean, epsy_r_0_LaDBSDE_mean],
             'vareps_y_r_0_std': [epsy_r_0_DBSDE_std, epsy_r_0_LDBSDE_std, epsy_r_0_LaDBSDE_std],
             'vareps_z_r_0_mean': [epsz_r_0_DBSDE_mean, epsz_r_0_LDBSDE_mean, epsz_r_0_LaDBSDE_mean],
             'vareps_z_r_0_std': [epsz_r_0_DBSDE_std, epsz_r_0_LDBSDE_std, epsz_r_0_LaDBSDE_std]
            })

    pd_eps_r_mom_t0.to_csv(LaDBSDE_net_path+"moments_vareps_r_t0.csv", float_format='%.16f', index=False)

    L_valid_DBSDE_mean, L_valid_DBSDE_std = moments_f(L_valid_DBSDE, 1)
    epsy_0_valid_DBSDE_mean, epsy_0_valid_DBSDE_std =  moments_f(epsy_0_valid_DBSDE, 1)
    epsz_0_valid_DBSDE_mean, epsz_0_valid_DBSDE_std =  moments_f(epsz_0_valid_DBSDE, 1)
    tau_DBSDE_mean, tau_DBSDE_std = moments_f(tau_DBSDE, 0)
    
    L_valid_LDBSDE_mean, L_valid_LDBSDE_std = moments_f(L_valid_LDBSDE, 1)
    epsy_0_valid_LDBSDE_mean, epsy_0_valid_LDBSDE_std =  moments_f(epsy_0_valid_LDBSDE, 1)
    epsz_0_valid_LDBSDE_mean, epsz_0_valid_LDBSDE_std =  moments_f(epsz_0_valid_LDBSDE, 1)
    tau_LDBSDE_mean, tau_LDBSDE_std = moments_f(tau_LDBSDE, 0)

    L_valid_LaDBSDE_mean, L_valid_LaDBSDE_std = moments_f(L_valid_LaDBSDE, 1)
    epsy_0_valid_LaDBSDE_mean, epsy_0_valid_LaDBSDE_std =  moments_f(epsy_0_valid_LaDBSDE, 1)
    epsz_0_valid_LaDBSDE_mean, epsz_0_valid_LaDBSDE_std =  moments_f(epsz_0_valid_LaDBSDE, 1)
    tau_LaDBSDE_mean, tau_LaDBSDE_std = moments_f(tau_LaDBSDE, 0)
    
    save_dat(LaDBSDE_net_path, "L_valid", kappa, L_valid_DBSDE_mean, L_valid_DBSDE_std, L_valid_LDBSDE_mean, L_valid_LDBSDE_std, L_valid_LaDBSDE_mean, L_valid_LaDBSDE_std, '%d', '%.8e')
    
    save_dat(LaDBSDE_net_path, "vareps_Y_0_valid", kappa, epsy_0_valid_DBSDE_mean, epsy_0_valid_DBSDE_std, epsy_0_valid_LDBSDE_mean, epsy_0_valid_LDBSDE_std, epsy_0_valid_LaDBSDE_mean, epsy_0_valid_LaDBSDE_std, '%d', '%.8e')
    save_dat(LaDBSDE_net_path, "vareps_Z_0_valid", kappa, epsz_0_valid_DBSDE_mean, epsz_0_valid_DBSDE_std, epsz_0_valid_LDBSDE_mean, epsz_0_valid_LDBSDE_std, epsz_0_valid_LaDBSDE_mean, epsz_0_valid_LaDBSDE_std, '%d', '%.8e')
        
    plot_fig(LaDBSDE_net_path, "L_valid", kappa, L_valid_DBSDE_mean, L_valid_DBSDE_std, L_valid_LDBSDE_mean, L_valid_LDBSDE_std, L_valid_LaDBSDE_mean, L_valid_LaDBSDE_std, r'$\kappa$', r'$\tilde{\mathbf{L}}$', r'$\tilde{\mathbf{L}}$' + "DBSDE", r'$\tilde{\mathbf{L}}$' + "LDBSDE", r'$\tilde{\mathbf{L}}$' + "LaDBSDE")
    
    plot_fig(LaDBSDE_net_path, "vareps_Y_0_valid", kappa, epsy_0_valid_DBSDE_mean, epsy_0_valid_DBSDE_std, epsy_0_valid_LDBSDE_mean, epsy_0_valid_LDBSDE_std, epsy_0_valid_LaDBSDE_mean, epsy_0_valid_LaDBSDE_std, r'$\kappa$', r'$\tilde{\varepsilon}^{y}_0$', r'$\tilde{\varepsilon}^{y}_0$' + "DBSDE", r'$\tilde{\varepsilon}^{y}_0$' + "LDBSDE", r'$\tilde{\varepsilon}^{y}_0$' + "LaDBSDE")
    plot_fig(LaDBSDE_net_path, "vareps_Z_0_valid", kappa, epsz_0_valid_DBSDE_mean, epsz_0_valid_DBSDE_std, epsz_0_valid_LDBSDE_mean, epsz_0_valid_LDBSDE_std, epsz_0_valid_LaDBSDE_mean, epsz_0_valid_LaDBSDE_std, r'$\kappa$', r'$\tilde{\varepsilon}^{z}_0$', r'$\tilde{\varepsilon}^{z}_0$' + "DBSDE", r'$\tilde{\varepsilon}^{z}_0$' + "LDBSDE", r'$\tilde{\varepsilon}^{z}_0$' + "LaDBSDE")
    
    if example_type == "SB" and d == 1:
        tau_LDBSDE_RNN_mean, tau_LDBSDE_RNN_std = moments_f(tau_LDBSDE_RNN, 0)
        tau_LDBSDE_LSTM_mean, tau_LDBSDE_LSTM_std = moments_f(tau_LDBSDE_LSTM, 0)
        
        pd_mom_comput_time = pd.DataFrame(
        {'Scheme': ["DBSDE", "LDBSDE", "LDBSDE_RNN", "LDBSDE_LSTM", "LaDBSDE"],
         'tau_mean': [tau_DBSDE_mean, tau_LDBSDE_mean, tau_LDBSDE_RNN_mean, tau_LDBSDE_LSTM_mean, tau_LaDBSDE_mean],
         'tau_std': [tau_DBSDE_std, tau_LDBSDE_std, tau_LDBSDE_RNN_std, tau_LDBSDE_LSTM_std, tau_LaDBSDE_std]
        })
        
    else:
        pd_mom_comput_time = pd.DataFrame(
        {'Scheme': ["DBSDE", "LDBSDE", "LaDBSDE"],
         'tau_mean': [tau_DBSDE_mean, tau_LDBSDE_mean, tau_LaDBSDE_mean],
         'tau_std': [tau_DBSDE_std, tau_LDBSDE_std, tau_LaDBSDE_std]     
        })

    pd_mom_comput_time.to_csv(LaDBSDE_net_path+"moments_comput_time.csv", float_format='%.16f', index=False)   
    
else:
    Yhat_0_DBSDE_arr = np.zeros([Q,])
    Yhat_0_LDBSDE_arr = np.zeros([Q,])
    Yhat_0_LaDBSDE_arr = np.zeros([Q,])
    
    epsy_r_0_DBSDE = np.zeros([Q,])
    epsy_r_0_LDBSDE = np.zeros([Q,])
    epsy_r_0_LaDBSDE = np.zeros([Q,])
    
    kk = int(net_config_other["Kf"]/net_config_other["k_disp"])+1
    L_valid_DBSDE = np.zeros([kk,Q])
    epsy_0_valid_DBSDE = np.zeros([kk,Q])
    tau_DBSDE = np.zeros([Q,])

    L_valid_LDBSDE = np.zeros([kk,Q])
    epsy_0_valid_LDBSDE = np.zeros([kk,Q])
    tau_LDBSDE = np.zeros([Q,])

    L_valid_LaDBSDE = np.zeros([kk,Q])
    epsy_0_valid_LaDBSDE = np.zeros([kk,Q])
    tau_LaDBSDE = np.zeros([Q,])

    for q in range(Q):      
        print("############################# Results for run: %d ##########################"%(q+1))    
        # Fix seed to get reproducable results
        np.random.seed(q+1)
        python_random.seed(q+1)
        tf.random.set_seed(q+1)
        DBSDE_run_path_r = DBSDE_net_path +  "run_" + str(q+1)+"/"
        LDBSDE_run_path_r = LDBSDE_net_path +  "run_" + str(q+1)+"/"
        LaDBSDE_run_path_r = LaDBSDE_net_path +  "run_" + str(q+1)+"/"

        dW_test, X_test, exp_X_test = bsde.sample(net_config_other["B_test"])

        model_DBSDE = tf.keras.models.load_model(DBSDE_run_path_r+ "DBSDE_model", compile=False)    
        Yhat_test_DBSDE, Zhat_test_DBSDE = model_DBSDE((X_test, dW_test), training=False)

        model_LDBSDE = tf.keras.models.load_model(LDBSDE_run_path_r+ "LDBSDE_model", compile=False)    
        Yhat_test_LDBSDE, Zhat_test_LDBSDE = model_LDBSDE((bsde.t, X_test, dW_test))

        model_LaDBSDE = tf.keras.models.load_model(LaDBSDE_run_path_r+ "LaDBSDE_model", compile=False)    
        Yhat_test_LaDBSDE, Zhat_test_LaDBSDE = model_LaDBSDE((bsde.t, X_test, dW_test))

        pd_data_DBSDE = pd.read_csv(DBSDE_run_path_r+'Train_Valid_data.csv')  
        pd_data_LDBSDE = pd.read_csv(LDBSDE_run_path_r+'Train_Valid_data.csv')  
        pd_data_LaDBSDE = pd.read_csv(LaDBSDE_run_path_r+'Train_Valid_data.csv')  
        
        data_DBSDE = pd_data_DBSDE.to_numpy()
        data_LDBSDE = pd_data_LDBSDE.to_numpy()
        data_LaDBSDE = pd_data_LaDBSDE.to_numpy()
        
        kappa = data_DBSDE[:, 0]        
        L_valid_DBSDE[:, q] = data_DBSDE[:, 2]        
        epsy_0_valid_DBSDE[:, q] = data_DBSDE[:, 3]
        tau_DBSDE[q] = data_DBSDE[-1, -1]

        L_valid_LDBSDE[:, q] = data_LDBSDE[:, 2]        
        epsy_0_valid_LDBSDE[:, q] = data_LDBSDE[:, 3]
        tau_LDBSDE[q] = data_LDBSDE[-1, -1]        
        
        L_valid_LaDBSDE[:, q] = data_LaDBSDE[:, 2]        
        epsy_0_valid_LaDBSDE[:, q] = data_LaDBSDE[:, 3]
        tau_LaDBSDE[q] = data_LaDBSDE[-1, -1] 
        
        Y_0_test = bsde.Y_tf(bsde.t[0], exp_X_test[:, :, 0])
        Yhat_0_DBSDE_arr[q] = Yhat_test_DBSDE[0][0]
        Yhat_0_LDBSDE_arr[q] = Yhat_test_LDBSDE[0][0]
        Yhat_0_LaDBSDE_arr[q] = Yhat_test_LaDBSDE[0][0]
        
        epsy_r_0_DBSDE[q] = epsy_r(Y_0_test, Yhat_test_DBSDE[0])
        epsy_r_0_LDBSDE[q] = epsy_r(Y_0_test, Yhat_test_LDBSDE[0])
        epsy_r_0_LaDBSDE[q] = epsy_r(Y_0_test, Yhat_test_LaDBSDE[0])
        
    # Moments of performance metrics
    Yhat_0_DBSDE_mean = np.mean(Yhat_0_DBSDE_arr)
    Yhat_0_DBSDE_std = np.std(Yhat_0_DBSDE_arr)
    
    Yhat_0_LDBSDE_mean = np.mean(Yhat_0_LDBSDE_arr)
    Yhat_0_LDBSDE_std = np.std(Yhat_0_LDBSDE_arr)

    Yhat_0_LaDBSDE_mean = np.mean(Yhat_0_LaDBSDE_arr)
    Yhat_0_LaDBSDE_std = np.std(Yhat_0_LaDBSDE_arr)
    
    epsy_r_0_DBSDE_mean, epsy_r_0_DBSDE_std = moments_f(epsy_r_0_DBSDE, 0)    
    epsy_r_0_LDBSDE_mean, epsy_r_0_LDBSDE_std = moments_f(epsy_r_0_LDBSDE, 0)    
    epsy_r_0_LaDBSDE_mean, epsy_r_0_LaDBSDE_std = moments_f(epsy_r_0_LaDBSDE, 0)    
    
    pd_metrics_Y0 = pd.DataFrame(
        {'Scheme': ["DBSDE", "LDBSDE", "LaDBSDE"],
         'Y_0': [Y_0_test, Y_0_test, Y_0_test],
         'Yhat_0_mean': [Yhat_0_DBSDE_mean, Yhat_0_LDBSDE_mean, Yhat_0_LaDBSDE_mean],
         'Yhat_0_std': [Yhat_0_DBSDE_std, Yhat_0_LDBSDE_std, Yhat_0_LaDBSDE_std],
         'vareps_y_r_0_mean':[epsy_r_0_DBSDE_mean, epsy_r_0_LDBSDE_mean, epsy_r_0_LaDBSDE_mean],
         'vareps_y_r_0_std':[epsy_r_0_DBSDE_std, epsy_r_0_LDBSDE_std, epsy_r_0_LaDBSDE_std]
        })
    
    pd_metrics_Y0.to_csv(LaDBSDE_run_path_r+"Y_metrics_t0.csv", float_format='%.16f', index=False)   
    
    L_valid_DBSDE_mean, L_valid_DBSDE_std = moments_f(L_valid_DBSDE, 1)
    epsy_0_valid_DBSDE_mean, epsy_0_valid_DBSDE_std =  moments_f(epsy_0_valid_DBSDE, 1)
    tau_DBSDE_mean, tau_DBSDE_std = moments_f(tau_DBSDE, 0)
    
    L_valid_LDBSDE_mean, L_valid_LDBSDE_std = moments_f(L_valid_LDBSDE, 1)
    epsy_0_valid_LDBSDE_mean, epsy_0_valid_LDBSDE_std =  moments_f(epsy_0_valid_LDBSDE, 1)
    tau_LDBSDE_mean, tau_LDBSDE_std = moments_f(tau_LDBSDE, 0)

    L_valid_LaDBSDE_mean, L_valid_LaDBSDE_std = moments_f(L_valid_LaDBSDE, 1)
    epsy_0_valid_LaDBSDE_mean, epsy_0_valid_LaDBSDE_std =  moments_f(epsy_0_valid_LaDBSDE, 1)
    tau_LaDBSDE_mean, tau_LaDBSDE_std = moments_f(tau_LaDBSDE, 0)
    
    save_dat(LaDBSDE_net_path, "L_valid", kappa, L_valid_DBSDE_mean, L_valid_DBSDE_std, L_valid_LDBSDE_mean, L_valid_LDBSDE_std, L_valid_LaDBSDE_mean, L_valid_LaDBSDE_std, '%d', '%.8e')    
    save_dat(LaDBSDE_net_path, "vareps_Y_0_valid", kappa, epsy_0_valid_DBSDE_mean, epsy_0_valid_DBSDE_std, epsy_0_valid_LDBSDE_mean, epsy_0_valid_LDBSDE_std, epsy_0_valid_LaDBSDE_mean, epsy_0_valid_LaDBSDE_std, '%d', '%.8e')
        
    plot_fig(LaDBSDE_net_path, "L_valid", kappa, L_valid_DBSDE_mean, L_valid_DBSDE_std, L_valid_LDBSDE_mean, L_valid_LDBSDE_std, L_valid_LaDBSDE_mean, L_valid_LaDBSDE_std, r'$\kappa$', r'$\tilde{\mathbf{L}}$', r'$\tilde{\mathbf{L}}$' + "DBSDE", r'$\tilde{\mathbf{L}}$' + "LDBSDE", r'$\tilde{\mathbf{L}}$' + "LaDBSDE")    
    plot_fig(LaDBSDE_net_path, "vareps_Y_0_valid", kappa, epsy_0_valid_DBSDE_mean, epsy_0_valid_DBSDE_std, epsy_0_valid_LDBSDE_mean, epsy_0_valid_LDBSDE_std, epsy_0_valid_LaDBSDE_mean, epsy_0_valid_LaDBSDE_std, r'$\kappa$', r'$\tilde{\varepsilon}^{y}_0$', r'$\tilde{\varepsilon}^{y}_0$' + "DBSDE", r'$\tilde{\varepsilon}^{y}_0$' + "LDBSDE", r'$\tilde{\varepsilon}^{y}_0$' + "LaDBSDE")

    pd_mom_comput_time = pd.DataFrame(
    {'Scheme': ["DBSDE", "LDBSDE", "LaDBSDE"],
     'tau_mean': [tau_DBSDE_mean, tau_LDBSDE_mean, tau_LaDBSDE_mean],
     'tau_std': [tau_DBSDE_std, tau_LDBSDE_std, tau_LaDBSDE_std]     
    })

    pd_mom_comput_time.to_csv(LaDBSDE_net_path+"moments_comput_time.csv", float_format='%.16f', index=False)   
