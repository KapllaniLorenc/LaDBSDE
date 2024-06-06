import numpy as np
import equation as eq

def int_var(example_type, flag_verbose, method, d, T, N):
    if example_type in ["BSLog", "BSBLog", "DIROneDimLog", "DIRLog"]:
        flag_log_transform = True
    else:
        flag_log_transform = False
        
    if example_type in ["DIRLog"]:
        flag_exact_solution = False
    else:
        flag_exact_solution = True    

    eqn_config = {"d":d, "T": T, "N":N, "flag_log_transform":flag_log_transform, "flag_exact_solution": flag_exact_solution}
    
    if method == "DBSDE":
        """
        net_config = {"L":2, "eta": [d+10, d+10], "Kf": 60000,
                      "alpha_values": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], "alpha_boundaries": [20000, 30000, 40000, 50000],
                      "k_disp": 100, "verbose": flag_verbose, "B_valid": 1024, "B_test": 1024, "B": 128}
        """
        net_config = {"L":2, "eta": [d+10, d+10], "Kf": 25000,
                      "alpha_values": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4], "alpha_boundaries": [5000, 10000, 15000, 20000],
                      "k_disp": 100, "verbose": flag_verbose, "B_valid": 1024, "B_test": 1024, "B": 128}
        
        if example_type == "SB":
            net_config["Y_0_min"] = 0.0
            net_config["Y_0_max"] = 5.0

        elif example_type == "QZ":
            net_config["Y_0_min"] = 0.0
            net_config["Y_0_max"] = 4.0

        elif example_type == "BSBLog":
            if d == 2:
                net_config["Y_0_min"] = 0.0
                net_config["Y_0_max"] = 5.0
            elif d == 10:
                net_config["Y_0_min"] = 5.0
                net_config["Y_0_max"] = 10.0            
            elif d == 50:
                net_config["Y_0_min"] = 30.0
                net_config["Y_0_max"] = 40.0            
            elif d == 100:
                net_config["Y_0_min"] = 70.0
                net_config["Y_0_max"] = 80.0    

        elif example_type == "DIRLog":
            if d == 100:
                net_config["Y_0_min"] = 10.0
                net_config["Y_0_max"] = 30.0
        
    else:
        """
        net_config = {"L":4, "eta": [d+10, d+10, d+10, d+10], "Kf": 60000,
                      "alpha_values": [1e-3, 3e-4, 1e-4, 3e-5, 1e-5], "alpha_boundaries": [20000, 30000, 40000, 50000],
                      "k_disp": 100, "verbose": flag_verbose, "B_valid": 1024, "B_test": 1024, "B": 128}
        """
        net_config = {"L":4, "eta": [d+10, d+10, d+10, d+10], "Kf": 25000,
                      "alpha_values": [1e-3, 3e-4, 1e-4, 3e-5, 1e-5], "alpha_boundaries": [5000, 10000, 15000, 20000],
                      "k_disp": 100, "verbose": flag_verbose, "B_valid": 1024, "B_test": 1024, "B": 128}
        
    if example_type == "SB":
        bsde = eq.SimpleBounded(eqn_config)
        bsde_params = "X0_"+str(np.mean(bsde.X0))+"_a_"+str(bsde.a)+"_b_"+str(np.round(bsde.b, 2))

    elif example_type == "QZ":
        bsde = eq.QuadraticZ(eqn_config)    
        bsde_params = "X0_"+str(np.mean(bsde.X0))+"_c_"+str(bsde.c)
                
    elif example_type == "BSBLog":
        bsde = eq.BlackScholesBarenblattLog(eqn_config)    
        bsde_params = "X0_"+str(np.round(np.mean(bsde.X0), 2))+"_b_"+str(bsde.b)+"_R_"+str(bsde.R)

    elif example_type == "BSLog":
        bsde = eq.BlackScholesLog(eqn_config)    
        bsde_params = "X0_"+str(np.mean(bsde.X0))+"_a_"+str(bsde.a)+"_b_"+str(bsde.b)+"_R_"+str(bsde.R)+"_ck_"+str(bsde.ck)+"_K_"+str(bsde.K)+"_del_"+str(bsde.delta)

    elif example_type == "DIROneDimLog":
        bsde = eq.DIROneDimLog(eqn_config)    
        bsde_params = "X0_"+str(np.mean(bsde.X0))+"_a_"+str(bsde.a)+"_b_"+str(bsde.b)+"_R1_"+str(bsde.R1)+"_R2_"+str(bsde.R2)+"_K_"+str(bsde.K)

    elif example_type == "DIRLog":
        bsde = eq.DIRLog(eqn_config)    
        bsde_params = "X0_"+str(np.mean(bsde.X0))+"_a_"+str(bsde.a)+"_b_"+str(bsde.b)+"_R1_"+str(bsde.R1)+"_R2_"+str(bsde.R2)+"_K1_"+str(bsde.K1)+"_K2_"+str(bsde.K2)
        
    return eqn_config, net_config, bsde, bsde_params 