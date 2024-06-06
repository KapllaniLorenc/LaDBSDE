import time
import numpy as np
import tensorflow as tf

class BSDESolver(object):
    """The LDBSDERNN solver."""
    def __init__(self, eqn_config, net_config, bsde):
        self.eqn_config = eqn_config
        self.net_config = net_config
        self.bsde = bsde
        self.model = ldbsdeRNN(eqn_config, net_config, bsde)
                    
        alpha_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        self.net_config["alpha_boundaries"], self.net_config["alpha_values"])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_schedule)
            
    def train(self):
        start_time = time.time()
        training_history = []
        dW_valid, X_valid, exp_X_valid = self.bsde.sample(self.net_config["B_valid"])
        Y_N_valid = self.bsde.g_tf(X_valid[:, :, -1])
        t = self.bsde.t
        N = self.bsde.N
        # begin sgd iteration
        for kappa in range(self.net_config["Kf"]+1):
            dW_train, X_train, exp_X_train = self.bsde.sample(self.net_config["B"])
            Y_N_train = self.bsde.g_tf(X_train[:, :, -1])
            if kappa % self.net_config["k_disp"] == 0:
                Yhat_valid, Zhat_valid = self.model((t, X_valid, dW_valid))                
                if self.eqn_config["flag_exact_solution"] == True:
                    L_train = self.loss_fn((t, X_train, dW_train), Y_N_train).numpy()                 
                    L_valid = self.loss_fn((t, X_valid, dW_valid), Y_N_valid).numpy()                 
                    Y_0 = self.bsde.Y_tf(self.bsde.t[0], exp_X_valid[:, :, 0])
                    Z_0 = self.bsde.Z_tf(self.bsde.t[0], exp_X_valid[:, :, 0])                
                    Yhat_0 = Yhat_valid[0]
                    Zhat_0 = Zhat_valid[0]                      
                    epsy_0 = tf.reduce_mean(tf.pow(Y_0-Yhat_0, 2)).numpy()
                    epsz_0 = tf.reduce_mean(tf.reduce_sum(tf.pow(Z_0-Zhat_0, 2), axis = 1)).numpy()    
                else:
                    L_train = self.loss_fn((t, X_train, dW_train), Y_N_train).numpy()                 
                    L_valid = self.loss_fn((t, X_valid, dW_valid), Y_N_valid).numpy()                 
                    Y_0 = self.bsde.Y_tf(self.bsde.t[0], exp_X_valid[:, :, 0])
                    Yhat_0 = Yhat_valid[0]
                    epsy_0 = tf.reduce_mean(tf.pow(Y_0-Yhat_0, 2)).numpy()
                    
                tau = time.time() - start_time     
                if self.eqn_config["flag_exact_solution"] == True:
                    training_history.append([kappa, L_train, L_valid, epsy_0, epsz_0, tau])
                else:
                    training_history.append([kappa, L_train, L_valid, epsy_0, tau])
                    
                if self.net_config["verbose"]:                    
                    if self.eqn_config["flag_exact_solution"] == True:
                        print("kappa: %2u, L_train: %.2e, L_valid: %.2e, epsy_0: %.2e, epsz_0: %.2e time: %2u" % (kappa, L_train, L_valid, epsy_0, epsz_0, tau))
                    else:
                        print("kappa: %2u, L_train: %.2e, L_valid: %.2e, epsy_0: %.2e, time: %2u" % (kappa, L_train, L_valid, epsy_0, tau))
                        
            self.train_step((t, X_train, dW_train), Y_N_train)                         
        return np.array(training_history)

    def loss_fn(self, inputs, outputs):
        t, X, dW = inputs        
        Yhat, Zhat = self.model(inputs)
        Y_N = outputs
        N = self.bsde.N
        dt = self.bsde.dt
        L_vec = tf.zeros_like(Y_N)
        L = 0
        for n in range(0, N):
            Yhat_np1_EM = Yhat[n] - self.bsde.f_tf(t[n], X[:, :, n], Yhat[n], Zhat[n])*dt + tf.reduce_sum(Zhat[n]*dW[:, :, n], axis = 1, keepdims=True)
            L_vec += tf.square(Yhat[n+1] - Yhat_np1_EM)
        L_vec += tf.square(Yhat[N] - Y_N)  
        L = tf.reduce_mean(L_vec)
        return L

    def grad(self, inputs, outputs):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, outputs)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, input_train_data, output_train_data):
        grad = self.grad(input_train_data, output_train_data)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

class ldbsdeRNN(tf.keras.Model):
    def __init__(self, eqn_config, net_config, bsde):
        super().__init__()
        self.eqn_config = eqn_config
        self.net_config = net_config
        self.bsde = bsde
        eta = net_config["eta"]
        self.h_m1 = tf.Variable(tf.random.normal([1, eta[0]], stddev=np.sqrt(2) / np.sqrt(1 + eta[0]), dtype=tf.float32))
        self.DNN_y_z = FeedForwardNet_y_z(eqn_config, net_config, bsde)

    def call(self, inputs, training):
        t, X, dW = inputs
        Yhat = []
        Zhat = []
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(X)[0], 1]), dtype=tf.float32)
        h = all_one_vec * self.h_m1
        for n in range(0, self.bsde.N+1):
            t_n_tf = t[n]*all_one_vec
            y_n, z_n, h = self.DNN_y_z((t_n_tf, X[:, :, n], h))
            Yhat.append(y_n)
            Zhat.append(z_n)
        return Yhat, Zhat

class FeedForwardNet_y_z(tf.keras.Model):
    def __init__(self, eqn_config, net_config, bsde):
        super().__init__()
        d = eqn_config["d"]
        self.bsde = bsde
        eta = net_config["eta"]        
        self.dense_layers_X = tf.keras.layers.Dense(eta[0],
                                                   use_bias=False,
                                                   activation=None)
        self.dense_layers_h = tf.keras.layers.Dense(eta[0],
                                                   use_bias=True,
                                                   activation=None)
        # final output have size 1 for Y
        self.dense_layer_y = tf.keras.layers.Dense(1, activation=None)
        
    def call(self, x):
        """structure: RNN cell"""
        with tf.GradientTape(persistent=True) as tape:
            t = x[0]
            X = x[1]
            h = x[2]
            tape.watch(X)
            inp = tf.concat([t, X], axis = 1)
            h = tf.tanh(tf.add(self.dense_layers_h(h), self.dense_layers_X(inp)))
            y = self.dense_layer_y(h)
        dy = tape.gradient(y, X)
        z = dy*self.bsde.diffusion_tf(t[0], X)
        del tape
        return y, z, h

    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))