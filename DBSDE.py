import time
import numpy as np
import tensorflow as tf

class BSDESolver(object):
    """The DBSDE solver."""
    def __init__(self, eqn_config, net_config, bsde):
        self.eqn_config = eqn_config
        self.net_config = net_config
        self.bsde = bsde
        self.model = dbsde(eqn_config, net_config, bsde)
                    
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
                Yhat_valid, Zhat_valid = self.model((X_valid, dW_valid), training=False)                
                if self.eqn_config["flag_exact_solution"] == True:
                    L_train = self.loss_fn((X_train, dW_train), Y_N_train, training=False).numpy()                 
                    L_valid = self.loss_fn((X_valid, dW_valid), Y_N_valid, training=False).numpy()                 
                    Y_0 = self.bsde.Y_tf(self.bsde.t[0], exp_X_valid[:, :, 0])
                    Z_0 = self.bsde.Z_tf(self.bsde.t[0], exp_X_valid[:, :, 0])                
                    Yhat_0 = Yhat_valid[0]
                    Zhat_0 = Zhat_valid[0]                      
                    epsy_0 = tf.reduce_mean(tf.pow(Y_0-Yhat_0, 2)).numpy()
                    epsz_0 = tf.reduce_mean(tf.reduce_sum(tf.pow(Z_0-Zhat_0, 2), axis = 1)).numpy()   
                else:
                    L_train = self.loss_fn((X_train, dW_train), Y_N_train, training=False).numpy()                 
                    L_valid = self.loss_fn((X_valid, dW_valid), Y_N_valid, training=False).numpy()                 
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
                        
            self.train_step((X_train, dW_train), Y_N_train)                         
        return np.array(training_history)

    def loss_fn(self, inputs, outputs, training):
        X, dW = inputs        
        Yhat, Zhat = self.model(inputs, training)
        Y_N = outputs
        delta = Yhat[-1] - Y_N
        # use linear approximation outside the clipped range
        DELTA_CLIP = 50.0
        L = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                       2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        return L

    def grad(self, inputs, outputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, outputs, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, input_train_data, output_train_data):
        grad = self.grad(input_train_data, output_train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

class dbsde(tf.keras.Model):
    def __init__(self, eqn_config, net_config, bsde):
        super().__init__()
        self.eqn_config = eqn_config
        self.net_config = net_config
        self.bsde = bsde
        self.Yhat_0 = tf.Variable(np.random.uniform(low=self.net_config["Y_0_min"],
                                                    high=self.net_config["Y_0_max"],
                                                    size=[1]), dtype=tf.float32
                                  )
        self.Zhat_0 = tf.Variable(np.random.uniform(low=-1, high=1,
                                                    size=[1, self.eqn_config["d"]]), dtype=tf.float32
                                  )

        self.DNNs_z = [FeedForwardNet_z(eqn_config, net_config) for _ in range(self.bsde.N-1)]

    def call(self, inputs, training):
        X, dW = inputs
        t = self.bsde.t
        dt = self.bsde.dt
        N = self.bsde.N
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(X)[0], 1]), dtype=tf.float32)
        Yhat = []
        Zhat = []
        Yhat_0 = all_one_vec * self.Yhat_0
        Zhat_0 = tf.matmul(all_one_vec, self.Zhat_0)
        Yhat.append(Yhat_0)
        Zhat.append(Zhat_0)        
        for n in range(0, N-1):
            Yhat_np1 = Yhat[n] - self.bsde.f_tf(t[n], X[:, :, n], Yhat[n], Zhat[n])*dt + tf.reduce_sum(Zhat[n]*dW[:, :, n], axis = 1, keepdims=True)
            Zhat_np1 = self.DNNs_z[n](X[:, :, n + 1], training) / self.eqn_config["d"]
            Yhat.append(Yhat_np1)
            Zhat.append(Zhat_np1)
            
        # terminal time
        Yhat_N = Yhat[N-1] - self.bsde.f_tf(t[N-1], X[:, :, N-1], Yhat[N-1], Zhat[N-1])*dt + tf.reduce_sum(Zhat[N-1]*dW[:, :, N-1], axis = 1, keepdims=True)
        Yhat.append(Yhat_N)
        
        return Yhat, Zhat
    
class FeedForwardNet_z(tf.keras.Model):
    def __init__(self, eqn_config, net_config):
        super().__init__()
        d = eqn_config["d"]
        self.L = net_config["L"]
        eta = net_config["eta"]        
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(self.L + 2)]
        self.dense_layers = [tf.keras.layers.Dense(eta[l],
                                                   use_bias=False,
                                                   activation=None)
                             for l in range(self.L)]
        # final output have size d for z
        self.dense_layers.append(tf.keras.layers.Dense(d, activation=None))

    def call(self, x, training):
        """structure: input -> bn -> (dense -> bn -> relu) * L -> dense"""
        h = self.bn_layers[0](x, training)
        for l in range(self.L):
            h = self.dense_layers[l](h)
            h = self.bn_layers[l+1](h, training)
            h = tf.nn.relu(h)
        h = self.dense_layers[-1](h)
        z = self.bn_layers[-1](h, training)
        return z
      
    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))