import time
import numpy as np
import tensorflow as tf

class BSDESolver(object):
    """The LaDBSDE solver."""
    def __init__(self, eqn_config, net_config, bsde):
        self.eqn_config = eqn_config
        self.net_config = net_config
        self.bsde = bsde
        self.model = ladbsde(eqn_config, net_config, bsde)
                    
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        self.net_config["lr_boundaries"], self.net_config["lr_values"])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            
    def train(self):
        start_time = time.time()
        loss_history = []
        Yhat_0_history = []
        Zhat_0_history = []
        time_history = []
        dW_valid, X_valid = self.bsde.sample(self.net_config["M_valid"])
        YN_valid = self.bsde.g_tf(X_valid[:, :, -1])
        t = self.bsde.t
        N = self.bsde.N
        # begin sgd iteration
        for step in range(self.net_config["k"]+1):
            dW_train, X_train = self.bsde.sample(self.net_config["batch_size"])
            YN_train = self.bsde.g_tf(X_train[:, :, -1])
            if step % self.net_config["k_disp"] == 0:
                loss_valid = self.loss_fn((t, X_valid, dW_valid), YN_valid, training=False).numpy()                 
                Yhat_valid, Zhat_valid = self.model((t, X_valid, dW_valid), training=False)                
                Y0 = tf.reduce_mean(self.bsde.Y_tf(t[0], X_valid[:, :, 0])).numpy()
                Z0 = tf.reduce_mean(self.bsde.Z_tf(t[0], X_valid[:, :, 0]), axis = 0).numpy()
                Yhat_0 = tf.reduce_mean(Yhat_valid[0]).numpy()
                Zhat_0 = tf.reduce_mean(Zhat_valid[0], axis = 0).numpy()                
                elapsed_time = time.time() - start_time                
                loss_history.append(loss_valid)                
                Yhat_0_history.append(Yhat_0)                
                Zhat_0_history.append(Zhat_0)                
                time_history.append(elapsed_time)                
                if self.net_config["verbose"]:
                    print("k: %2u, L: %.2e, Y_0: %.4f, Yh_0: %.4f, Z_0: %.4f, Zh_0: %.4f, time: %2u" % (step, loss_valid, Y0, Yhat_0, np.average(Z0), np.average(Zhat_0), elapsed_time))
                    
            self.train_step((t, X_train, dW_train), YN_train, training=False)                         
        print("k_final: %2u, L: %.2e, Y_0: %.4f, Yh_0: %.4f, Z_0: %.4f, Zh_0: %.4f, time: %2u" % (step, loss_valid, Y0, Yhat_0, np.average(Z0), np.average(Zhat_0), elapsed_time))

        return [loss_history, Yhat_0_history, Zhat_0_history, time_history, Y0, Z0]

    def loss_fn(self, inputs, outputs, training):
        t, X, dW = inputs        
        Y_hat, Z_hat = self.model(inputs, training)
        Y_N = outputs
        N = self.bsde.N
        h = self.bsde.h
        loss_vec = tf.zeros_like(Y_N)
        Y_add = []
        Yi = Y_N
        for j in range(N-1, -1, -1):
            Yi = Yi + self.bsde.f_tf(t[j], X[:, :, j], Y_hat[j], Z_hat[j])*h - tf.reduce_sum(Z_hat[j]*dW[:, :, j], axis = 1, keepdims=True)
            Y_add.append(Yi)            
        for i in range(0, N):
            loss_vec += tf.square(Y_hat[i] - Y_add[N-1-i])
        loss = tf.reduce_mean(loss_vec)
        return loss

    def grad(self, inputs, outputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, outputs, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, input_train_data, output_train_data, training):
        grad = self.grad(input_train_data, output_train_data, training)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

class ladbsde(tf.keras.Model):
    def __init__(self, eqn_config, net_config, bsde):
        super(ladbsde, self).__init__()
        self.eqn_config = eqn_config
        self.net_config = net_config
        self.bsde = bsde
        self.DNN_y_z = FeedForwardNet_y_z(eqn_config, net_config, bsde)

    def call(self, inputs, training):
        t, X, dW = inputs
        Y_hat = []
        Z_hat = []
        for i in range(0, self.bsde.N):
            ti = t[i]*tf.ones(shape=[tf.shape(X)[0], 1])
            yi, zi = self.DNN_y_z((ti, X[:, :, i]), training)
            Y_hat.append(yi)
            Z_hat.append(zi)
        return Y_hat, Z_hat

class FeedForwardNet_y_z(tf.keras.Model):
    def __init__(self, eqn_config, net_config, bsde):
        super(FeedForwardNet_y_z, self).__init__()
        d = eqn_config["d"]
        self.L = net_config["L"]
        self.bsde = bsde
        hidden_neurons = net_config["hidden_neurons"]        
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(self.L + 2)]
        self.dense_layers = [tf.keras.layers.Dense(hidden_neurons[j],
                                                   use_bias=False,
                                                   activation=None)
                             for j in range(self.L)]
        # final output have size 1 for y
        self.dense_layers.append(tf.keras.layers.Dense(1, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> tanh) * L -> dense"""
        with tf.GradientTape(persistent=True) as tape:
            t = x[0]
            X = x[1]
            tape.watch(X)            
            h = self.bn_layers[0](tf.concat([t, X], axis = 1), training)
            for j in range(self.L):
                h = self.dense_layers[j](h)
                h = self.bn_layers[j+1](h, training)
                h = tf.tanh(h)
            h = self.dense_layers[-1](h)
            y = self.bn_layers[-1](h, training)
        dy = tape.gradient(y, X)
        z = dy*self.bsde.diffusion_tf(t, X)
        del tape
        return y, z
      
    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))