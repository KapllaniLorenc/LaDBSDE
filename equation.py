import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# Define a single scalar Normal distribution.
dist = tfd.Normal(loc=0., scale=1.)

class Equation(object):
    """Base class for defining BSDE related function."""

    def __init__(self, eqn_config):
        self.T = np.float32(eqn_config["T"])
        self.d = eqn_config["d"]
        self.N = eqn_config["N"]
        self.h = self.T / self.N
        self.t = np.arange(0, self.N+1, dtype = np.float32)*self.h

    # Consider a sample of size m
    # We work with ABM or GBM
    
    def diffusion_tf(self, t, X): # R x R^(m x d) -> R^(m x d)
        """Diffusion function of the forward SDE."""
        raise NotImplementedError        
        
    def sample(self, num_sample): # N (integer) -> R^(m x d x N), R^(m x d x (N+1))  (dWi, Xi, i=0,...,(N-1,N)) 
        """Sample of Brownian motion increment and forward SDE."""
        raise NotImplementedError

    def f_tf(self, t, X, Y, Z): # R x R^(m x d) x R^m x R^(m x d) -> R^m
        """Generator function of the BSDE."""
        raise NotImplementedError

    def g_tf(self, X): # R^(m x d) -> R^m
        """Terminal condition of the BSDE."""
        raise NotImplementedError
        
    def dg_tf(self, X): # R^(m x d) -> R^(m x d)
        """Gradient of the terminal condition of the BSDE."""
        raise NotImplementedError

    def Y_tf(self, t, X): # R x R^(m x d) -> R^m
        """Exact solution Y of the BSDE."""
        raise NotImplementedError

    def Z_tf(self, t, X): # R x R^(m x d) -> R^(m x d)
        """Exact solution Z of the BSDE."""
        raise NotImplementedError

class Simplelinear(Equation):
    """
    Simple linear in Section 5 (Example 1) of Journal of Applied Num. Math.
    https://doi.org/10.1016/j.apnum.2019.09.016
    """
    def __init__(self, eqn_config):
        super(Simplelinear, self).__init__(eqn_config)
        self.X0 = np.zeros(self.d)
        self.T = eqn_config["T"]

    def diffusion_tf(self, t, X):
        return tf.ones_like(X)
    
    def sample(self, m):
        dW = np.random.normal(size=[m, self.d, self.N]) * np.sqrt(self.h)
        X = np.zeros([m, self.d, self.N + 1])
        X[:, :, 0] = np.ones([m, self.d]) * self.X0
        for i in range(self.N):
            X[:, :, i + 1] = X[:, :, i] + dW[:, :, i]
        return np.float32(dW), np.float32(X)

    def f_tf(self, t, X, Y, Z):
        F = -np.float32(5/8)*Y
        return tf.cast(F, tf.float32)

    def g_tf(self, X):
        G = tf.exp(0.5*tf.reduce_sum(X, axis=1, keepdims = True) + 0.5*self.T)
        return tf.cast(G, tf.float32)
    
    def dg_tf(self, X):
        with tf.GradientTape() as tape:
            tape.watch(X)
            g = self.g_tf(X)
        dG = tape.gradient(g, X)
        return tf.cast(dG, tf.float32)
    
    def Y_tf(self, t, X):        
        Y = tf.exp(0.5*tf.reduce_sum(X, axis=1, keepdims = True) + 0.5*t)
        return tf.cast(Y, tf.float32)
    
    def Z_tf(self, t, X):
        Z = 0.5*tf.exp(0.5*tf.reduce_sum(X, axis=1, keepdims = True) + 0.5*t)*tf.ones([1, self.d])
        return tf.cast(Z, tf.float32)
    
    
class BlackScholes(Equation):
    """
    Black-Scholes in Section 6 (Example 6.2) of Journal of Comp. Math. paper
    https://doi:10.4208/jcm.1212-m4014
    """
    def __init__(self, eqn_config):
        super(BlackScholes, self).__init__(eqn_config)
        self.X0 = np.ones(self.d)*100
        self.K = 100
        self.r = 0.03
        self.alpha = 1/self.d
        self.delta = 0.0
        self.mu = 0.05        
        self.sigma = 0.2

    def diffusion_tf(self, t, X):
        return self.sigma*X
    
    def sample(self, m):
        dW = np.random.normal(size=[m, self.d, self.N]) * np.sqrt(self.h)
        X = np.zeros([m, self.d, self.N + 1])
        X[:, :, 0] = np.ones([m, self.d]) * self.X0
        factor = np.exp((self.mu-(self.sigma**2)/2)*self.h)
        for i in range(self.N):
            #X[:, :, i + 1] = X[:, :, i] + self.mu * X[:, :, i]*self.h + self.sigma * X[:, :, i] * dW[:, :, i]            
            X[:, :, i + 1] = (factor * np.exp(self.sigma * dW[:, :, i])) * X[:, :, i]
        return np.float32(dW), np.float32(X)

    def f_tf(self, t, X, Y, Z):
        F = -( np.float32(self.r)*Y + np.float32((self.mu - self.r + self.delta)/self.sigma)*tf.reduce_sum(Z, axis=1, keepdims = True))
        return tf.cast(F, tf.float32)

    def g_tf(self, X):
        G = tf.maximum(tf.reduce_prod(X**self.alpha, axis=1, keepdims = True) - self.K, 0)
        return tf.cast(G, tf.float32)
    
    def dg_tf(self, X):
        with tf.GradientTape() as tape:
            tape.watch(X)
            g = self.g_tf(X)
        dG = tape.gradient(g, X)
        return tf.cast(dG, tf.float32)
    
    def Y_tf(self, t, X):
        sigma_h = self.sigma/np.sqrt(self.d)
        delta_h = (self.delta + 0.5*self.sigma**2) - 0.5*sigma_h**2
        d1 = ( tf.math.log( tf.reduce_prod(X**self.alpha, axis=1, keepdims = True) /self.K ) + (self.r - delta_h + 0.5*sigma_h**2)*(self.T - t) ) / ( sigma_h*tf.sqrt(self.T - t) )
        d2 = d1 - sigma_h*tf.sqrt(self.T - t)
        N_d1 = dist.cdf(d1)
        N_d2 = dist.cdf(d2)
        X_prod = tf.reduce_prod(X**self.alpha, axis=1, keepdims = True)
        Y = tf.exp(-np.float32(delta_h)*(self.T - t))*X_prod*N_d1 - tf.exp(-np.float32(self.r)*(self.T - t))*self.K*N_d2
        return tf.cast(Y, tf.float32)
    
    def Z_tf(self, t, X):
        sigma_h = self.sigma/np.sqrt(self.d)
        delta_h = (self.delta + 0.5*self.sigma**2) - 0.5*sigma_h**2
        d1 = ( tf.math.log( tf.reduce_prod(X**self.alpha, axis=1, keepdims = True) /self.K ) + (self.r - delta_h + 0.5*sigma_h**2)*(self.T - t) ) / ( sigma_h*tf.sqrt(self.T - t) )
        N_d1 = dist.cdf(d1)
        X_prod = tf.reduce_prod(X**self.alpha, axis=1, keepdims = True)
        Z = self.alpha*tf.exp(-np.float32(delta_h)*(self.T - t))*X_prod*N_d1*np.float32(self.sigma)*tf.ones([1, self.d])
        return tf.cast(Z, tf.float32)
    
class BurgersType(Equation):
    """
    Multidimensional Burgers-type PDE in Section 4.5 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super(BurgersType, self).__init__(eqn_config)
        self.d = eqn_config["d"]  
        self.T = eqn_config["T"]  
        self.X0 = np.zeros(self.d)
        self.sigma = self.d     

    def diffusion_tf(self, t, X):
        return self.sigma*tf.ones_like(X)

    def sample(self, m):
        dW = np.random.normal(size=[m, self.d, self.N]) * np.sqrt(self.h)
        X = np.zeros([m, self.d, self.N + 1])
        X[:, :, 0] = np.ones([m, self.d]) * self.X0
        for i in range(self.N):
            X[:, :, i + 1] = X[:, :, i] + self.sigma * dW[:, :, i]
        return np.float32(dW), np.float32(X)

    def f_tf(self, t, X, Y, Z):
        F = ((self.sigma/self.d)*Y - (2*self.d+self.sigma**2) /(2*self.sigma*self.d)) * tf.reduce_sum(Z, 1, keepdims=True)
        return tf.cast(F, tf.float32)

    def g_tf(self, X):
        G =  1 - 1.0 / (1 + tf.exp(self.T + tf.reduce_sum(X, 1, keepdims=True) / self.d))
        return tf.cast(G, tf.float32)
    
    def dg_tf(self, X):
        with tf.GradientTape() as tape:
            tape.watch(X)
            g = self.g_tf(X)
        dG = tape.gradient(g, X)
        return tf.cast(dG, tf.float32)
    
    def Y_tf(self, t, X):
        Y = 1 - 1.0 / (1 + tf.exp(t + tf.reduce_sum(X, 1, keepdims=True) / self.d))
        return tf.cast(Y, tf.float32)
    
    def Z_tf(self, t, X):
        a = tf.exp(t + tf.reduce_sum(X, 1, keepdims=True) / self.d)
        Z = (self.sigma/self.d)*(a/(1+a)**2)*tf.ones([1, self.d])
        return tf.cast(Z, tf.float32)