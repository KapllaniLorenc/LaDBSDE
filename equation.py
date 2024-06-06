import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# Define a single scalar Normal distribution.
dist = tfd.Normal(loc=0., scale=1.)

class Equation:
    """Base class for defining BSDE related function."""

    def __init__(self, eqn_config):
        self.T = np.float32(eqn_config["T"])
        self.d = eqn_config["d"]
        self.N = eqn_config["N"]
        self.dt = np.float32(self.T / self.N)
        self.t = np.arange(0, self.N+1, dtype = np.float32)*self.dt

    # Consider a sample of size B
    # We work with ABM or GBM
    def diffusion_tf(self, t, X): # R x R^(B x d) -> R^(B x d)
        """Diffusion function of the forward SDE."""
        raise NotImplementedError
    
    def sample(self, B): # N (integer) -> R^(B x d x N), R^(B x d x (N+1)), R^(B x d x (N+1))  (dWn, Xn, exp(Xn) if ln-transofrm, n=0,...,(N-1,N)) 
        """Sample of Brownian motion increment and forward SDE"""
        raise NotImplementedError

    def f_tf(self, t, X, Y, Z): # R x R^(B x d) x R^B x R^(B x d) -> R^B
        """Generator function of the BSDE."""
        raise NotImplementedError

    def g_tf(self, X): # R^(B x d) -> R^B
        """Terminal condition of the BSDE."""
        raise NotImplementedError

    def Y_tf(self, t, X): # R x R^(B x d) -> R^B
        """Exact solution Y of the BSDE."""
        raise NotImplementedError

    def Z_tf(self, t, X): # R x R^(B x d) -> R^(B x d)
        """Exact solution Z of the BSDE."""
        raise NotImplementedError

class SimpleBounded(Equation):
    """
    Simple bounded BSDE in Section 5.1 of Math. Comp. paper
    https://doi.org/10.1090/mcom/3514
    """
    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.X0 = np.ones(self.d, dtype=np.float32)
        self.a = np.float32(0.2/self.d)
        self.b = np.float32(1/np.sqrt(self.d))        

    def diffusion_tf(self, t, X):
        return self.b*tf.ones_like(X)

    def sample(self, B):
        dW = np.float32(np.random.normal(size=[B, self.d, self.N]) * np.sqrt(self.dt))
        X = np.zeros([B, self.d, self.N + 1], dtype=np.float32)
        X[:, :, 0] = np.ones([B, self.d], dtype=np.float32) * self.X0
        for n in range(self.N):
            X[:, :, n + 1] = X[:, :, n] + self.a * np.ones_like(X[:, :, n])*self.dt + self.b * dW[:, :, n]
        return dW, X, X

    def f_tf(self, t, X, Y, Z):
        alp = np.float32(0.5)
        scale = np.float32(0.5)
        expMat = tf.exp(alp*(self.T-t))
        xSum=tf.reduce_sum(X, axis=1, keepdims = True)
        c1 = (tf.cos(xSum) + tf.sin(xSum)*np.float32(0.2))*expMat
        c2 = - scale*tf.multiply( tf.pow(tf.cos(xSum)*expMat,2),tf.pow(-tf.sin(xSum)*expMat,2))
        c3 = scale*tf.multiply( tf.pow(Y,2), tf.pow(tf.reduce_mean(Z, axis=1, keepdims = True),2))
        F = c1 + c2 + c3
        return F
    
    def g_tf(self, X):
        xSum = tf.reduce_sum(X, axis=1, keepdims = True)
        G = tf.cos(xSum)
        return G

    def Y_tf(self, t, X):
        alp = np.float32(0.5)
        xSum = tf.reduce_sum(X, axis=1, keepdims = True)
        Y = tf.exp(alp*(self.T-t))*tf.cos(xSum)
        return Y
    
    def Z_tf(self, t, X):
        alp = np.float32(0.5)
        xSum = tf.reduce_sum(X, axis=1, keepdims = True)
        Z = -self.b*(tf.exp(alp*(self.T-t))*tf.sin(xSum))*tf.ones([1, self.d], dtype=tf.float32)
        return Z
    
class QuadraticZ(Equation):
    """
    PDE with quadratically growing derivatives in Section 4.6 of Comm. Math. Stat. paper
    doi.org/10.1007/s40304-017-0117-6
    """
    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.X0 = np.zeros(self.d, dtype=np.float32)
        self.c = np.float32(0.4)      

    def diffusion_tf(self, t, X):
        return tf.ones_like(X)

    def sample(self, B):
        dW = np.float32(np.random.normal(size=[B, self.d, self.N]) * np.sqrt(self.dt))
        X = np.zeros([B, self.d, self.N + 1], dtype=np.float32)
        X[:, :, 0] = np.ones([B, self.d], dtype=np.float32) * self.X0
        for n in range(self.N):
            X[:, :, n + 1] = X[:, :, n] + dW[:, :, n]
        return dW, X, X

    def f_tf(self, t, X, Y, Z):
        X_square = tf.reduce_sum(tf.square(X), 1, keepdims=True)
        base = self.T - t + X_square / self.d
        base_c = tf.pow(base, self.c)
        derivative = self.c * tf.pow(base, self.c - 1) * tf.cos(base_c)
        c1 = tf.reduce_sum(tf.square(Z), 1, keepdims=True)
        c2 = -4.0 * (derivative ** 2) * X_square / (self.d ** 2)
        c3 = derivative
        c4 = -0.5 * (
            2.0 * derivative + 4.0 / (self.d ** 2) * X_square * self.c * (
                (self.c - 1) * tf.pow(base, self.c - 2) * tf.cos(base_c) - (
                    self.c * tf.pow(base, 2 * self.c - 2) * tf.sin(base_c)
                    )
                )
            )
        F = c1 + c2 + c3 + c4
        return F
    
    def g_tf(self, X):   
        X_square = tf.reduce_sum(tf.square(X), 1, keepdims=True)
        base = X_square / self.d
        base_c = tf.pow(base, self.c)
        G = tf.sin(base_c)
        return G

    def Y_tf(self, t, X):
        X_square = tf.reduce_sum(tf.square(X), 1, keepdims=True)
        base = self.T - t + X_square / self.d
        base_c = tf.pow(base, self.c)
        Y = tf.sin(base_c)
        return Y
    
    def Z_tf(self, t, X):
        X_square = tf.reduce_sum(tf.square(X), 1, keepdims=True)
        base = self.T - t + X_square / self.d
        base_c = tf.pow(base, self.c)
        derivative = self.c * tf.pow(base, self.c - 1) * tf.cos(base_c)
        Z = 2.0 * derivative * X * (1/self.d) 
        return Z
    
class BlackScholesBarenblattLog(Equation):
    """
    Multidimensional BlackScholesBarenblatt PDE in Section 4.1 of Peter Carr 
    Gedenkschrift: Research Advances in Mathematical Finance paper
    https://www.worldscientific.com/doi/10.1142/9789811280306_0018
    using transformation X = ln(X)
    """
    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.X0 = np.array([1.0, 0.5]*int(self.d/2), dtype=np.float32)
        self.a = 0
        self.R = np.float32(0.05)
        self.b = np.float32(0.4)      

    def diffusion_tf(self, t, X):
        return self.b*tf.ones_like(X)

    def sample(self, B):
        dW = np.float32(np.random.normal(size=[B, self.d, self.N]) * np.sqrt(self.dt))
        X = np.zeros([B, self.d, self.N + 1], dtype=np.float32)
        exp_X = np.zeros([B, self.d, self.N + 1], dtype=np.float32)
        X[:, :, 0] = np.ones([B, self.d], dtype=np.float32) * np.log(self.X0)         
        exp_X[:, :, 0] = np.ones([B, self.d], dtype=np.float32) * self.X0         
        for n in range(self.N):
            X[:, :, n + 1] = X[:, :, n] + (self.a - 0.5*self.b**2) * np.ones_like(X[:, :, n])*self.dt + self.b * dW[:, :, n]
            exp_X[:, :, n + 1] = exp_X[:, :, n]*np.exp((self.a-(self.b**2)/2)*self.dt+self.b * dW[:, :, n]) 
        return dW, X, exp_X

    def f_tf(self, t, X, Y, Z):
        F = -self.R*(Y - (1/self.b)*tf.reduce_sum(Z, 1, keepdims = True))
        return F

    def g_tf(self, X):        
        G = tf.reduce_sum(tf.exp(X)**2, 1, keepdims = True)
        return tf.cast(G, tf.float32)
    
    def Y_tf(self, t, X):
        c = tf.exp(np.float32((self.R+self.b**2)*(self.T-t)))
        Y = c*tf.reduce_sum(X**2, 1, keepdims = True)
        return Y
    
    def Z_tf(self, t, X):
        c = tf.exp(np.float32((self.R+self.b**2)*(self.T-t)))
        Z = 2*self.b*c*X**2
        return Z
    
class BlackScholesLog(Equation):
    """
    Black-Scholes in Section 6 (Example 6.2) of Journal of Comp. Math. paper
    https://doi:10.4208/jcm.1212-m4014 using transformation X = ln(X)
    """
    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.X0 = np.ones(self.d, dtype=np.float32)*100
        self.K = 100
        self.R = np.float32(0.03)
        self.ck = np.float32(1/self.d)
        self.delta = np.float32(0.0)
        self.a = np.float32(0.05)        
        self.b = np.float32(0.2)      

    def diffusion_tf(self, t, X):
        return self.b*tf.ones_like(X)

    def sample(self, B):
        dW = np.float32(np.random.normal(size=[B, self.d, self.N]) * np.sqrt(self.dt))
        X = np.zeros([B, self.d, self.N + 1], dtype=np.float32)
        exp_X = np.zeros([B, self.d, self.N + 1], dtype=np.float32)
        X[:, :, 0] = np.ones([B, self.d], dtype=np.float32) * np.log(self.X0)         
        exp_X[:, :, 0] = np.ones([B, self.d], dtype=np.float32) * self.X0         
        for n in range(self.N):
            X[:, :, n + 1] = X[:, :, n] + (self.a - 0.5*self.b**2) * np.ones_like(X[:, :, n])*self.dt + self.b * dW[:, :, n]
            exp_X[:, :, n + 1] = exp_X[:, :, n]*np.exp((self.a-(self.b**2)/2)*self.dt+self.b * dW[:, :, n]) 
        return dW, X, exp_X

    def f_tf(self, t, X, Y, Z):
        F = -( self.R*Y + ((self.a - self.R + self.delta)/self.b)*tf.reduce_sum(Z, axis=1, keepdims = True))
        return F

    def g_tf(self, X):
        G = tf.maximum(tf.reduce_prod(tf.exp(X)**self.ck, axis=1, keepdims = True) - self.K, 0)
        return tf.cast(G, tf.float32)    

    def Y_tf(self, t, X):
        b_h = np.float32(self.b/np.sqrt(self.d))
        delta_h = np.float32((self.delta + 0.5*self.b**2) - 0.5*b_h**2)
        tau = self.T - t
        sqr_tau = tf.cast(tf.sqrt(tau), tf.float32)
        d1 = ( tf.math.log( tf.reduce_prod(X**self.ck, axis=1, keepdims = True) /self.K ) + (self.R - delta_h + 0.5*b_h**2)*tau ) / (b_h*sqr_tau)
        d2 = d1 - b_h*sqr_tau
        N_d1 = dist.cdf(d1)
        N_d2 = dist.cdf(d2)
        X_prod = tf.reduce_prod(X**self.ck, axis=1, keepdims = True)
        Y = tf.exp(-delta_h*tau)*X_prod*N_d1 - tf.exp(-self.R*tau)*self.K*N_d2
        return Y
    
    def Z_tf(self, t, X):
        b_h = np.float32(self.b/np.sqrt(self.d))
        delta_h = np.float32((self.delta + 0.5*self.b**2) - 0.5*b_h**2)
        tau = self.T - t
        sqr_tau = tf.cast(tf.sqrt(tau), tf.float32)
        d1 = ( tf.math.log( tf.reduce_prod(X**self.ck, axis=1, keepdims = True) /self.K ) + (self.R - delta_h + 0.5*b_h**2)*tau ) / (b_h*sqr_tau)
        N_d1 = dist.cdf(d1)
        X_prod = tf.reduce_prod(X**self.ck, axis=1, keepdims = True)
        Z = self.ck*tf.exp(-delta_h*tau)*X_prod*N_d1*self.b*tf.ones([1, self.d], dtype=tf.float32)
        return Z
        
class DIROneDimLog(Equation):
    """
    Different interest rates in Section 6.3.1 of Ann. Appl. Probab. paper
    https://www.jstor.org/stable/30038387 using transformation X = ln(X)
    for d = 1
    """
    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.X0 = np.ones(self.d, dtype=np.float32)*100
        self.K = 100
        self.R1 = np.float32(0.04)
        self.R2 = np.float32(0.06)
        self.a = np.float32(0.06)        
        self.b = np.float32(0.2)      

    def diffusion_tf(self, t, X):
        return self.b*tf.ones_like(X)

    def sample(self, B):
        dW = np.float32(np.random.normal(size=[B, self.d, self.N]) * np.sqrt(self.dt))
        X = np.zeros([B, self.d, self.N + 1], dtype=np.float32)
        exp_X = np.zeros([B, self.d, self.N + 1], dtype=np.float32)
        X[:, :, 0] = np.ones([B, self.d], dtype=np.float32) * np.log(self.X0)         
        exp_X[:, :, 0] = np.ones([B, self.d], dtype=np.float32) * self.X0         
        for n in range(self.N):
            X[:, :, n + 1] = X[:, :, n] + (self.a - 0.5*self.b**2) * np.ones_like(X[:, :, n])*self.dt + self.b * dW[:, :, n]
            exp_X[:, :, n + 1] = exp_X[:, :, n]*np.exp((self.a-(self.b**2)/2)*self.dt+self.b * dW[:, :, n]) 
        return dW, X, exp_X
    
    def f_tf(self, t, X, Y, Z):  
        temp = tf.reduce_sum(Z, 1, keepdims=True) / self.b
        F = -self.R1 * Y - (self.a - self.R1) * temp + (
            (self.R2 - self.R1) * tf.maximum(temp - Y, 0))
        return F

    def g_tf(self, X):
        G = tf.maximum(tf.reduce_sum(tf.exp(X), axis=1, keepdims = True) - self.K, 0)
        return tf.cast(G, tf.float32)    

    def Y_tf(self, t, X):
        tau = self.T - t
        b_h = np.float32(self.b*np.sqrt(tau))
        d1 = ( tf.math.log( tf.reduce_sum(X, axis=1, keepdims = True) /self.K ) + (self.R2 + 0.5*self.b**2)*tau) / b_h
        d2 = d1 - b_h
        N_d1 = dist.cdf(d1)
        N_d2 = dist.cdf(d2)
        Y = tf.reduce_sum(X, axis=1, keepdims = True)*N_d1 - tf.exp(-np.float32(self.R2*tau))*self.K*N_d2
        return Y
    
    def Z_tf(self, t, X):
        tau = self.T - t
        b_h = np.float32(self.b*np.sqrt(tau))
        d1 = ( tf.math.log( tf.reduce_sum(X, axis=1, keepdims = True) /self.K ) + (self.R2 + 0.5*self.b**2)*tau) / b_h
        N_d1 = dist.cdf(d1)
        Z = tf.reduce_sum(X, axis=1, keepdims = True)*N_d1*self.b*tf.ones([1, self.d], dtype=tf.float32)
        return Z

class DIRLog(Equation):
    """
    Different interest rates in Section 6.3.1 of Ann. Appl. Probab. paper
    https://www.jstor.org/stable/30038387 using transformation X = ln(X)
    """
    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.X0 = np.ones(self.d, dtype=np.float32)*100
        self.K1 = 120
        self.K2 = 150
        self.R1 = np.float32(0.04)
        self.R2 = np.float32(0.06)
        self.a = np.float32(0.06)        
        self.b = np.float32(0.2)      

    def diffusion_tf(self, t, X):
        return self.b*tf.ones_like(X)

    def sample(self, B):
        dW = np.float32(np.random.normal(size=[B, self.d, self.N]) * np.sqrt(self.dt))
        X = np.zeros([B, self.d, self.N + 1], dtype=np.float32)
        exp_X = np.zeros([B, self.d, self.N + 1], dtype=np.float32)
        X[:, :, 0] = np.ones([B, self.d], dtype=np.float32) * np.log(self.X0)         
        exp_X[:, :, 0] = np.ones([B, self.d], dtype=np.float32) * self.X0         
        for n in range(self.N):
            X[:, :, n + 1] = X[:, :, n] + (self.a - 0.5*self.b**2) * np.ones_like(X[:, :, n])*self.dt + self.b * dW[:, :, n]
            exp_X[:, :, n + 1] = exp_X[:, :, n]*np.exp((self.a-(self.b**2)/2)*self.dt+self.b * dW[:, :, n]) 
        return dW, X, exp_X
    
    def f_tf(self, t, X, Y, Z):  
        temp = tf.reduce_sum(Z, 1, keepdims=True) / self.b
        F = -self.R1 * Y - (self.a - self.R1) * temp + (
            (self.R2 - self.R1) * tf.maximum(temp - Y, 0))
        return F

    def g_tf(self, X):
        temp = tf.reduce_max(tf.exp(X), 1, keepdims=True)
        G = tf.maximum(temp - self.K1, 0) - 2 * tf.maximum(temp - self.K2, 0)     
        return tf.cast(G, tf.float32)    

    # Each reference value calculated using Multilevel Picard method. 
    # Parameters used for the Matlab code: 
    # rho = 7, nr_runs = 10 (two different seeds rng(2016), rng(2017));
    def Y_tf(self, t, X):
        if self.d == 50:
            Y_0_MPI = 17.97429
        elif self.d == 100:
            Y_0_MPI = 21.2988
        else:
            Y_0_MPI = 0.0
        
        return Y_0_MPI