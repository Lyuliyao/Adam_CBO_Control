import tensorflow as tf
import numpy as np

# Set data type
DTYPE='float64'
tf.keras.backend.set_floatx(DTYPE)
print('TensorFlow version used: {}'.format(tf.__version__))

class DeepBSDE(tf.keras.Model):
    def __init__(self, t0=0.0, t1=1.0, dim=10, time_steps=20, sigma=np.sqrt(2),
                 learning_rate=1e-2, num_hidden_layers=2, num_neurons=200, **kwargs):
        super().__init__(**kwargs)
        
        self.t0 = t0
        self.t1 = t1
        self.N = time_steps
        self.dim = dim
        self.sigma = sigma
        self.x = 0.0*np.ones(self.dim)
        self.dt = (t1 - t0)/(self.N)
        self.sqrt_dt = np.sqrt(self.dt)
        self.t_space = np.linspace(self.t0, self.t1, self.N + 1)[:-1]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-8)
        
        self.u0 = tf.Variable(np.random.uniform(.3, .5, size=(1)).astype(DTYPE))
        self.gradu0 = tf.Variable(np.random.uniform(-1e-1, 1e-1, size=(1, dim)).astype(DTYPE))
        
        _dense = lambda dim: tf.keras.layers.Dense(units=dim, activation=None, use_bias=False)
        _bn = lambda : tf.keras.layers.BatchNormalization(
            momentum=.99, epsilon=1e-6,
            beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
            gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
        )
        
        self.gradui = []
        for _ in range(self.N - 1):
            this_grad = tf.keras.Sequential()
            this_grad.add(tf.keras.layers.Input(dim))
            this_grad.add(_bn())
            for _ in range(num_hidden_layers):
                this_grad.add(_dense(num_neurons))
                this_grad.add(_bn())
                this_grad.add(tf.keras.layers.ReLU())
            this_grad.add(_dense(dim))
            this_grad.add(_bn())
            self.gradui.append(this_grad)
    
    def draw_X_and_dW(self, num_sample):
        dW = np.random.normal(loc=0.0, scale=self.sqrt_dt, size=(num_sample, self.dim, self.N)).astype(DTYPE)
        X = np.zeros((num_sample, self.dim, self.N+1), dtype=DTYPE)
        X[:, :, 0] = np.ones((num_sample, self.dim)) * self.x
        for i in range(self.N):
            X[:, :, i+1] = X[:, :, i] + self.sigma * dW[:, :, i]
        return X, dW

    @tf.function
    def call_all(self, inp):
        y_list = []
        X, dW = inp
        num_sample = X.shape[0]
        e_num_sample = tf.ones(shape=[num_sample, 1], dtype=DTYPE)
        y = e_num_sample * self.u0
        z = e_num_sample * self.gradu0
        for i in range(self.N-1):
            t = self.t_space[i]
            eta1 = - self.fun_f(t, X[:, :, i], y, z) * self.dt
            eta2 = tf.reduce_sum(z * dW[:, :, i], axis=1, keepdims=True)
            y = y + eta1 + eta2
            z = self.gradui[i](X[:, :, i + 1], training=False) / self.dim
            y_list.append(y)
        eta1 = - self.fun_f(self.t_space[self.N-1], X[:, :, self.N-1], y, z) * self.dt
        eta2 = tf.reduce_sum(z * dW[:, :, self.N-1], axis=1, keepdims=True)
        y = y + eta1 + eta2
        y_list.append(y)
        return y_list
    
    @tf.function
    def call(self, inp, training=False):
        X, dW = inp
        num_sample = X.shape[0]
        e_num_sample = tf.ones(shape=[num_sample, 1], dtype=DTYPE)
        y = e_num_sample * self.u0
        z = e_num_sample * self.gradu0
        for i in range(self.N-1):
            t = self.t_space[i]
            eta1 = - self.fun_f(t, X[:, :, i], y, z) * self.dt
            eta2 = tf.reduce_sum(z * dW[:, :, i], axis=1, keepdims=True)
            y = y + eta1 + eta2
            z = self.gradui[i](X[:, :, i + 1], training) / self.dim
        eta1 = - self.fun_f(self.t_space[self.N-1], X[:, :, self.N-1], y, z) * self.dt
        eta2 = tf.reduce_sum(z * dW[:, :, self.N-1], axis=1, keepdims=True)
        y = y + eta1 + eta2
        return y
    
    def loss_fn(self, inputs, training=False):
        X, _ = inputs
        y_pred = self.call(inputs, training)
        y = self.fun_g(X[:, :, -1])
        y_diff = y - y_pred
        loss = tf.reduce_mean(tf.square(y_diff))
        return loss

    @tf.function
    def train(self, inp):
        loss, grad = self.grad(inp, training=True)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss

    @tf.function
    def grad(self, inputs, training=False):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.trainable_variables)
        return loss, grad
    
    def fun_f(self, t, x, y, z):
        raise NotImplementedError
    
    def fun_g(self, t, x, y, z):
        raise NotImplementedError

class HJBSolver(DeepBSDE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.y_star = 4.839719451613728

    @tf.function
    def fun_f(self, t, x, y, z):
        return - tf.reduce_sum(tf.square(z), axis=1, keepdims=True) / (self.sigma**2)

    @tf.function
    def fun_g(self, t, x, y, z):
        raise NotImplementedError
    
    def estimate_solution(self, mc_iter=1e5, N_runs=200, replace=False):
        N_total = 0
        total_mean = 0
        old_est = 0
        tf.random.set_seed(0)
        for _ in range(int(N_runs)):
            W = tf.random.normal((int(mc_iter), self.dim),mean=0.0, stddev=tf.sqrt(self.t1-self.t0))
            X_T = self.x + self.sigma * W
            this_mean = tf.math.reduce_mean(tf.exp(-self.fun_g(X_T)))
            total_mean = (N_total * total_mean + mc_iter * this_mean) / (N_total + mc_iter)
            N_total += mc_iter
            total_est = - tf.math.log(total_mean)
            est_diff = np.abs(total_est-old_est)
            print('Current estimate: ', total_est.numpy(), '\tDiff: ', est_diff)
            old_est = total_est
        if replace:
            self.y_star = total_est
        return total_est



