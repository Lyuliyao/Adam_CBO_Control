import control_han
import tensorflow as tf
import argparse
import numpy as np
from time import time
from scipy.stats import binned_statistic_2d


parser = argparse.ArgumentParser(description="Run HJB Solver with custom settings.")
parser.add_argument('--case', type=str, required=True, help='Choose the case for fun_g.')
args = parser.parse_args()



class CustomHJBSolver(control_han.HJBSolver):
    @tf.function
    def fun_g(self, x):
        if args.case == 'case1':
            return tf.math.log( (1+tf.reduce_sum(tf.square(x), axis=1, keepdims=True)) / 2)
        elif args.case == 'case2':
            return tf.math.log( (1+ tf.square(tf.reduce_sum( tf.square(x), axis=1, keepdims=True)-1) ))
        elif args.case == 'case3':
            return tf.math.log( (1+ tf.math.sqrt(tf.reduce_max(tf.square(x), axis=1, keepdims=True)))/ 2)
        elif args.case == 'case4':
            return tf.math.log( 1+ tf.reduce_sum(tf.square(x) - 10*tf.math.cos(x)+10 , axis=1, keepdims=True))
        elif args.case == 'case5':
            return tf.math.log( (1+ tf.square(tf.reduce_sum( tf.square(x), axis=1, keepdims=True)-5) ))


t1 = 1
batch_size = 64
lr = 1e-2
num_neurons = 110
num_time_steps = 20
num_hidden_layers = 2
suffix = 'orig'

dim = 1
History = []
model = CustomHJBSolver(t1=t1,
                time_steps=num_time_steps,
                dim=dim,
                learning_rate=lr,
                num_neurons=num_neurons,
                num_hidden_layers=num_hidden_layers)


print('  Iter        Loss        y   L1_rel    L1_abs   |   Time  Stepsize')

# Init timer and history list
t0 = time()
history=[]
for i in range(50000):
    
    inp = model.draw_X_and_dW(batch_size)
    loss = model.train(inp)

    # Get current Y_0 \approx u(0,x)
    y = model.u0.numpy()[0]

    currtime = time() - t0
    l1abs = np.abs(y - model.y_star)
    l1rel = l1abs / model.y_star

    hentry = (i, loss.numpy(), y, l1rel, l1abs, currtime, lr)
    history.append(hentry)
    if i%100 == 0:
        print('{:5d} {:12.4f} {:8.4f} {:8.4f}  {:8.4f}   | {:6.1f}  {:6.2e}'.format(*hentry))


inp = model.draw_X_and_dW(500000)
res = model.call_all(inp)
X = inp[0]
res  = np.array(res)

X_reshaped = X[:,:,:-1].squeeze() 
t = np.linspace(0, 1, 21)  # 20 steps between 0 and 1
t = t[:-1]  # Remove last element
res_reshaped = res.squeeze()  # Shape will be (20, 100)
X_flat = X_reshaped.flatten()  # Shape will be (100*20,)
t_flat = np.tile(t, X.shape[0])  # Shape will be (100*20,)
res_flat = res_reshaped.T.flatten()  # Shape will be (100*20,)



# Create bins in the X and t dimensions
x_bins = np.arange(-4, 4, 0.1)
t_bins = np.linspace(0.05, 1, 20)

statistic, x_edges, t_edges, _ = binned_statistic_2d(X_flat, t_flat, res_flat, statistic='mean', bins=[x_bins, t_bins])

# Generate the meshgrid for contour plotting
X_mesh, T_mesh = np.meshgrid(x_edges[:-1], t_edges[:-1], indexing='ij')

np.save(f'./data_hjb_1D_plot_{args.case}.npz', {'X': X_mesh, 't': T_mesh, 'res': statistic})