import control_han
import tensorflow as tf
import argparse
import numpy as np
from time import time

parser = argparse.ArgumentParser(description="Run HJB Solver with custom settings.")
parser.add_argument('--dim', type=int, default=5, help='Dimension of the problem.')
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
            return tf.math.log( (1+ tf.square(tf.reduce_sum( tf.square(x), axis=1, keepdims=True)-3) ))
        elif args.case == 'case4':
            return tf.math.log( 1+ tf.reduce_sum(tf.square(x) - 10*tf.math.cos(x)+10 , axis=1, keepdims=True))
        elif args.case == 'case5':
            return tf.math.log( (1+ tf.square(tf.reduce_sum( tf.square(x), axis=1, keepdims=True)-5) ))
t1 = 1
batch_size = 64
lr = 1e-3
num_neurons = 110
num_time_steps = 20
num_hidden_layers = 2
suffix = 'orig'

dim = args.dim
History = []
model = CustomHJBSolver(t1=t1,
                time_steps=num_time_steps,
                dim=dim,
                learning_rate=lr,
                num_neurons=num_neurons,
                num_hidden_layers=num_hidden_layers)


def experiment(model, num_epochs=2000):
    # Initialize header
    print('  Iter        Loss        y   L1_rel    L1_abs   |   Time  Stepsize')
    
    # Init timer and history list
    t0 = time()
    history=[]
    for i in range(num_epochs):
        
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
    return history

model.estimate_solution(replace=True)
history = experiment(model, num_epochs=50000)

History.append(history)
X=np.stack(History)
epochs = X[0,:,0].astype(int)

# Derive statistics for table
Loss_mean = np.mean(X[:,:,1], axis=0)
Loss_std = np.std(X[:,:,1], axis=0)
y_mean = np.mean(X[:,:,2], axis=0)
y_std = np.std(X[:,:,2], axis=0)
y_exact = model.y_star.numpy()*np.ones_like(y_mean)
L1rel_mean = np.mean(X[:,:,3], axis=0)
L1rel_std = np.std(X[:,:,3], axis=0)
L1abs_mean = np.mean(X[:,:,4], axis=0)
L1abs_std = np.std(X[:,:,4], axis=0)
Time_mean = np.mean(X[:,:,5], axis=0)
Time_std = np.std(X[:,:,5], axis=0)

# Write table
SAVE_CSV = True
if SAVE_CSV:
    import csv
    with open('case_{:s}_hjb_dim_{:03d}.csv'.format(args.case,dim),'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['Iter', 'LossMean', 'LossStd', 'yMean', 'yStd', 'yExact'
                            , 'L1relMean', 'L1relStd', 'L1absMean', 'L1absStd', 'TimeMean','TimeStd'])
        for i in epochs:
            csv_out.writerow((i, Loss_mean[i], Loss_std[i],
                            y_mean[i], y_std[i], y_exact[i],
                            L1rel_mean[i], L1rel_std[i],
                            L1abs_mean[i], L1abs_std[i], Time_mean[i], Time_std[i]))
