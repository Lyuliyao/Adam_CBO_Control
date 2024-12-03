import jax
import jax.numpy as jnp
from typing import Callable, Tuple
import logging
from NN import create_nn
from optim import create_cbo
import numpy as np
from gen_config import generate_configure
import argparse
from scipy.stats import binned_statistic_2d,binned_statistic
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description="Run HJB Solver with custom settings.")
parser.add_argument('--N_mv', type=int, default=5, help='Dimension of the problem.')
args = parser.parse_args()

dim = 1
config  = generate_configure(dim)
problem_configure = config['problem']
init, apply = create_nn(1,**config["NN"])
config["sde"]["N_step"] = 100
config["sde"]["N_sample"] = 1000
config["sde"]["N_mv"] =  args.N_mv
N_mv = config["sde"]["N_mv"]



# Compute the loss function
def generate_control_loss(
    fcn_g: Callable =  lambda x : x,
    fcn_f: Callable = lambda x : x,
    x_start: jnp.ndarray = jnp.zeros(2),
    N_step: int = 10,
    N_sample: int = 10,
    N_mv: int = 10,
    problem_configure: dict = None,
):
    
    T0 = problem_configure["T0"]
    T1 = problem_configure["T1"]
    dim = problem_configure["dim"]
    k = problem_configure["k"]
    q = problem_configure["q"]
    
    


    def mixture_two_gaussian(key):
        k = jnp.sqrt(3)/ 10
        theta = 0.1
        p = 0.5  # Bernoulli parameter
        key, subkey = jax.random.split(key)
        P = jax.random.bernoulli(subkey, p)
        key, subkey = jax.random.split(key)
        Y = jax.random.normal(subkey)

        key, subkey = jax.random.split(key)
        Y_bar = jax.random.normal(subkey)
        X0 = P * (-k + theta * Y) + (1 - P) * (k + theta * Y_bar)
        return X0
    
    def compute_loss(
        rng: jax.random.PRNGKey, 
        params: dict, 
        x_start_new: jnp.ndarray = jnp.zeros(2), 
        T0_new: float = 0.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # x = x_start_new[None, None, ...].repeat(N_sample, axis=0).repeat(N_mv, axis=1)
        keys = jax.random.split(rng, (N_sample, N_mv, dim))
        x_start_new = jax.vmap(jax.vmap(jax.vmap(mixture_two_gaussian)))(keys)
        x = jnp.copy(x_start_new)
        t = jnp.linspace(T0_new, T1, N_step + 1).reshape(-1, 1)
        dt = t[1] - t[0]
        loss = jnp.zeros(N_sample)
        for i in range(N_step):
            rng, key = jax.random.split(rng)
            t_current = t[i][None, None, ...].repeat(N_sample, axis=0).repeat(N_mv, axis=1)
            input = jnp.concatenate([x, t_current], axis=-1)
            dis = apply(params["distribution"], input)
            dis_mean = jnp.mean(dis, axis=1,keepdims=True).repeat(N_mv, axis=1)
            input = jnp.concatenate([dis_mean, x, t_current], axis=-1)
            m = apply(params["control"], input)
            x_mean = x.mean(axis=1, keepdims=True)
            Qt = problem_configure["fcn_Q"](t[i])
            m2 = (q+2*Qt)*(x_mean-x)
            loss += fcn_f(x,m) * dt
            if i == 0:
                m0 = jnp.copy(m)
                m1 = jnp.copy(m2)
            x = x + dt *( k*(x_mean - x) + m) + jnp.sqrt(dt)  * jax.random.normal(key, shape=(N_sample, N_mv, dim))
            # x = x + 2 * dt * m + jnp.sqrt(2 * dt) * jax.random.normal(rng, shape=(N_sample, dim))
        loss += fcn_g(x)
        return loss,x_start_new,m0,m1
    
    return compute_loss





compute_loss = generate_control_loss(**config["sde"])
params_old = init(jax.random.PRNGKey(0))



params= np.load(f"./result_{dim}/params.npy", allow_pickle=True)
params_new_all = {}
for key_params in params.any().keys():
    params_new = []
    print(key_params)
    for params_i in params.any()[key_params]:
        for key in params_i.keys():
            params_i[key] = params_i[key][0,0,...]
        params_new.append(params_i)
    params_new_all[key_params] = params_new
rng = jax.random.PRNGKey(100)


x = np.linspace(0,1,10)
value_test1,x_start_new,m0,m1 =  jax.vmap(lambda x: compute_loss(rng,params_new_all, jnp.zeros(1), x))(x)
value_test1 = value_test1.mean(axis=-1)
value_test2 = problem_configure["fcn_Qs"](x)*N_mv+N_mv*problem_configure["fcn_Q"](x)*0.04
np.savez(f"./result_{dim}/value_test_mv_{N_mv}_gaussian_2.npz",value_test1=value_test1,value_test2=value_test2)


