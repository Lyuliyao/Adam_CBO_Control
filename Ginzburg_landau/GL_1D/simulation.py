import jax
import jax.numpy as jnp
from typing import Callable, Tuple
import logging
from NN import create_nn
from optim import create_cbo
import numpy as np
from gen_config import generate_configure
import argparse
import sys
dim  = 2
config  = generate_configure(dim)
# Set up loggingx

# Compute the loss function
def generate_control_loss(
    fcn_g: Callable =  lambda x : x,
    fcn_f: Callable = lambda x : x,
    x_start: jnp.ndarray = jnp.zeros(2),
    T1: float = 1.0,
    T0: float = 0.0,
    N_step: int = 10,
    N_sample: int = 10,
    dim: int = 2,
    control=True
):
    lambd = 0.2
    mu = 5
    z = np.linspace(0, 1, dim+2)
    hz = z[1] - z[0]
    def sde(rng: jax.random.PRNGKey, params: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = x_start[None, ...].repeat(N_sample, axis=0)
        x_extend = jnp.concatenate([jnp.zeros((N_sample, 1)),x, jnp.zeros((N_sample, 1))], axis=-1)
        t = jnp.linspace(T0, T1, N_step+1).reshape(-1, 1)
        dt = t[1] - t[0]
        loss = jnp.zeros(N_sample)
        for i in range(N_step):
            rng, key = jax.random.split(rng)
            t_current = t[i][None, ...].repeat(N_sample, axis=0)
            if control:
                m = apply(params, jnp.concatenate([x, t_current], axis=-1))
            else:
                m = 0
            nabla_V = lambd/hz * ( 2*x - x_extend[:, :-2] - x_extend[:, 2:] )  - mu *x*hz  + mu* hz* x**3
            b = -nabla_V +  2* m * (z[1:-1]<0.6)*(z[1:-1]>0.25)
            x = x +   dt * b + jnp.sqrt(2 * dt) * jax.random.normal(rng, shape=(N_sample, dim))
            x_extend = jnp.concatenate([jnp.zeros((N_sample, 1)),x, jnp.zeros((N_sample, 1))], axis=-1)
        return x
    return sde

init, apply = create_nn(1,**config["NN"])
config["sde"]["N_step"] = 100
config["sde"]["N_sample"] = 100000

params_old = init(jax.random.PRNGKey(0))
params= np.load(f"./result_{dim}/params.npy", allow_pickle=True)
params_new = []
for params_i in params:
    for key in params_i.keys():
        params_i[key] = params_i[key][0,0:1,...]
    params_new.append(params_i)
    
    
sde = generate_control_loss(**config["sde"],control=True)
rng = jax.random.PRNGKey(100)
x_control = jax.vmap(sde,(None,0))(rng, params_new)
sde = generate_control_loss(**config["sde"],control=False)
rng = jax.random.PRNGKey(100)
x_free = jax.vmap(sde,(None,0))(rng, params_new)
np.savez(f"./simulation_data.npz", x_control=x_control, x_free=x_free)
