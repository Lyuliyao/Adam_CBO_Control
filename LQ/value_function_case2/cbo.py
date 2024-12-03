import sys
sys.path.append('../cbo_case2')

import jax
import jax.numpy as jnp
from typing import Callable, Tuple
import logging
from NN import create_nn
from optim import create_cbo
import numpy as np
from gen_config import generate_configure
import argparse
import pdb
# from jax.config import config
# config.update("jax_disable_jit", True)

parser = argparse.ArgumentParser(description="Run HJB Solver with custom settings.")
parser.add_argument('--dim', type=int, default=5, help='Dimension of the problem.')
args = parser.parse_args()
dim = args.dim
config  = generate_configure(args.dim)
# Set up logging
logging.basicConfig(level=getattr(logging, config["logging"]["log_level"].upper()))
logger = logging.getLogger(__name__)

devices = jax.devices()
n_devices = len(devices)
logger.info(f"Number of devices: {n_devices}")

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
):
    
    def compute_loss(rng: jax.random.PRNGKey, params: dict,x_start_new: jnp.ndarray = jnp.zeros(2), T0_new: float = 0.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_start_new  = x_start_new[None]
        x = x_start_new[None, ...].repeat(N_sample, axis=0)
        t = jnp.linspace(T0_new, T1, N_step).reshape(-1, 1)
        dt = t[1] - t[0]
        loss = jnp.zeros(N_sample)
        for i in range(N_step - 1):
            rng, key = jax.random.split(rng)
            t_current = t[i][None, ...].repeat(N_sample, axis=0)
            m = apply(params, jnp.concatenate([x, t_current], axis=-1))
            loss += fcn_f(m) * dt
            x = x + 2 * dt * m + jnp.sqrt(2 * dt) * jax.random.normal(rng, shape=(N_sample, dim))
        loss += fcn_g(x)
        return loss
    
    return compute_loss



# Create parameter update function
def create_params_update(update_fn,
                         N_iteration: int = 1,
                         N_CBO_sampler: int = 10,
                         N_CBO_batch: int = 10,
                         ) -> Tuple[Callable, Callable]:
    
        
    def compute_error(x_start_new: jnp.ndarray, t:float, params: dict, rng: jax.random.PRNGKey) -> jnp.ndarray:
        total_error = 0.0
        for _ in range(N_iteration):
            rng, key = jax.random.split(rng)
            error = compute_loss(key, params,x_start_new,t)
            total_error += error
        return jnp.mean(total_error / N_iteration)
    
    def compute_error_all(x_start_new: jnp.ndarray, t:float, params: dict, rng: jax.random.PRNGKey) -> jnp.ndarray:
        rng, key = jax.random.split(rng)
        error = jax.vmap(compute_error, (None, None, 0, None))(x_start_new,t, params, key)
        return error
    
    return compute_error_all




init, apply = create_nn(1,**config["NN"])
config["sde"]["N_step"] = 100
config["sde"]["N_sample"] = 100000

compute_loss = generate_control_loss(**config["sde"])
optim_init, update_adam_cbo = create_cbo(**config["optimizer"]["CBO_configure"])
compute_error_all = create_params_update(
    update_adam_cbo, 
    N_iteration =1, 
    N_CBO_sampler=1, 
    N_CBO_batch=1
    )
params_old = init(jax.random.PRNGKey(0))

params= np.load(f"../cbo_case2/result_{dim}/params.npy", allow_pickle=True)
params_new = []
for params_i in params:
    for key in params_i.keys():
        params_i[key] = params_i[key][0,0:1,...]
    params_new.append(params_i)
rng = jax.random.PRNGKey(100)

data = np.load("./data_hjb_1D_plot.npz")
X =  jnp.array(data["X"])
t =  jnp.array(data["t"])
value_test1 =  jax.vmap(jax.vmap(lambda x,y: compute_error_all(x,y,params_new, rng)))(X,t)
cbo_value = value_test1
np.savez(f"./cbo_hjb_1D.npz",cbo_value=cbo_value)
