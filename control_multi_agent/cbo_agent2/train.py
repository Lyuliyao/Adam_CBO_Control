# import os
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=100"
import jax
import jax.numpy as jnp
from typing import Callable, Tuple
import logging
from NN import create_nn
from optim import create_cbo
import numpy as np
from gen_config import generate_configure
import pdb
from functools import partial
import argparse

jax.config.update("jax_enable_x64", True)
# Define the target function

parser = argparse.ArgumentParser(description="Run HJB Solver with custom settings.")
parser.add_argument('--dim', type=int, default=5, help='Dimension of the problem.')
args = parser.parse_args()

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
    
    def compute_loss(rng: jax.random.PRNGKey, params: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = (x_start+ 0.1*jax.random.normal(rng, shape=(dim,)))[None, ...].repeat(N_sample, axis=0)
        t = jnp.linspace(T0, T1, N_step+1).reshape(-1, 1)
        dt = t[1] - t[0]
        loss = jnp.zeros(N_sample)
        for i in range(N_step ):
            rng, key = jax.random.split(rng)
            t_current = t[i][None, ...].repeat(N_sample, axis=0)
            m = apply(params, jnp.concatenate([x, t_current], axis=-1))
            loss += fcn_f(x,m) * dt
            x = x + 10 * dt * m + 0.1*jnp.sqrt(dt) * jax.random.normal(rng, shape=(N_sample, dim))
        loss += fcn_g(x)
        return loss
    
    return compute_loss

# Create parameter update function
def create_params_update(update_fn,
                         N_iteration: int = 1,
                         N_CBO_sampler: int = 10,
                         N_CBO_batch: int = 10,
                         ) -> Tuple[Callable, Callable]:
    
    def compute_error(params: dict, rng: jax.random.PRNGKey) -> jnp.ndarray:
        total_error = 0.0
        for _ in range(N_iteration):
            rng, key = jax.random.split(rng)
            error = compute_loss(key, params)
            total_error += error
        return jnp.mean(total_error / N_iteration)
        
    # def compute_error_batch(params: dict, rng: jax.random.PRNGKey) -> jnp.ndarray:
    #     rng_perm = jax.random.permutation(rng, N_CBO_sampler)
    #     params_perm = jax.tree_map(lambda x: jnp.take(x, rng_perm[:N_CBO_batch], axis=0), params)
    #     rng, key = jax.random.split(rng)
    #     value = jax.vmap(compute_error, (0, None))(params_perm, rng)
    #     return value, params_perm
    
    def compute_error_all(params: dict, rng: jax.random.PRNGKey) -> jnp.ndarray:
        rng, key = jax.random.split(rng)
        key  = jax.random.split(key, N_CBO_sampler_per_device)
        error = jax.vmap(compute_error)(params, key)
        return error
    
    def params_update(i: int, input: Tuple[dict, dict, jax.random.PRNGKey, jnp.ndarray]) -> Tuple[dict, dict, jax.random.PRNGKey, jnp.ndarray]:
        params, coeff, rng, _ = input
        rng, key = jax.random.split(rng)
        rng_perm = jax.random.permutation(key, N_CBO_sampler)
        params_perm = jax.tree_map(lambda x: jnp.take(x, rng_perm[:N_CBO_batch], axis=0), params)
        rng, key = jax.random.split(rng)
        value = jax.vmap(compute_error, (0, None))(params_perm, key)
        params, coeff = update_fn["fcn_update_params"](params, params_perm, value, coeff)
        return params, coeff, rng, value
    
    return params_update, compute_error_all, compute_error

N_CBO_sampler = config["optimizer"]["N_CBO_sampler"]
N_CBO_batch = config["optimizer"]["N_CBO_batch"]

rng = jax.random.PRNGKey(config["seed"])

if N_CBO_sampler % n_devices != 0:
    logger.error(f"Number of samples ({N_CBO_sampler}) must be divisible by the number of devices ({n_devices})")
    raise ValueError("Number of samples must be divisible by the number of devices")
if N_CBO_batch % n_devices != 0:
    logger.error(f"Number of samples ({N_CBO_batch}) must be divisible by the number of devices ({n_devices})")
    raise ValueError("Number of samples must be divisible by the number of devices")

N_CBO_sampler_per_device = N_CBO_sampler // n_devices
N_CBO_batch_per_device = N_CBO_batch // n_devices



compute_loss = generate_control_loss(**config["sde"])

# Initialize neural network and parameters
init, apply = create_nn(N_CBO_sampler_per_device,**config["NN"])
key = jax.random.split(rng, n_devices)
params = jax.pmap(lambda key: init(key))(key)
rng, key = jax.random.split(rng)

# Initialize optimizer and parameters update function
optim_init, update_adam_cbo = create_cbo(**config["optimizer"]["CBO_configure"])
key = jax.random.split(rng, n_devices)
coeff = jax.pmap(optim_init)(params, key)
rng, key = jax.random.split(rng)
params_update, compute_error_all, compute_error = create_params_update(
    update_adam_cbo, 
    N_iteration =1, 
    N_CBO_sampler=N_CBO_sampler_per_device, 
    N_CBO_batch=N_CBO_batch_per_device
    )

# Training loop
N_iteration = config["optimizer"]["N_iteration"]//config["optimizer"]["N_print"]
value = jnp.zeros(N_CBO_batch).reshape(n_devices, -1)
unc = 0

def batch_iteration(params, coeff, rng, value):
    params, coeff, rng, value = jax.lax.fori_loop(0, config["optimizer"]["N_print"], params_update, (params, coeff, rng, value))
    return params, coeff, rng, value

value_save = np.zeros((N_iteration, n_devices))
for i in range(N_iteration):
    rng_pmap = jax.random.split(rng, n_devices)
    params, coeff, rng_pmap, value = jax.pmap(batch_iteration, axis_name='batch')(params, coeff, rng_pmap, value)
    rng, key = jax.random.split(rng)
    key = jax.random.split(key, n_devices)
    value_test1 = jax.pmap(compute_error_all)(params, key)
    rng, key = jax.random.split(rng)
    key = jax.random.split(key, n_devices)
    value_test2 = jax.pmap(compute_error_all)(params, key)
    value_test1 = value_test1.reshape(-1)
    value_test2 = value_test2.reshape(-1)
    values = jnp.abs(value_test2 + value_test1)/2
    errors = jnp.abs(value_test2 - value_test1)
    value_relative = jnp.max(values)
    unc = jnp.mean(errors) / value_relative
    logger.info(f"Uncertainty: {unc}")
    
    coeff["sigma"] *= 0.99
    
    if coeff["kappa_l"][0] < 1e5:
        coeff["kappa_l"] /= 0.9
    
    logger.info(f"Relative loss: {jnp.max(values) - jnp.min(values)}")
    logger.info(f"Step: {coeff['step']}")
    logger.info(f"Kappa_l: {coeff['kappa_l']}")
    logger.info(f"Sigma: {coeff['sigma']}")
    logger.info(f"Learning rate: {coeff['learning_rate']}")
    logger.info(f"Loss: {jnp.mean(values)}")

    var = jax.pmap(lambda param: jax.tree_map(lambda y: jnp.max(y, axis=0) - jnp.min(y, axis=0), param))(params)
    var = jax.pmap(lambda param: jax.tree_map(lambda x: jnp.max(x), param))(var)
    value_save[i] = jnp.mean(values)
    
    if i % 10 == 0:
        np.save(f"{config['save_dir']}/value.npy", value_save)
        np.save(f"{config['save_dir']}/params.npy", params)
    
    logger.info(f"Variance: {var}")
