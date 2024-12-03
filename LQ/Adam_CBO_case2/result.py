import jax
import jax.numpy as jnp
from typing import Callable, Tuple
import logging
from NN import create_nn
from optim import create_cbo
import numpy as np
from gen_config import generate_configure
import argparse


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
    
    def compute_loss(rng: jax.random.PRNGKey, params: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = x_start[None, ...].repeat(N_sample, axis=0)
        t = jnp.linspace(T0, T1, N_step).reshape(-1, 1)
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
        error = jax.vmap(compute_error, (0, None))(params, key)
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




init, apply = create_nn(1,**config["NN"])
config["sde"]["N_step"] = 1000
config["sde"]["N_sample"] = 50000

compute_loss = generate_control_loss(**config["sde"])
optim_init, update_adam_cbo = create_cbo(**config["optimizer"]["CBO_configure"])
params_update, compute_error_all, compute_error = create_params_update(
    update_adam_cbo, 
    N_iteration =1, 
    N_CBO_sampler=1, 
    N_CBO_batch=1
    )
params_old = init(jax.random.PRNGKey(0))
# params_old



params= np.load(f"./result_{dim}/params.npy", allow_pickle=True)
params_new = []
for params_i in params:
    for key in params_i.keys():
        params_i[key] = params_i[key][0,0:1,...]
    params_new.append(params_i)
rng = jax.random.PRNGKey(100)
value_test1 =  compute_error_all(params_new, rng)


cbo_value = value_test1
print(cbo_value)
y_star = config["y_star"]
loss = np.load(f"./result_{dim}/value.npy", allow_pickle=True)
np.savez(f"./result_{dim}/result.npz", loss=loss, cbo_value=cbo_value, y_star=y_star)
