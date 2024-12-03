import jax
import jax.numpy as jnp
from typing import Callable, Tuple
import logging
from NN import create_nn
from optim import create_cbo
import numpy as np
from gen_config import generate_configure
import sys

# Load configuration
dim = int(sys.argv[1])
config = generate_configure(dim)

def generate_control_loss(
    fcn_g: Callable = lambda x: x,
    fcn_f: Callable = lambda x: x,
    x_start: jnp.ndarray = jnp.zeros(2),
    T1: float = 1.0,
    T0: float = 0.0,
    N_step: int = 10,
    N_sample: int = 10,
    dim: int = 2,
) -> Callable:
    lambd, mu = 0.2, 5
    z = np.linspace(0, 1, dim + 2)
    hz = z[1] - z[0]

    def compute_loss(
        rng: jax.random.PRNGKey, 
        params: dict, 
        x_start_new: jnp.ndarray = jnp.zeros(2), 
        T0_new: float = 0.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = x_start_new[None, ...].repeat(N_sample, axis=0)
        x_extend = jnp.concatenate([jnp.zeros((N_sample, 1)), x, jnp.zeros((N_sample, 1))], axis=-1)
        t = jnp.linspace(T0_new, T1, N_step + 1).reshape(-1, 1)
        dt = t[1] - t[0]
        loss = jnp.zeros(N_sample)
        
        for i in range(N_step):
            rng, key = jax.random.split(rng)
            t_current = t[i][None, ...].repeat(N_sample, axis=0)
            m = apply(params, jnp.concatenate([x, t_current], axis=-1))
            loss += fcn_f(m, x) * dt
            
            nabla_V = (
                lambd / hz * (2 * x - x_extend[:, :-2] - x_extend[:, 2:]) 
                - mu * x * hz + mu * hz * x ** 3
            )
            b = -nabla_V + 2 * m * ((z[1:-1] < 0.6) & (z[1:-1] > 0.25))
            x = x + dt * b + jnp.sqrt(2 * dt) * jax.random.normal(rng, shape=(N_sample, dim))
            x_extend = jnp.concatenate([jnp.zeros((N_sample, 1)), x, jnp.zeros((N_sample, 1))], axis=-1)
        
        loss += fcn_g(x)
        return loss
    
    return compute_loss

def create_params_update(
    update_fn: Callable, 
    N_iteration: int = 1, 
    N_CBO_sampler: int = 10, 
    N_CBO_batch: int = 10
) -> Tuple[Callable, Callable]:

    def compute_error(x_start_new: jnp.ndarray, t: float, params: dict, rng: jax.random.PRNGKey) -> jnp.ndarray:
        total_error = 0.0
        for _ in range(N_iteration):
            rng, key = jax.random.split(rng)
            error = compute_loss(key, params, x_start_new, t)
            total_error += error
        return jnp.mean(total_error / N_iteration)
    
    def compute_error_all(x_start_new: jnp.ndarray, t: float, params: dict, rng: jax.random.PRNGKey) -> jnp.ndarray:
        rng, key = jax.random.split(rng)
        error = jax.vmap(compute_error, (None, None, 0, None))(x_start_new, t, params, key)
        return error
    
    return compute_error_all

# Initialization
init, apply = create_nn(1, **config["NN"])
config["sde"]["N_step"] = 50
config["sde"]["N_sample"] = 5000

compute_loss = generate_control_loss(**config["sde"])
optim_init, update_adam_cbo = create_cbo(**config["optimizer"]["CBO_configure"])

compute_error_all = create_params_update(
    update_adam_cbo, 
    N_iteration=20, 
    N_CBO_sampler=1, 
    N_CBO_batch=1
)

params_old = init(jax.random.PRNGKey(0))
params = np.load(f"./result_{dim}/params.npy", allow_pickle=True)

params_new = []
for params_i in params:
    for key in params_i.keys():
        params_i[key] = params_i[key][0, 0:1, ...]
    params_new.append(params_i)

rng = jax.random.PRNGKey(100)

data = np.load(f"./result_{dim}/simulation_data.npz")["x_control"]
T0 = np.linspace(0, 1, data.shape[0] + 1)[1:][40:90]

@jax.jit
def value_fun(input):
    return jax.vmap(
        lambda z: jax.vmap(
            lambda x, y: compute_error_all(x, y, params_new, rng)
        )(z, T0), in_axes=1
    )(input)

@jax.jit
def fun_action(x, t_current, params):
    inputs = jnp.concatenate([x, t_current[..., None]], axis=-1)
    output = jax.vmap(apply, (0, None))(params, inputs)
    return output, inputs

@jax.jit
def action_fun(input):
    return jax.vmap(
        lambda z: jax.vmap(
            lambda x, y: fun_action(x, y, params_new)
        )(z, T0), in_axes=1
    )(input)

# Generate predictions
a_pred = np.zeros((50, 0))
action = np.zeros((50, 0))

for i in range(0, 1000, 10):
    X0 = data[40:90, 0, i:i + 10, :]
    value = value_fun(X0)
    z = np.linspace(0, 1, dim + 2)
    w = (z[1:-1] < 0.6) & (z[1:-1] > 0.25)
    
    a = np.zeros((X0.shape[0], X0.shape[1]))
    for i in range(dim):
        x_p = X0.copy()
        x_p[:, :, i] += 0.01
        value_p = value_fun(x_p)
        grad = (value_p - value) / 0.01
        a -= grad[..., 0].T * w[i]
    print(a.shape)
    a_pred = np.concatenate([a_pred, a], axis=1)
    a, inputs = action_fun(X0)
    action = np.concatenate([action, a[:, :, 0, 0].T], axis=1)
    np.savez(f"./result_{dim}/action_data.npz", action=action, a_pred=a_pred)
