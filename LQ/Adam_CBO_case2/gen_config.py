import numpy as np
import jax.numpy as jnp
import os
import jax
def generate_configure(dim):
    problem_configure = {
        "dim": dim,
        "T1": 1.0,
        "T0": 0.0,
    }
    
    def fcn_f(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x**2, axis=-1)

    def fcn_g(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(1 + (jnp.sum(x**2, axis=-1) -1 )**2) 

    sde_configure = {
        "fcn_f": fcn_f,
        "fcn_g": fcn_g,
        "x_start": jnp.zeros(problem_configure["dim"]),
        "T1": problem_configure["T1"],
        "T0": problem_configure["T0"],
        "N_step": 20,
        "N_sample": 64,
        "dim": problem_configure["dim"],
    }
    
    NN_configure = {
        "input_dim": problem_configure["dim"] + 1,
        "output_dim": problem_configure["dim"],
        "layers": [5*problem_configure["dim"] ,5*problem_configure["dim"] ,5*problem_configure["dim"]],
        "activation": jax.nn.silu,
    }
    
    CBO_configure = {
        "learning_rate": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-3,
        "kappa_l": 100,
        "gamma": 1,
    }
    optimizer_configure = {
        "CBO_configure": CBO_configure,
        "N_iteration": 500000,
        "N_print": 500,
        "N_CBO_sampler": 5000,
        "N_CBO_batch": 100,
    }
    
    logging_configure = {
        "log_level": "INFO",
        "log_dir": "log",
        "log_file": "log.txt",
    }
    save_dir = f"result_{dim}"
    os.makedirs(logging_configure["log_dir"], exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    configure = {
        "seed": 100,
        "problem": problem_configure,
        "sde": sde_configure,
        "NN": NN_configure,
        "optimizer": optimizer_configure,
        "logging": logging_configure,
        "y_star":0.3994605939133122,
        "save_dir": save_dir,
    }
    return configure
