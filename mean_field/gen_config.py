import numpy as np
import jax.numpy as jnp
import os
import jax
import pdb
def generate_configure(dim):
    problem_configure = {
        "dim": dim,
        "T1": 1.0,
        "T0": 0.0,
        "q": 0,
        "eta":2,
        "k":0.6,
        "c":2,
    }
    q = problem_configure["q"]
    eta = problem_configure["eta"]
    c = problem_configure["c"]
    k = problem_configure["k"]
    T1 = problem_configure["T1"]
    delta = (k+q)**2 + eta  - q**2
    sqrt_delta = jnp.sqrt(delta)
    
    def fcn_Q(t):
        Qt = sqrt_delta*jnp.sinh(sqrt_delta*(T1-t)) + (k+q+c)*jnp.cosh(sqrt_delta*(T1-t))
        Qt = Qt/(sqrt_delta*jnp.cosh(sqrt_delta*(T1-t)) + (k+q+c)*jnp.sinh(sqrt_delta*(T1-t)))
        Qt = -1/2*(k+q-sqrt_delta*Qt)
        return Qt
    
    def fcn_Qs(t):
        Qs = jnp.log(jnp.cosh(sqrt_delta*(T1-t)) + (k+q+c)*jnp.sinh(sqrt_delta*(T1-t))/sqrt_delta)
        Qs = 1/2*Qs - 1/2*(k+q)*(T1-t)
        return Qs
    problem_configure["fcn_Q"] = fcn_Q
    problem_configure["fcn_Qs"] = fcn_Qs
    
    def fcn_f(x: jnp.ndarray,
              m: jnp.ndarray) -> jnp.ndarray:
        x_mean = x.mean(axis=1, keepdims=True)
        loss = 1/2*(m**2).sum(axis=-1).sum(axis=-1)
        loss =  loss - (q * m * (x_mean-x)).sum(axis=-1).sum(axis=-1)
        loss =  loss + eta/2*((x_mean - x )**2).sum(axis=-1).sum(axis=-1)
        return loss

    def fcn_g(x: jnp.ndarray) -> jnp.ndarray:
        x_mean = x.mean(axis=1, keepdims=True)
        return 0.5*c*((x_mean-x)**2).sum(axis=-1).sum(axis=-1)

    sde_configure = {
        "fcn_f": fcn_f,
        "fcn_g": fcn_g,
        "x_start": jnp.zeros(problem_configure["dim"]),
        "N_step": 20,
        "N_sample": 64,
        "N_mv": 100,
        "problem_configure": problem_configure,
    }
    
    NN_configure = {
        "control": {
            "input_dim": problem_configure["dim"] + 1 + 10,
            "output_dim": problem_configure["dim"],
            "layers": [40 ,40 ,40],
        },
        "distribution": {
            "input_dim": problem_configure["dim"] + 1,
            "output_dim": 10 ,
            "layers": [10 ,10 ,10],
        },
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
