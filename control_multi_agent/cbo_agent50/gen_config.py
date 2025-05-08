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
    }
    
    
        
    obstacles = [
        (jnp.array([9.0, 15.0])/10, 2/10),
        (jnp.array([8.0, 9.0])/10, 1.8/10),
        (jnp.array([7.0, 3.0])/10, 2.0/10),
        (jnp.array([9.0, -2.0])/10, 1.2/10),
        (jnp.array([12.0, -6.0])/10, 1.7/10),
        (jnp.array([10.0, -13.0])/10, 1.5/10),
    ]
    
    def smooth_penalty(distance: jnp.ndarray, radius: float) -> jnp.ndarray:
        # soft_zone = (distance > radius) & (distance <= radius * 1.1)
        hard_penalty = (distance <= radius).astype(jnp.float64)
        # soft_penalty = soft_zone.astype(jnp.float64) * (
        #     0.5 + 0.5 * jnp.cos(jnp.pi * (distance - radius) / (radius * 0.1))
        # )
        return hard_penalty #+ soft_penalty
    @jax.jit
    def fcn_f(x: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
        loss = jnp.sum(m**2, axis=-1)
        x_reshaped = x.reshape(x.shape[0], -1, 2)  # shape: (N, num_points, 2)

        for pos, radius in obstacles:
            diff = x_reshaped - jnp.array(pos)  # shape: (N, num_points, 2)
            dist = jnp.linalg.norm(diff, axis=-1)  # shape: (N, num_points)
            loss += jnp.sum(smooth_penalty(dist, radius), axis=-1)  # sum over points
            
        return loss
    
    x_target = jnp.zeros((100))
    for i in range(10):
        for j in range(5):
            x_target  = x_target.at[10*i+2*j].set(20+10/2*j)
            x_target  = x_target.at[10*i+2*j+1].set(10/4*i-45/4)
    x_target = x_target/10
    
    # def fcn_g(x: jnp.ndarray) -> jnp.ndarray:
    #     return jnp.log((1 + jnp.sum((x-x_target[None,:])**2, axis=-1)) / 2)
    def fcn_g(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum((x-x_target[None,:])**2, axis=-1)
    x_start = jnp.zeros((100))
    for i in range(10):
        for j in range(5):
            x_start  = x_start.at[10*i+2*j].set(-10/3*j)
            x_start  = x_start.at[10*i+2*j+1].set(10/3*i-15)
    x_start = x_start/10
    sde_configure = {
        "fcn_f": fcn_f,
        "fcn_g": fcn_g,
        "x_start": x_start,
        "T1": problem_configure["T1"],
        "T0": problem_configure["T0"],
        "N_step": 50,
        "N_sample": 100,
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
        "N_CBO_sampler": 600,
        "N_CBO_batch": 200,
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
