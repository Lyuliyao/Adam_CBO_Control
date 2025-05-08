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
    def smooth_penalty(distance: jnp.ndarray, radius: float) -> jnp.ndarray:
        soft_zone = (distance > radius) & (distance <= radius * 1.2)
        hard_penalty = (distance <= radius).astype(jnp.float64)
        soft_penalty = soft_zone.astype(jnp.float64) * (
            0.5 + 0.5 * jnp.cos(jnp.pi * (distance - radius) / (radius * 0.2))
        )
        return hard_penalty + soft_penalty

    def fcn_f(x: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
        obstacle_position = jnp.array([2.5, 0.0])
        obstacle_radius = 1.2

        x1, y1 = x[:, 0], x[:, 1]
        x2, y2 = x[:, 2], x[:, 3]

        dis1 = jnp.linalg.norm(jnp.stack([x1 - obstacle_position[0], y1 - obstacle_position[1]], axis=-1), axis=-1)
        dis2 = jnp.linalg.norm(jnp.stack([x2 - obstacle_position[0], y2 - obstacle_position[1]], axis=-1), axis=-1)

        loss = jnp.sum(m**2, axis=-1)
        loss += smooth_penalty(dis1, obstacle_radius)
        loss += smooth_penalty(dis2, obstacle_radius)

        return loss
    
    # def fcn_f(x: jnp.ndarray,m: jnp.ndarray) -> jnp.ndarray:
    #     obstacle_position = [2.5, 0.0]
    #     obstacle_radius = 1.5
    #     x1 = x[:,0]
    #     y1 = x[:,1]
    #     x2 = x[:,2] 
    #     y2 = x[:,3]
    #     dis1 = jnp.sqrt((x1-obstacle_position[0])**2 + (y1-obstacle_position[1])**2)
    #     dis2 = jnp.sqrt((x2-obstacle_position[0])**2 + (y2-obstacle_position[1])**2)
    #     loss = jnp.sum(m**2, axis=-1)
    #     loss += (dis1<=obstacle_radius).astype(jnp.float64)
    #     # loss += (dis1>obstacle_radius).astype(jnp.float64)*(dis1<=obstacle_radius*1.2).astype(jnp.float64)*(0.5+ 0.5*jnp.cos(jnp.pi*(dis1-obstacle_radius)/(obstacle_radius*0.2)))
    #     loss += (dis2<=obstacle_radius).astype(jnp.float64)
    #     # loss += (dis2>obstacle_radius).astype(jnp.float64)*(dis2<=obstacle_radius*1.2).astype(jnp.float64)*(0.5+ 0.5*jnp.cos(jnp.pi*(dis2-obstacle_radius)/(obstacle_radius*0.2)))
    #     return loss
    x_target = jnp.array([5.0, 1.0 , 5.0, -1.0])
    # def fcn_g(x: jnp.ndarray) -> jnp.ndarray:
    #     return jnp.log((1 + jnp.sum((x-x_target[None,:])**2, axis=-1)) / 2)
    def fcn_g(x: jnp.ndarray) -> jnp.ndarray:
        return 10*jnp.sum((x-x_target[None,:])**2, axis=-1)
    x_start = jnp.array([0.0, 1.0 , 0.0, -1.0])
    sde_configure = {
        "fcn_f": fcn_f,
        "fcn_g": fcn_g,
        "x_start": x_start,
        "T1": problem_configure["T1"],
        "T0": problem_configure["T0"],
        "N_step": 20,
        "N_sample": 500,
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
