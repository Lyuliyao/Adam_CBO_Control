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
    
    
        
    r = 1.0

    obstacles_one_petal = [
        ((2.0, 0.0), r),
        ((2.0 + np.sqrt(3)/2 * r, 0.5 * r), r),
        ((2.0 + np.sqrt(3)/2 * r, -0.5 * r), r),
    ]

    def rotate_point(p, angle, origin=(0, 0)):
        """Rotate point p around origin by angle (radians)."""
        ox, oy = origin
        px, py = p
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return (qx, qy)

    center = (7.5, 0.0)

    obstacles = []

    for i in range(5):
        angle = i * 2 * np.pi / 5 
        for (pos, radius) in obstacles_one_petal:
            rotated_pos = rotate_point(pos, angle, origin=center)
            obstacles.append((rotated_pos, radius))
            
    
    def smooth_penalty(distance: jnp.ndarray, radius: float) -> jnp.ndarray:
        soft_zone = (distance > radius) & (distance <= radius * 1.2)
        hard_penalty = (distance <= radius).astype(jnp.float64)
        soft_penalty = soft_zone.astype(jnp.float64) * (
            0.5 + 0.5 * jnp.cos(jnp.pi * (distance - radius) / (radius * 0.2))
        )
        return hard_penalty + soft_penalty
    @jax.jit
    def fcn_f(x: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
        loss = jnp.sum(m**2, axis=-1)
        x_reshaped = x.reshape(x.shape[0], -1, 2)  # shape: (N, num_points, 2)

        for pos, radius in obstacles:
            # pos: (2,), broadcast to (N, num_points, 2)
            diff = x_reshaped - jnp.array(pos)  # shape: (N, num_points, 2)
            dist = jnp.linalg.norm(diff, axis=-1)  # shape: (N, num_points)
            loss += jnp.sum(smooth_penalty(dist, radius), axis=-1)  # sum over points

        diff_center = x_reshaped - jnp.array(center)  # shape: (N, num_points, 2)
        dist_center = jnp.linalg.norm(diff_center, axis=-1)  # shape: (N, num_points)
        loss += jnp.sum(smooth_penalty(dist_center, 1.5 * r), axis=-1)  # sum over points
        return loss
    
    x_target = jnp.array([18.0, 2.0 , 16.0, 2.0, 16.0, -2.0 , 18.0, -2.0])
    # def fcn_g(x: jnp.ndarray) -> jnp.ndarray:
    #     return jnp.log((1 + jnp.sum((x-x_target[None,:])**2, axis=-1)) / 2)
    def fcn_g(x: jnp.ndarray) -> jnp.ndarray:
        return  jnp.log(1 + jnp.sum((x-x_target[None,:])**2, axis=-1)) 
    x_start = jnp.array([0.0, 3.0 , 0.0, 1.0, 0.0, -1.0, 0.0, -3.0])
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
        "layers": [5*problem_configure["dim"] ,5*problem_configure["dim"] ,5*problem_configure["dim"],5*problem_configure["dim"]],
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
