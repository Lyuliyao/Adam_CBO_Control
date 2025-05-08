import numpy as np
import jax.numpy as jnp
import os
import jax
import pdb
import gymnasium as gym
import gymnax
from sbx import DDPG

def generate_configure():
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_act, key_step = jax.random.split(rng, 4)
    env, env_params = gymnax.make("Pendulum-v1")
    obs, state = env.reset(key_reset, env_params)
    action = env.action_space(env_params).sample(key_act)
    n_obs, n_state, reward, done, _ = env.step(key_step, state, action, env_params)
    # model = DDPG("MlpPolicy", env, tensorboard_log="./ddpg_logs",verbose=1)
    # obs,info = env.reset()
    # action, _ = model.predict(obs, deterministic=True)
    # obs, reward, _, _, _ = env.step(action)
    
    sde_configure = {
        "env": env,
        "env_params": env_params,
        "N_sample": 100,
    }
    
    NN_configure = {
        "input_dim": obs.shape[0],
        "output_dim": action.shape[0],
        "layers": [128,128],
        "activation": jnp.tanh,
    }
    
    CBO_configure = {
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-3,
        "kappa_l": 100,
        "gamma": 1,
    }
    optimizer_configure = {
        "CBO_configure": CBO_configure,
        "N_iteration": 500_000,
        "N_print": 500,
        "N_CBO_sampler": 5000,
        "N_CBO_batch": 100,
    }
    
    logging_configure = {
        "log_level": "INFO",
        "log_dir": "log",
        "log_file": "log.txt",
    }
    save_dir = f"result"
    os.makedirs(logging_configure["log_dir"], exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    configure = {
        "seed": 100,
        "sde": sde_configure,
        "NN": NN_configure,
        "optimizer": optimizer_configure,
        "logging": logging_configure,
        "y_star":0.3994605939133122,
        "save_dir": save_dir,
    }
    return configure
