import jax
import jax.numpy as jnp
from typing import Callable, MutableMapping, Optional, Tuple, Sequence
import chex
import pdb

def create_cbo(
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-2,
    kappa_l: float = 100,
    gamma: float = 1,
    learning_rate: float = 1e-2,
):
    """
    Creates the initialization and update functions for an Adam-based optimizer
    with Cooperative Bayesian Optimization (CBO).

    Args:
        beta1: Exponential decay rate for the first moment estimate.
        beta2: Exponential decay rate for the second moment estimate.
        epsilon: Small constant for numerical stability.
        kappa_l: Scaling factor for the low variance weights.
        gamma: Momentum factor.
        learning_rate: Base learning rate.

    Returns:
        A tuple containing:
        - init: Initialization function.
        - update_adam_cbo: Update function for the original Adam-CBO variant.
    """
    
    def init(params: Sequence[jnp.ndarray],
             rng_key: jnp.ndarray) -> MutableMapping[str, jnp.ndarray]:
        M_params = jax.tree_map(lambda x: jnp.zeros_like(jnp.mean(x, axis=0)), params)
        V_params = jax.tree_map(lambda x: jnp.zeros_like(jnp.var(x, axis=0)), params)
        Vel_params = jax.tree_map(jnp.zeros_like, params)
        sigma = 1
        coeff = {
            "rng_key": rng_key,
            "step": 1,
            "learning_rate": learning_rate,
            "M": M_params,
            "V": V_params,
            "Vel": Vel_params,
            "kappa_l": kappa_l,
            "sigma": sigma,
        }
        return coeff

    def update_params(
        params: Sequence[jnp.ndarray],
        params_perm: Sequence[jnp.ndarray],
        weight: jnp.ndarray,
        coeff: MutableMapping[str, jnp.ndarray],
    ):
        step = coeff["step"]
        kappa_l = coeff["kappa_l"]
        Vel = coeff["Vel"]
        learning_rate = coeff["learning_rate"]
        sigma = coeff["sigma"]
        rng, key = jax.random.split(coeff["rng_key"])
        min_weight = jax.lax.pmin(jnp.min(weight),axis_name="batch")

        weight = weight-min_weight
        
        exp_weight = jnp.exp(-kappa_l*weight)
        exp_weight_norm = jax.lax.psum(jnp.sum(exp_weight),axis_name="batch")
        normalized_values = exp_weight/exp_weight_norm
        x_start = jax.tree_map(lambda x: jax.lax.psum(jnp.sum(jax.vmap(lambda a, b: a * b )
                                                              (normalized_values, x), axis=0) ,axis_name="batch"), params_perm)
        M_params = jax.tree_map(lambda x , y : beta1*x + (1-beta1)*y  ,coeff["M"],x_start)
        M_params_hat = jax.tree_map(lambda x: x/(1-beta1**step), M_params)
        x_start2 = jax.tree_map(lambda x , y : jax.lax.psum(jnp.sum(jax.vmap(lambda a, b, c: a * (b-c)**2,(0,0,None))
                                                       (normalized_values, x, y), axis=0),axis_name="batch"), params_perm, x_start) 
        V_params = jax.tree_map(lambda x , y : beta2*x + (1-beta2)*y
                                ,coeff["V"],x_start2)
        V_params_hat = jax.tree_map(lambda x: x/(1-beta2**step), V_params)
        Vel = jax.tree_map(lambda v, x, y, z: v - learning_rate * (x- y) / 
                           ( z + epsilon) + 
                           - learning_rate* gamma * v +
                              sigma*jnp.sqrt(2*learning_rate*gamma)*jax.random.normal(key, x.shape)
                              , Vel, params, M_params_hat, V_params_hat)
        params = jax.tree_map(lambda x, y: x + learning_rate* y, params, Vel)
        
        
        coeff["step"] = step + 1
        coeff["Vel"] = Vel
        coeff["M"] = M_params   
        coeff["V"] = V_params
        coeff["rng_key"] = rng
        return params, coeff
    
    update_adam_cbo = {
        # "fcn_update_coeff": updata_coeff,
        "fcn_update_params": update_params,
    }
    return init ,  update_adam_cbo



 
    # def update_adam_cbo2(
    #     params: Sequence[jnp.ndarray],
    #     weight: jnp.ndarray,
    #     coeff: MutableMapping[str, jnp.ndarray],
    # ):
    #     step = coeff["step"]
    #     kappa_l = coeff["kappa_l"]
    #     Vel = coeff["Vel"]
    #     learning_rate = coeff["learning_rate"]
    #     sigma = coeff["sigma"]
    #     rng, key = jax.random.split(coeff["rng_key"])
    #     weight = weight-jnp.min(weight)
    #     normalized_values = jnp.exp(-kappa_l*weight)/jnp.sum(jnp.exp(-kappa_l*weight)) 
    #     x_start = jax.tree_map(lambda x: jnp.sum(jax.vmap(lambda a, b: a * b )(normalized_values, x), axis=0), params)
    #     M_params = jax.tree_map(lambda x , y : beta1*x + (1-beta1)*y  ,coeff["M"],x_start)
    #     M_params_hat = jax.tree_map(lambda x: x/(1-beta1**step), M_params)
    #     x_start2 = jax.tree_map(lambda x , y : jnp.sum(jax.vmap(lambda a, b, c: a * (b-c)**2,(0,0,None)) (normalized_values, x, y), axis=0), params, M_params_hat)
    #     V_params = jax.tree_map(lambda x , y : beta2*x + (kappa_l+kappa_h)*(1-beta2)*y  ,coeff["V"],x_start2)
    #     V_params_hat = jax.tree_map(lambda x: x/(1-beta2**step), V_params)
    #     params = jax.tree_map(lambda  x, y, z: x - learning_rate * (x- y) / ( z + epsilon) /gamma +
    #                           sigma*jnp.sqrt(2*learning_rate/kappa_h/gamma)*jax.random.normal(key, x.shape)
    #                           , params, M_params_hat, V_params_hat)
    #     coeff["step"] = step + 1
    #     coeff["Vel"] = Vel
    #     coeff["M"] = M_params   
    #     coeff["V"] = V_params
    #     coeff["rng_key"] = rng
    #     return params, coeff
        