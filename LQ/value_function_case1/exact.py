import jax 
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import chex


def estimate_solution(
                      x0:float,
                      t0:float,
                      key:chex.PRNGKey,
                      mc_iter:int,
                      N_runs:int,
                      ):
    N_total = 0 
    total_mean = 0
    old_est = 0
    t1 = 1
    dim  = 1
    sigma = np.sqrt(2)
    for _ in range(int(N_runs)):
        key,subkey = jax.random.split(key)
        W = jax.random.normal(subkey, shape=(int(mc_iter), dim)) * jnp.sqrt(t1 - t0)
        X_T = x0 + sigma * W
        this_mean = jnp.mean(jnp.exp(-fun_g(X_T)))
        total_mean = (N_total * total_mean + mc_iter * this_mean) / (N_total + mc_iter)
        N_total += mc_iter
        total_est = -jnp.log(total_mean)
        old_est = total_est
    return total_est

def fun_g(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log((1 + jnp.sum(x**2, axis=-1)) / 2)



data = np.load("./data_hjb_1D_plot.npz")
rng = jax.random.PRNGKey(0)
res_est = jax.vmap(jax.vmap(lambda x , y : estimate_solution(x,y,rng,1000,1) ))(jnp.array(data["X"]),jnp.array(data["t"]))
np.savez(f'./data_hjb_1D_plot_est.npz', res=res_est)
