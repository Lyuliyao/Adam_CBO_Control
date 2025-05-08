import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Sequence, Optional
import chex
import numpy as np
import pdb

def init_linear_layer(
    key: chex.PRNGKey,
    nn_size: int,
    shape_in: int,
    shape_out: int,
    include_bias: bool = True
) -> dict:
    """
    Initialize the parameters of a linear layer.

    Args:
        key: PRNGKey for random number generation.
        nn_size: Size of the neural network ensemble.
        shape_in: Number of input features.
        shape_out: Number of output features.
        include_bias: Whether to include a bias term.

    Returns:
        dict: Dictionary containing the initialized weights and optional bias.
    """
    key, subkey = jax.random.split(key)
    W = jax.random.normal(subkey, shape=(nn_size, shape_in, shape_out))/ jnp.sqrt(shape_in)
    
    params = {"W": W}
    
    if include_bias:
        key, subkey = jax.random.split(key)
        b = jax.random.normal(subkey, shape=(nn_size, shape_out)) / jnp.sqrt(shape_in)
        params["b"] = b
    
    return params


def linear_layer(
    y: jnp.ndarray,
    W: jnp.ndarray,
    b: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Apply a linear transformation to the input.

    Args:
        y: Input array.
        W: Weight matrix.
        b: Optional bias vector.

    Returns:
        jnp.ndarray: Transformed output.
    """
    output = jnp.dot(y, W)
    if b is not None:
        output += b
    return output


def create_nn(
    nn_size: int,
    input_dim: int=1,
    output_dim: int=1,
    layers: Sequence[int]=[1,1],
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu
) -> Tuple[Callable, Callable]:
    """
    Create a simple feedforward neural network.

    Args:
        nn_size: Size of the neural network ensemble.
        input_dim: Dimension of the input features.
        output_dim: Dimension of the output.
        layers: List of hidden layer sizes.
        activation: Activation function to use between layers.

    Returns:
        Tuple[Callable, Callable]: Initialization and application functions.
    """
    def init(key):
        params = []
        keys = jax.random.split(key, num=len(layers) + 1)

        # Initialize input to first hidden layer
        params.append(init_linear_layer(keys[0], nn_size, input_dim, layers[0]))

        # Initialize hidden layers
        for i in range(1, len(layers)):
            params.append(init_linear_layer(keys[i], nn_size, layers[i-1], layers[i]))

        # Initialize last hidden layer to output layer without bias
        params.append(init_linear_layer(keys[-1], nn_size, layers[-1], output_dim, include_bias=False))
        
        return params
    # def init(key):
    #     params = []
    #     key1,key2 = jax.random.split(key)
    #     params.append(init_linear_layer(key1,nn_size,input_dim,layers[0]))
    #     for i in range(len(layers)-1):
    #         key1,key2 = jax.random.split(key2)
    #         params.append(init_linear_layer(key1,nn_size,layers[i],layers[i+1]))
    #     key1,key2 = jax.random.split(key2)
    #     params.append(init_linear_layer(key1,nn_size,layers[-1],output_dim,include_bias=False))
    #     return params

    def apply(params, x):
        for layer_params in params[:-1]:
            x = linear_layer(x, layer_params["W"], layer_params.get("b"))
            x = activation(x)/ jnp.linalg.norm(x, axis=-1, keepdims=True) 
        # Apply final layer without activation
        x = linear_layer(x, params[-1]["W"])
        return x

    return init, apply
