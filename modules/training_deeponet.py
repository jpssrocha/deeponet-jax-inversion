"""
Module for making it remote DeepOnet training more robust. Been using notebooks
but having problems with remote connectivity and the jupyter kernel not reconnecting
properly. With a script i can do a more robust training procedure.
"""

# Stdlib
from pathlib import Path
from typing import Callable
from copy import deepcopy
from dataclasses import dataclass, asdict, field
import json


# 3th party
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import h5py
import matplotlib.pyplot as plt


# Setting JAX device to the last so that i use the least used chip on the server

devices = jax.devices()

if len(devices) > 1:
    jax.config.update("jax_default_device", devices[-1])




# Setting up global constants
DATA_FILE = Path("outputs/poisson_pressure_simulations.h5")
TESTING_DATA_FILE = Path("outputs/poisson_pressure_simulations_test.h5")

CACHE = True
MODEL_OUTPUT = Path("outputs/fivespot_deeponet.eqx")
BEST_MODEL_OUTPUT = Path("outputs/fivespot_deeponet_best.eqx")
LOSS_CACHE_PATH = Path("training_cache/loss_history.json")
SAVING_INTERVAL = 1000

RANDOM_SEED = 5828
LR = 0.001
TOTAL_EPOCHS = 100_000


@dataclass
class LossTrainingState:
    training_losses: list[float] = field(default_factory=list)
    test_losses: list[float] = field(default_factory=list)
    min_loss: float = 1_000_000_000.0


class DeepOnet(eqx.Module):
    """ Simplified deeponet using twin  MLP as architecture """

    branch_net: eqx.nn.MLP
    trunk_net: eqx.nn.MLP
    bias: jax.Array

    def __init__(
        self, 
        in_branch: int,
        in_trunk: int,
        width: int,
        depth: int,
        interact: int,
        activation: Callable,
        *,
        key
    ):

        b_key, t_key = jax.random.split(key)

        self.branch_net = eqx.nn.MLP(
            in_branch,
            interact,
            width,
            depth,
            activation,
            key=b_key
        )

        self.trunk_net = eqx.nn.MLP(
            in_trunk,
            interact,
            width,
            depth,
            activation,
            final_activation=activation,
            key=b_key
        )

        self.bias = jnp.zeros((1,))

    
    def __call__(self, x_branch, x_trunk):
        """
        Parameters:
        -----------
        x_branch.shape = (in_size_branch,)
        x_trunk.shape = (in_size_trunk,)

        Returns:
        --------

        return shape: "scalar"
        """

        branch_out = self.branch_net(x_branch)
        trunk_out = self.trunk_net(x_trunk)

        inner_product = jnp.sum(branch_out*trunk_out, keepdims=True)

        return (inner_product + self.bias)[0]  # Taking the zeroth index 


@eqx.filter_jit
def loss_fn(model, X, theta, P):
    """MSE function to be used as loss function"""

    outputs = jax.vmap(jax.vmap(model, in_axes=(None, 0)), in_axes=(0, None))(theta, X)
    mse = jnp.mean(jnp.square(outputs - P))

    return mse


@eqx.filter_jit
def update(
    opt_state: optax.OptState,
    model: eqx.Module,
    optimizer: optax.GradientTransformation,
    X: jax.Array,
    theta: jax.Array,
    P: jax.Array
) -> tuple[jax.Array, eqx.Module, optax.OptState]:
    """Update function for training loop. """

    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, X, theta, P)
    updates, new_state = optimizer.update(grad, opt_state, model)  # pyright: ignore
    new_model = eqx.apply_updates(model, updates)

    return loss, new_model, new_state


def train_deeponet_pressure_field(continue_training: bool = True):
    """Main training function for deeponet"""

    print("Loading training data ...")
    # Loading training data
    with h5py.File(DATA_FILE, "r") as f:

        X = jnp.array(f["positions"])
        T = jnp.array(f["kl_transform_matrice"])
        P = jnp.array(f["pressure_fields"])
        theta = jnp.array(f["kl_coefficients"])
        Y = jnp.array(f["Y_fields"])

    
    # Loading testing data
    print("Loading testing data ...")
    with h5py.File(TESTING_DATA_FILE, "r") as f:
    
        X_test = jnp.array(f["positions"])
        T_test = jnp.array(f["kl_transform_matrice"])
        P_test = jnp.array(f["pressure_fields"])
        theta_test = jnp.array(f["kl_coefficients"])
        Y_test = jnp.array(f["Y_fields"])

    # Initializing neural network and auxiliary variables for training

    key = jax.random.key(RANDOM_SEED)

    deeponet = DeepOnet(
        24,
        2,
        256,
        5,
        128,
        activation=jax.nn.relu,
        key=key
    )

    best_model = deepcopy(deeponet)

    loss_history = LossTrainingState()

    if continue_training:

        print("Loading last run models")

        if MODEL_OUTPUT.exists():
            deeponet = eqx.tree_deserialise_leaves(MODEL_OUTPUT, deeponet)
            
        if BEST_MODEL_OUTPUT.exists():
            best_model = eqx.tree_deserialise_leaves(BEST_MODEL_OUTPUT, best_model)

        if LOSS_CACHE_PATH.exists():
            with open(LOSS_CACHE_PATH, "r") as f:
                loss_history = LossTrainingState(**json.load(f))


    optimizer = optax.adam(LR)
    opt_state = optimizer.init(eqx.filter(deeponet, eqx.is_array))

    min_loss = loss_history.min_loss if loss_history.training_losses else 1_000_000_000

    N = len(loss_history.training_losses) if loss_history.training_losses else 0

    epoch = -1
    loss = -1
    test_loss = -1

    for epoch in range(N, TOTAL_EPOCHS):

        loss, deeponet, opt_state = update(opt_state, deeponet, optimizer, X, theta, P)

        loss_history.training_losses.append(float(loss))

        if epoch % 100 == 0:
            test_loss = loss_fn(deeponet, X_test, theta_test, P_test)
            loss_history.test_losses.append(float(test_loss))
        
            print(f"{epoch = } | {loss = } | {test_loss = }")

        if loss <= min_loss and len(loss_history.training_losses) > 10_000:
            print(f"Best model  at step: {len(loss_history.training_losses)} | {loss = }")
            best_model = deepcopy(deeponet)
            loss_history.min_loss = float(loss)
            min_loss = loss

        if epoch % SAVING_INTERVAL == 0:
            print("Checkpointing!")

            eqx.tree_serialise_leaves(BEST_MODEL_OUTPUT, best_model)
            eqx.tree_serialise_leaves(MODEL_OUTPUT, deeponet)

            with open(LOSS_CACHE_PATH, "w") as f:
                json.dump(asdict(loss_history), f, indent=4)
            

    print("Training finished. Saving final results.")
    print(f"{epoch = } | {loss = } | {test_loss = }")
    
    eqx.tree_serialise_leaves(BEST_MODEL_OUTPUT, best_model)
    eqx.tree_serialise_leaves(MODEL_OUTPUT, deeponet)

    with open(LOSS_CACHE_PATH, "w") as f:
        json.dump(asdict(loss_history), f, indent=4)

    return 0
        

if __name__ == "__main__":

    continue_training = input("Do you want to continue training? (start new training otherwise) [y | n]  ").lower()
    loop = True

    
    while loop:

        match continue_training:

            case "y":
                train_deeponet_pressure_field(True)
                loop = False

            case "n":
                train_deeponet_pressure_field(False)
                loop = False

            case _:
                print("Choose y or n")

    print("Finishing program.")
