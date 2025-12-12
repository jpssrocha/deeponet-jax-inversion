"""
 
"""
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from typing import Callable
from pathlib import Path
import h5py

from simulators_tools_marcio import coordinates, simulationpar, five_spot_simulation
import simulators_tools_marcio as sim

# Setting up the necessary constants

TESTING_DATA_FILE = Path("../../outputs/poisson_pressure_simulations_test.h5")

with h5py.File(TESTING_DATA_FILE, "r") as f:
    X = jnp.array(f["positions"])
    T = jnp.array(f["kl_transform_matrice"])


P_ref_file = Path("../../raw_data/pressure_ref.dat")
Y_ref_file = Path("../../raw_data/YS_ref.dat")


P_ref = jnp.array(np.loadtxt(P_ref_file))
Y_ref = jnp.array(np.loadtxt(Y_ref_file))

pos  = [255, 755, 1255, 1755, 2255,  265,  765, 1265, 1765, 2265,  275, 775,
        1275, 1775, 2275,  285,  785, 1285, 1785, 2285,  295,  795, 1295, 1795, 2295]

X_ref = X[pos, :]

DATA_FILE = Path("../../outputs/poisson_pressure_simulations.h5")

with h5py.File(DATA_FILE, "r") as f:
    theta = jnp.array(f["kl_coefficients"])

def generate_intervals():
    min_, max_ = theta.min(axis=0), theta.max(axis=0)
    intervals = [(float(mi), float(ma)) for mi, ma in zip(min_, max_)]

    return intervals, min_, max_

INTERVALS, MIN, MAX = generate_intervals()
SIGMA = 0.5

def log_likelihood_numerical(theta):
    """
    A log likelihood function for implementing within EMCEE
    with the numerical simulator within.
    """
    
    simul_setup ='fivespot2D'
    beta = 9.8692e-14
    rho  = 1.
    mu   = 1.0e-03
    Dom  = [100., 100., 1.]
    mesh = [50, 50, 1]
    BHP  = 101325.0
    PL   = 0.
    PR   = 0.
    rw   = 0.125
    q    = 100.
    pos  = [255, 755, 1255, 1755, 2255,  265,  765, 1265, 1765, 2265,  275, 775,
            1275, 1775, 2275,  285,  785, 1285, 1785, 2285,  295,  795, 1295, 1795, 2295]

    inputpar = simulationpar(simul_setup,beta,rho,mu,Dom,mesh,BHP,PR,PL,rw,q,pos,0)
    nx = mesh[0]
    ny = mesh[1]
    
    Y = T@theta
    
    p = five_spot_simulation(inputpar, Y)

    return -0.5 * jnp.sum((P_ref - p[pos]) ** 2 / SIGMA**2)    


class DeepOnet(eqx.Module):
    """
        DeepONet definition

        Parameters
        ----------

        
    """
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
        """
        Simplified deeponet using twin hidden architectures
        """

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
        x_branch.shape = (in_size_branch,)
        x_trunk.shape = (1,)

        return shape: "scalar"
        """

        branch_out = self.branch_net(x_branch)
        trunk_out = self.trunk_net(x_trunk)

        inner_product = jnp.sum(branch_out*trunk_out, keepdims=True)

        return (inner_product + self.bias)[0]


key = jax.random.key(1325)

deeponet = DeepOnet(
    24,
    2,
    256,
    5,
    128,
    activation=jax.nn.relu,
    key=key
)

deeponet = eqx.tree_deserialise_leaves("../../outputs/fivespot_deeponet_best.eqx", deeponet)

@jax.jit
def log_likelihood(theta):
    out = jax.vmap(deeponet, in_axes=(None, 0))(theta, X_ref)
    return -0.5 * jnp.sum((P_ref - out) ** 2 / SIGMA**2)     


def log_prior(theta):
    if  all( (MIN <= theta) & (theta <= MAX) ):
        return 0.0
    return -jnp.inf


def log_probability(theta):
    lp = log_prior(theta)
    if not jnp.isfinite(lp):
        return -jnp.inf
    return lp + log_likelihood(theta)


def log_probability_numerical(theta):
    lp = log_prior(theta)
    if not jnp.isfinite(lp):
        return -jnp.inf
    return lp + log_likelihood_numerical(theta)
        
