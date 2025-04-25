import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class ModelParams:
    def __init__(self, M, true_P, true_E):
        self.M = M
        self.true_P = true_P
        self.true_E = true_E

    def tree_flatten(self):
        children = (self.M, self.true_P, self.true_E)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class VariationalParams:
    def __init__(self, mu_P, sigma_P, mu_E, sigma_E):
        self.mu_P = mu_P
        self.sigma_P = sigma_P
        self.mu_E = mu_E
        self.sigma_E = sigma_E

    def tree_flatten(self):
        children = (self.mu_P, self.sigma_P, self.mu_E, self.sigma_E)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class ELBOOutput:
    def __init__(self, elbo, recon_loss, kl_P, kl_E):
        self.elbo = elbo
        self.recon_loss = recon_loss
        self.kl_P = kl_P
        self.kl_E = kl_E

    def tree_flatten(self):
        children = (self.elbo, self.recon_loss, self.kl_P, self.kl_E)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)