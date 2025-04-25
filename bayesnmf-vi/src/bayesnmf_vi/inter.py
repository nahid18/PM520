import jax
import jax.numpy as jnp
import optax
from .core import VariationalParams, ModelParams, ELBOOutput
from .log import Logger

def debug_print(name, value):
    """Print debug information that works with JAX tracing"""
    jax.debug.print("{}: {}", name, value)
    return value

def check_nan(name, value):
    """Check for NaN values in a JAX-compatible way"""
    has_nan = jnp.any(jnp.isnan(value))
    jax.debug.print("{} has NaN: {}", name, has_nan)
    return value

# Utility to sample from truncated normal (basic rejection sampling)
def truncated_normal_sample(key, lower, mu, sigma, shape):
    def body_fn(val):
        key = val
        key, subkey = jax.random.split(key)
        sample = mu + sigma * jax.random.normal(subkey, shape)
        accept = sample > lower
        return jax.lax.cond(jnp.all(accept), lambda k: k, lambda k: body_fn(k), key)
    return body_fn(key)

# Compute KL divergence between two truncated normals (approx)
def kl_truncated_normal(q_mu, q_sigma, p_mu, p_sigma):
    # Add small epsilon to prevent division by zero
    eps = 1e-8
    q_sigma = jnp.maximum(q_sigma, eps)
    p_sigma = jnp.maximum(p_sigma, eps)
    
    # Compute alpha terms with numerical stability
    alpha_q = (0 - q_mu) / q_sigma
    alpha_p = (0 - p_mu) / p_sigma
    
    # Compute terms with numerical stability
    term1 = jnp.log(p_sigma / q_sigma)
    term2 = (q_sigma**2 + (q_mu - p_mu)**2) / (2 * p_sigma**2)
    
    # Handle the CDF ratio term carefully
    cdf_q = jax.scipy.stats.norm.cdf(alpha_q)
    cdf_p = jax.scipy.stats.norm.cdf(alpha_p)
    
    # Add small epsilon to prevent log(0)
    cdf_ratio = jnp.maximum(cdf_p / jnp.maximum(cdf_q, eps), eps)
    term3 = jnp.log(cdf_ratio)
    
    # Compute final KL with numerical stability
    kl = term1 + term2 + term3 - 1
    kl = jnp.maximum(kl, 0)  # Ensure KL is non-negative
    
    return jnp.sum(kl)

# Sample from folded normal distribution (truncated at 0)
def sample_folded_normal(key, mu, sigma, shape):
    # Generate standard normal samples
    z = jax.random.normal(key, shape)
    
    # Compute the folded normal samples
    # For mu > 0, we use the formula: |mu + sigma * z|
    # For mu < 0, we need to handle the truncation at 0
    samples = jnp.abs(mu + sigma * z)
    
    # For mu < 0, we need to ensure samples are positive
    # We can do this by reflecting negative samples
    samples = jnp.where(mu < 0, 
                       jnp.abs(samples),  # Reflect negative samples
                       samples)           # Keep positive samples as is
    
    return samples

# Sample from log-normal distribution
def sample_lognormal(key, mu, sigma, shape):
    # Generate standard normal samples
    z = jax.random.normal(key, shape)
    
    # Transform to log-normal
    # mu and sigma are the mean and std of the underlying normal distribution
    samples = jnp.exp(mu + sigma * z)
    
    return samples

# Compute KL divergence between two log-normals
def kl_lognormal(q_mu, q_sigma, p_mu, p_sigma):
    # Add small epsilon to prevent division by zero
    eps = 1e-8
    q_sigma = jnp.maximum(q_sigma, eps)
    p_sigma = jnp.maximum(p_sigma, eps)
    
    # Compute KL divergence between two log-normal distributions
    term1 = jnp.log(p_sigma / q_sigma)
    term2 = (q_sigma**2 + (q_mu - p_mu)**2) / (2 * p_sigma**2)
    term3 = -0.5
    
    kl = term1 + term2 + term3
    kl = jnp.maximum(kl, 0)  # Ensure KL is non-negative
    
    return jnp.sum(kl)

# Sample from Gamma distribution
def sample_gamma(key, alpha, beta, shape):
    # Generate Gamma samples using shape-scale parameterization
    # alpha is shape, beta is rate (1/scale)
    samples = jax.random.gamma(key, alpha, shape) / beta
    return samples

# Compute KL divergence between two Gamma distributions
def kl_gamma(q_alpha, q_beta, p_alpha, p_beta):
    # Add small epsilon to prevent division by zero
    eps = 1e-8
    q_alpha = jnp.maximum(q_alpha, eps)
    q_beta = jnp.maximum(q_beta, eps)
    p_alpha = jnp.maximum(p_alpha, eps)
    p_beta = jnp.maximum(p_beta, eps)
    
    # Compute KL divergence between two Gamma distributions
    term1 = (q_alpha - p_alpha) * jax.scipy.special.digamma(q_alpha)
    term2 = -jax.scipy.special.gammaln(q_alpha) + jax.scipy.special.gammaln(p_alpha)
    term3 = p_alpha * (jnp.log(q_beta) - jnp.log(p_beta))
    term4 = q_alpha * (p_beta / q_beta - 1)
    
    kl = term1 + term2 + term3 + term4
    kl = jnp.maximum(kl, 0)  # Ensure KL is non-negative
    
    return jnp.sum(kl)

# Monte Carlo estimate of E_q[exp(P E)]
def monte_carlo_expectation(key, var_params, L=5):
    K, N = var_params.mu_P.shape
    N, G = var_params.mu_E.shape
    
    # Add small epsilon to prevent division by zero
    eps = 1e-8
    mu_P = var_params.mu_P
    sigma_P = jnp.maximum(var_params.sigma_P, eps)
    mu_E = var_params.mu_E
    sigma_E = jnp.maximum(var_params.sigma_E, eps)
    
    key_P, key_E = jax.random.split(key)
    
    # Sample from normal distribution
    P_samples = mu_P + sigma_P * jax.random.normal(key_P, (L, K, N))
    E_samples = mu_E + sigma_E * jax.random.normal(key_E, (L, N, G))
    
    # Compute exp(P E) with numerical stability
    PE = jnp.einsum('lkn,lng->lkg', P_samples, E_samples)
    # Clip values to prevent overflow
    PE = jnp.clip(PE, -20, 20)
    exp_PE = jnp.exp(PE)
    
    return jnp.mean(exp_PE, axis=0)

# Full ELBO computation
def compute_elbo(key, var_params, model_params, L=5):
    # Scale the data to prevent large values
    M = model_params.M / jnp.max(model_params.M)
    
    # Compute reconstruction term with numerical stability
    recon = M * (var_params.mu_P @ var_params.mu_E)
    recon = jnp.clip(recon, -10, 10)  # Prevent overflow
    
    # Compute Monte Carlo term
    monte_carlo_term = monte_carlo_expectation(key, var_params, L)
    monte_carlo_term = jnp.clip(monte_carlo_term, -10, 10)  # Prevent overflow
    
    # Compute reconstruction loss
    recon_loss = jnp.sum(recon) - jnp.sum(monte_carlo_term)
    
    # Compute KL terms using normal KL
    kl_P = kl_truncated_normal(var_params.mu_P, var_params.sigma_P, 
                             jnp.zeros_like(var_params.mu_P), jnp.ones_like(var_params.sigma_P))
    kl_E = kl_truncated_normal(var_params.mu_E, var_params.sigma_E,
                             jnp.zeros_like(var_params.mu_E), jnp.ones_like(var_params.sigma_E))
    
    # Compute final ELBO
    elbo = -recon_loss - kl_P - kl_E
    
    return ELBOOutput(elbo=elbo, recon_loss=recon_loss, kl_P=kl_P, kl_E=kl_E)

# Training function
def train(key, model_params, num_steps=10000, learning_rate=1e-2, tol=1e-6, patience=200):
    K, G = model_params.M.shape
    N = 5
    
    # Initialize logger
    logger = Logger()
    
    # Print initial model information
    jax.debug.print("Model dimensions: K={}, G={}, N={}", K, G, N)
    jax.debug.print("Initial M range: [{}, {}]", jnp.min(model_params.M), jnp.max(model_params.M))
    
    key, subkey1, subkey2 = jax.random.split(key, 3)
    # Initialize with proper scaling based on data
    M_scale = jnp.max(model_params.M)
    # Initialize with larger values and more variance
    init_mu_P = jax.random.normal(subkey1, (K, N)) * 2.0  # Increased from 0.1
    init_sigma_P = jnp.ones((K, N)) * 1.0  # Increased from 0.1
    init_mu_E = jax.random.normal(subkey2, (N, G)) * 2.0  # Increased from 0.1
    init_sigma_E = jnp.ones((N, G)) * 1.0  # Increased from 0.1
    
    var_params = VariationalParams(mu_P=init_mu_P, sigma_P=init_sigma_P,
                                 mu_E=init_mu_E, sigma_E=init_sigma_E)
    
    # Use Adam with momentum and gradient noise
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=num_steps,
        decay_rate=0.999  # Slower decay
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Less aggressive gradient clipping
        optax.add_noise(1e-4, 0.9, seed=42),  # Use a fixed scalar seed
        optax.adam(schedule, b1=0.9, b2=0.999)  # Higher momentum
    )
    opt_state = optimizer.init(var_params)

    def step(var_params, opt_state, key):
        key, subkey = jax.random.split(key)
        loss_fn = lambda params: -compute_elbo(subkey, params, model_params).elbo
        grads = jax.grad(loss_fn)(var_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        var_params = optax.apply_updates(var_params, updates)
        
        # Clip parameters to prevent explosion but allow more movement
        var_params = VariationalParams(
            mu_P=jnp.clip(var_params.mu_P, -5, 5),  # Wider range
            sigma_P=jnp.clip(var_params.sigma_P, 1e-4, 2.0),
            mu_E=jnp.clip(var_params.mu_E, -5, 5),  # Wider range
            sigma_E=jnp.clip(var_params.sigma_E, 1e-4, 2.0)
        )
        
        return var_params, opt_state, key

    # JIT compile the step function
    step = jax.jit(step)

    prev_elbo = -jnp.inf
    no_improvement_count = 0
    best_elbo = -jnp.inf
    best_params = var_params
    
    for step_idx in range(num_steps):
        var_params, opt_state, key = step(var_params, opt_state, key)
        
        # Compute ELBO outside the JIT-compiled function
        if step_idx % 100 == 0:  # Print less frequently
            elbo_out = compute_elbo(key, var_params, model_params)
            jax.debug.print("\nStep {}:", step_idx)
            jax.debug.print("  ELBO: {:.4f}", elbo_out.elbo)
            jax.debug.print("  Components:")
            jax.debug.print("    Recon loss: {:.4f}", elbo_out.recon_loss)
            jax.debug.print("    KL_P: {:.4f}", elbo_out.kl_P)
            jax.debug.print("    KL_E: {:.4f}", elbo_out.kl_E)
            
            # Update logger
            logger.update(elbo_out, step_idx)
            
            # Check for improvement with more tolerance
            if elbo_out.elbo > best_elbo + tol:
                best_elbo = elbo_out.elbo
                best_params = var_params
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Check convergence with more patience
            if no_improvement_count >= patience:
                jax.debug.print("Converged at step {} (no improvement for {} steps)", 
                              step_idx, patience)
                break
            
            # Ensure ELBO is increasing
            if elbo_out.elbo < prev_elbo:
                jax.debug.print("WARNING: ELBO decreased from {:.4f} to {:.4f}", 
                              prev_elbo, elbo_out.elbo)
            
            prev_elbo = elbo_out.elbo
    
    return best_params, logger
