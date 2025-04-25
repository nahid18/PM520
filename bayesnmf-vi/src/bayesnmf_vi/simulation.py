# simulate_and_run.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from bayesnmf_vi.core import ModelParams
from bayesnmf_vi.inter import train, compute_elbo
from bayesnmf_vi.log import Logger

def simulate_data(K=10, G=20, N=5, key=jax.random.PRNGKey(0)):
    jax.debug.print("Simulating data with dimensions: K={}, G={}, N={}", K, G, N)
    
    key, subkey1, subkey2 = jax.random.split(key, 3)
    true_P = jax.random.uniform(subkey1, (K, N)) * 2.0  # Bigger range
    true_E = jax.random.uniform(subkey2, (N, G)) * 2.0
    
    eta = jnp.dot(true_P, true_E)
    M = jax.random.poisson(key, eta)
    
    jax.debug.print("Generated data statistics:")
    jax.debug.print("  M mean: {:.2f}, std: {:.2f}", jnp.mean(M), jnp.std(M))
    
    return ModelParams(M=M, true_P=true_P, true_E=true_E)

def save_simulated_data(model_params, output_dir="simulated_data"):
    """Save simulated data to CSV files for use with R bayesNMF package"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save M matrix (count data)
    pd.DataFrame(model_params.M).to_csv(
        os.path.join(output_dir, "M.csv"), 
        index=False, 
        header=False
    )
    
    # Save true P matrix
    pd.DataFrame(model_params.true_P).to_csv(
        os.path.join(output_dir, "true_P.csv"), 
        index=False, 
        header=False
    )
    
    # Save true E matrix
    pd.DataFrame(model_params.true_E).to_csv(
        os.path.join(output_dir, "true_E.csv"), 
        index=False, 
        header=False
    )
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
        f.write(f"K={model_params.M.shape[0]}\n")
        f.write(f"G={model_params.M.shape[1]}\n")
        f.write(f"N={model_params.true_P.shape[1]}\n")

def plot_comparison(true_values, learned_values, title, ax):
    """Plot comparison between true and learned values"""
    # Flatten the arrays for scatter plot
    true_flat = true_values.flatten()
    learned_flat = learned_values.flatten()
    
    # Create scatter plot
    ax.scatter(true_flat, learned_flat, alpha=0.5)
    
    # Add identity line
    min_val = min(true_flat.min(), learned_flat.min())
    max_val = max(true_flat.max(), learned_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Identity line')
    
    # Add labels and title
    ax.set_xlabel('True Values')
    ax.set_ylabel('Learned Values')
    ax.set_title(title)
    ax.legend()
    
    # Calculate correlation
    corr = np.corrcoef(true_flat, learned_flat)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax.transAxes, verticalalignment='top')

def main():
    key = jax.random.PRNGKey(42)
    model_params = simulate_data(key=key)

    # Save simulated data
    save_simulated_data(model_params)
    jax.debug.print("\nSaved simulated data to 'simulated_data' directory")

    # Initialize
    K, G = model_params.M.shape
    N = 5

    jax.debug.print("\nStarting training...")
    key, subkey = jax.random.split(key)
    trained_var_params, logger = train(subkey, model_params, num_steps=100000, learning_rate=1e-1)

    # Final ELBO
    final_elbo = compute_elbo(key, trained_var_params, model_params)
    jax.debug.print("\nFinal results:")
    jax.debug.print("  ELBO: {:.4f}", final_elbo.elbo)
    jax.debug.print("  Recon loss: {:.4f}", final_elbo.recon_loss)
    jax.debug.print("  KL_P: {:.4f}", final_elbo.kl_P)
    jax.debug.print("  KL_E: {:.4f}", final_elbo.kl_E)

    # Plot training history
    logger.plot()

    # Compare true and learned values
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Compare P matrices
    learned_mean_P = trained_var_params.mu_P
    plot_comparison(model_params.true_P, learned_mean_P, 
                   'Comparison of P matrices', axes[0])
    
    # Compare E matrices
    learned_mean_E = trained_var_params.mu_E
    plot_comparison(model_params.true_E, learned_mean_E, 
                   'Comparison of E matrices', axes[1])
    
    plt.tight_layout()
    plt.savefig('simulated_data/comparison_plot.png')
    plt.show()

    # Print additional statistics
    jax.debug.print("\nComparison Statistics:")
    jax.debug.print("P matrix:")
    jax.debug.print("  True mean: {:.4f}, std: {:.4f}", 
                   jnp.mean(model_params.true_P), jnp.std(model_params.true_P))
    jax.debug.print("  Learned mean: {:.4f}, std: {:.4f}", 
                   jnp.mean(learned_mean_P), jnp.std(learned_mean_P))
    jax.debug.print("  Mean absolute error: {:.4f}", 
                   jnp.mean(jnp.abs(model_params.true_P - learned_mean_P)))
    
    jax.debug.print("\nE matrix:")
    jax.debug.print("  True mean: {:.4f}, std: {:.4f}", 
                   jnp.mean(model_params.true_E), jnp.std(model_params.true_E))
    jax.debug.print("  Learned mean: {:.4f}, std: {:.4f}", 
                   jnp.mean(learned_mean_E), jnp.std(learned_mean_E))
    jax.debug.print("  Mean absolute error: {:.4f}", 
                   jnp.mean(jnp.abs(model_params.true_E - learned_mean_E)))

if __name__ == "__main__":
    main()
