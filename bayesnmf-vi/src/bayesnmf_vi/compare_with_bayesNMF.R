# Load required packages
library(bayesNMF)
library(ggplot2)
library(gridExtra)

# Read the simulated data
M <- as.matrix(read.csv("simulated_data/M.csv", header = FALSE))
true_P <- as.matrix(read.csv("simulated_data/true_P.csv", header = FALSE))
true_E <- as.matrix(read.csv("simulated_data/true_E.csv", header = FALSE))

# Read metadata
metadata <- readLines("simulated_data/metadata.txt")
K <- as.numeric(gsub("K=", "", metadata[1]))
G <- as.numeric(gsub("G=", "", metadata[2]))
N <- as.numeric(gsub("N=", "", metadata[3]))

# Run bayesNMF
set.seed(42)  # For reproducibility
result <- bayesNMF(M, K=N, max_iter=10000, tol=1e-6)

# Extract learned parameters
learned_P <- result$P
learned_E <- result$E

# Create comparison plots
plot_comparison <- function(true, learned, title) {
  df <- data.frame(
    true = as.vector(true),
    learned = as.vector(learned)
  )
  
  p <- ggplot(df, aes(x=true, y=learned)) +
    geom_point(alpha=0.5) +
    geom_abline(slope=1, intercept=0, color="red", linetype="dashed") +
    labs(title=title, x="True Values", y="Learned Values") +
    theme_minimal()
  
  # Add correlation coefficient
  corr <- cor(df$true, df$learned)
  p <- p + annotate("text", x=min(df$true), y=max(df$learned),
                   label=sprintf("Correlation: %.3f", corr),
                   hjust=0, vjust=1)
  
  return(p)
}

# Create plots
p1 <- plot_comparison(true_P, learned_P, "Comparison of P matrices")
p2 <- plot_comparison(true_E, learned_E, "Comparison of E matrices")

# Save plots
pdf("simulated_data/bayesNMF_comparison.pdf", width=12, height=6)
grid.arrange(p1, p2, ncol=2)
dev.off()

# Print statistics
cat("\nComparison Statistics:\n")
cat("P matrix:\n")
cat(sprintf("  True mean: %.4f, std: %.4f\n", mean(true_P), sd(true_P)))
cat(sprintf("  Learned mean: %.4f, std: %.4f\n", mean(learned_P), sd(learned_P)))
cat(sprintf("  Mean absolute error: %.4f\n", mean(abs(true_P - learned_P))))

cat("\nE matrix:\n")
cat(sprintf("  True mean: %.4f, std: %.4f\n", mean(true_E), sd(true_E)))
cat(sprintf("  Learned mean: %.4f, std: %.4f\n", mean(learned_E), sd(learned_E)))
cat(sprintf("  Mean absolute error: %.4f\n", mean(abs(true_E - learned_E)))) 