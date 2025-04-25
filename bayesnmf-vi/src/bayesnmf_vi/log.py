# log.py
import matplotlib.pyplot as plt

class Logger:
    def __init__(self):
        self.history = {"elbo": [], "recon_loss": [], "kl_P": [], "kl_E": []}

    def update(self, elbo_out, step):
        self.history["elbo"].append((step, elbo_out.elbo))
        self.history["recon_loss"].append((step, elbo_out.recon_loss))
        self.history["kl_P"].append((step, elbo_out.kl_P))
        self.history["kl_E"].append((step, elbo_out.kl_E))

    def plot(self):
        plt.figure(figsize=(12, 8))
        for key in self.history.keys():
            steps, values = zip(*self.history[key])
            plt.plot(steps, values, label=key)
        plt.xlabel('Training Step')
        plt.ylabel('Value')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()
