import matplotlib.pyplot as plt
import numpy as np
from validation_simulation import iteractions_simulation

def plot_metrics_vs_num_kernels(iteraciones, t, x, y, num_kernels_list):
    """
    Generates plots showing how metrics change as a function of the number of kernels.
    """
    
    num_metrics = 4  # Number of metrics to evaluate
    algorithm_labels = ['Natural', 'Anti-Natural', 'RBF', 'Polynomial']
    
    # Dictionary to store results for each number of kernels
    results = {nk: iteractions_simulation(iteraciones, nk, t, x, y) for nk in num_kernels_list}
    
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Evolution of Metrics Based on Number of Kernels", fontsize=14)

    # Iterate over metrics (indices 0 to 3)
    metric_labels = ['Kernel Alignment', 'Kernel Polarization', 'FSM', 'Complex Ratio']
    for i, ax in enumerate(axes.flat):
        for j, label in enumerate(algorithm_labels):
            metric_values = [results[nk][j, i] for nk in num_kernels_list]
            ax.plot(num_kernels_list, metric_values, marker='o', label=label)
        
        ax.set_title(metric_labels[i])
        ax.set_xlabel("Number of Kernels")
        ax.set_ylabel("Metric Value")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
