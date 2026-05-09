import numpy as np
import pandas as pd
import os
import sys

# Add the base directory of the project to sys.path so we can import from src and extremalitymkl
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from validation_plot_results import plot_metrics_vs_num_kernels  # Import function for plotting results

# Construct the full path to the dataset file
csv_path = os.path.join(BASE_DIR, "data", "australian.csv")

# Load the dataset into a pandas DataFrame
data = pd.read_csv(csv_path)

# Extract feature matrix (X) and target vector (y)
x = np.array(data.iloc[:, :data.shape[1] - 1])  # Select all columns except the last as features
y = np.array(data.iloc[:, data.shape[1] - 1])   # Select the last column as labels

# Convert labels: Replace 0 with -1 to ensure binary classification format (-1, 1)
y = np.where(y == 0, -1 + y, y)

# Set parameters for iterations and kernel generation
iteraciones = 5  # Number of iterations
t = 5            # Maximum number of features per kernel
kernels_list = [5, 10,15]  # List of kernel counts to evaluate

# Generate and plot metrics vs. number of kernels
plot_metrics_vs_num_kernels(iteraciones, t, x, y, kernels_list)
