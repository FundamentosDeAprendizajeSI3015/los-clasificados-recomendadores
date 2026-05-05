import numpy as np

from src.kernel_metrics import complex_ratio, kernel_aligment, FSM, kernel_polarization
from extremalitymkl.extremality_weights import kernel_extremaly_weights
from src.weak_polinomial_kernel import create_weak_kernels

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn.metrics.pairwise import polynomial_kernel

def calculate_kernel_metrics(KL_train, y_train):
    """
    Computes kernel-based metrics for a given kernel matrix and labels.
    Returns a list of computed metrics.
    """
    measures = [
        kernel_aligment(KL_train, y_train),
        kernel_polarization(KL_train, y_train),
        FSM(KL_train, y_train),
        complex_ratio(KL_train)
    ]
    return measures

def train_and_evaluate(KL_train,X_train, y_train):
    """
    Trains and evaluates different models using a combination of kernels.
    Computes kernel weights, constructs new kernel matrices, and evaluates them
    using multiple kernel metrics.
    """
    # Compute kernel weights using extremality-based weighting
    result_comparison = kernel_extremaly_weights(KL_train, y_train, n=2)
    
    # Compute natural kernel metrics
    gram_train = np.einsum('ijk,i -> jk', KL_train, result_comparison.w_1)
    natural_metrics = calculate_kernel_metrics(gram_train, y_train)
    
    # Compute anti-natural kernel metrics
    gram_train = np.einsum('ijk,i -> jk', KL_train, result_comparison.w_2)
    anti_natural_metrics = calculate_kernel_metrics(gram_train, y_train)
    
    # Compute RBF kernel metrics
    gram = rbf_kernel(X_train)
    rbf_metrics = calculate_kernel_metrics(gram, y_train)
    
    # Compute polynomial kernel metrics
    gram = polynomial_kernel(X_train, degree=3)
    poly_metrics = calculate_kernel_metrics(gram, y_train)
    
    return natural_metrics, anti_natural_metrics, rbf_metrics, poly_metrics

def iteractions_simulation(iteraciones, numero_k, t, x, y):
    """
    Runs multiple iterations of kernel-based model evaluation.
    It normalizes the input data, splits it into training and testing sets,
    generates weak polynomial kernels, and computes kernel metrics for different models.
    """
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    
    num_algorithms = 4  # Number of models evaluated
    metrics_shape = (iteraciones, num_algorithms, 4)  # Adjust size based on computed metrics
    all_metrics = np.zeros(metrics_shape)
    
    for k in range(iteraciones):
        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=k)
        
        # Generate weak polynomial kernels
        KL_train, KL_test = create_weak_kernels(X_train, X_test, numero_k, t)
        
        # Train models and compute kernel metrics
        natural_metrics, anti_natural_metrics, rbf_metrics, poly_metrics = train_and_evaluate(KL_train,X_train, y_train)
        
        # Store computed metrics
        all_metrics[k, 0, :] = natural_metrics
        all_metrics[k, 1, :] = anti_natural_metrics
        all_metrics[k, 2, :] = rbf_metrics
        all_metrics[k, 3, :] = poly_metrics
    
    print(k, numero_k)
    
    # Compute the mean of the metrics across iterations
    mean_results = np.mean(all_metrics, axis=0)
    print(mean_results)
    
    return mean_results