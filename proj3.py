import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def gaussian_rbf(x, centers, width):
    return np.exp(-(euclidean_distances(x, centers) ** 2) / (2 * width ** 2))

def multiquadric_rbf(x, centers, epsilon):
    return np.sqrt((euclidean_distances(x, centers) / epsilon) ** 2 + 1)

def inverse_multiquadric_rbf(x, centers, epsilon):
    return 1 / np.sqrt((euclidean_distances(x, centers) / epsilon) ** 2 + 1)

def thin_plate_spline_rbf(x, centers):
    r = euclidean_distances(x, centers)
    return r ** 2 * np.log(r + np.finfo(float).eps)

def train_rbfn(X, y, num_centers, width, rbf_type, epsilon=1.0):
    kmeans = KMeans(n_clusters=num_centers)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    X_transformed = None

    if rbf_type == 'gaussian':
        X_transformed = gaussian_rbf(X, centers, width)
    elif rbf_type == 'multiquadric':
        X_transformed = multiquadric_rbf(X, centers, epsilon)
    elif rbf_type == 'inverse_multiquadric':
        X_transformed = inverse_multiquadric_rbf(X, centers, epsilon)
    elif rbf_type == 'thin_plate_spline':
        X_transformed = thin_plate_spline_rbf(X, centers)
    else:
        raise ValueError("Invalid RBF type")
    
    model = Ridge(alpha=0.01)
    model.fit(X_transformed, y)

    return model, centers

def predict_rbfn(X, model, centers, width, rbf_type, epsilon=1.0):
    X_transformed = None
    if rbf_type == 'gaussian':
        X_transformed = gaussian_rbf(X, centers, width)
    elif rbf_type == 'multiquadric':
        X_transformed = multiquadric_rbf(X, centers, epsilon)
    elif rbf_type == 'inverse_multiquadric':
        X_transformed = inverse_multiquadric_rbf(X, centers, epsilon)
    elif rbf_type == 'thin_plate_spline':
        X_transformed = thin_plate_spline_rbf(X, centers)
    else:
        raise ValueError("Invalid RBF type")
    return model.predict(X_transformed)

# Example usage:
# Generate data for two nonlinear functions
# Sine Function
X1 = np.linspace(-5, 5, 100).reshape(-1, 1)
y1 = np.sin(X1) + np.random.normal(0, 0.1, X1.shape)

# Exponential Decay Function
X2 = np.linspace(0, 10, 100).reshape(-1, 1)
y2 = np.exp(-X2 / 2) + np.random.normal(0, 0.05, X2.shape)

# Define the options for RBF type, regression type, and clustering algorithm
rbf_types = ['gaussian', 'multiquadric', 'inverse_multiquadric', 'thin_plate_spline']

# Perform the comparisons
for rbf_type in rbf_types:
    # Train RBFN
    num_centers = 10
    model, centers1 = train_rbfn(X1, y1, num_centers, 1.0, rbf_type)
    model2, centers2 = train_rbfn(X2, y2, num_centers, 0.5, rbf_type)

    # Predict using RBFN
    X_test = np.linspace(-6, 11, 200).reshape(-1, 1)
    y_pred = predict_rbfn(X_test, model, centers1, 1.0, rbf_type)
    y_pred2 = predict_rbfn(X_test, model2, centers2, 0.5, rbf_type)

    # Plot the results
    plt.scatter(X1, y1, label='Sine Data')
    plt.plot(X_test, y_pred, color='red', label='RBFN Approximation')
    
    plt.scatter(X2, y2, label='Exponential Decay Data')
    plt.plot(X_test, y_pred2, color='green', label='RBFN 2 Approximation')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'RBF Type: {rbf_type}')
    plt.legend()
    plt.show()