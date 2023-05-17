import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import Ridge

def gaussian_rbf(x, centers, width):
    return np.exp(-(euclidean_distances(x, centers) ** 2) / (2 * width ** 2))

def train_rbfn(X, y, num_centers, width):
    kmeans = KMeans(n_clusters=num_centers)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    X_transformed = gaussian_rbf(X, centers, width)
    model = Ridge(alpha=0.01)  # Ridge regression for training the RBFN
    model.fit(X_transformed, y)
    return model, centers

def predict_rbfn(X, model, centers, width):
    X_transformed = gaussian_rbf(X, centers, width)
    return model.predict(X_transformed)

# Example usage:
# Generate some nonlinear data
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.1, X.shape)

# Train RBFN
num_centers = 10
width = 1.0
model, centers = train_rbfn(X, y, num_centers, width)

# Predict using RBFN
X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
y_pred = predict_rbfn(X_test, model, centers, width)

# Plot the original data and the RBFN approximation
import matplotlib.pyplot as plt

plt.scatter(X, y, label='Original Data')
plt.plot(X_test, y_pred, color='red', label='RBFN Approximation')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()