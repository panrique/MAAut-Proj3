import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def gaussian_rbf(x, centers, epsilon=0.7):
    data_std = np.std(x, axis=0) * epsilon
    return np.exp(-(euclidean_distances(x, centers) ** 2) / (data_std ** 2))

def multiquadric_rbf(x, centers, epsilon=0.05):
    data_std = np.std(x, axis=0)
    return np.sqrt((euclidean_distances(x, centers) / epsilon) ** 2 + (data_std ** 2)) / data_std

def inverse_multiquadric_rbf(x, centers, epsilon=1):
    return 1 / np.sqrt((euclidean_distances(x, centers) / epsilon) ** 2 + 1)

def thin_plate_spline_rbf(x, centers):
    r = euclidean_distances(x, centers)
    return r ** 2 * np.log(r + np.finfo(float).eps)

def train_rbfn(X, y, num_centers, rbf_type, alpha=0.01):
    kmeans = KMeans(n_clusters=num_centers)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    X_transformed = None

    if rbf_type == 'gaussian':
        X_transformed = gaussian_rbf(X, centers)
    elif rbf_type == 'multiquadric':
        X_transformed = multiquadric_rbf(X, centers)
    elif rbf_type == 'inverse_multiquadric':
        X_transformed = inverse_multiquadric_rbf(X, centers)
    elif rbf_type == 'thin_plate_spline':
        X_transformed = thin_plate_spline_rbf(X, centers)
    else:
        raise ValueError("Invalid RBF type")
    
    # Minimizes the objective function: ||y - Xw||^2_2 + alpha * ||w||^2_2
    model = Ridge(alpha=alpha)
    model.fit(X_transformed, y)

    return model, centers

def predict_rbfn(X, model, centers, rbf_type):
    X_transformed = None
    if rbf_type == 'gaussian':
        X_transformed = gaussian_rbf(X, centers)
    elif rbf_type == 'multiquadric':
        X_transformed = multiquadric_rbf(X, centers)
    elif rbf_type == 'inverse_multiquadric':
        X_transformed = inverse_multiquadric_rbf(X, centers)
    elif rbf_type == 'thin_plate_spline':
        X_transformed = thin_plate_spline_rbf(X, centers)
    else:
        raise ValueError("Invalid RBF type")
    return model.predict(X_transformed)

# Set random seed for reproducibility
np.random.seed(42)

# Generate data for two nonlinear functions
# Sine Function
x = np.linspace(-5, 5, 400).reshape(-1, 1)
# Split the data into train and test sets
x1, X_test = train_test_split(x, test_size=0.3)
x1, X_test = np.sort(x1, axis=0), np.sort(X_test, axis=0)
y1 = np.sin(x1) + np.random.normal(0, 0.1, x1.shape)
y_test = np.sin(X_test) + np.random.normal(0, 0.1, X_test.shape)

# Exponential Decay Function
x = np.linspace(0, 10, 400).reshape(-1, 1)
x2, X_test2 = train_test_split(x, test_size=0.3)
x2, X_test2 = np.sort(x2, axis=0), np.sort(X_test2, axis=0)
y2 = np.exp(-x2 / 2) + np.random.normal(0, 0.05, x2.shape)
y_test2 = np.exp(-X_test2 / 2) + np.random.normal(0, 0.05, X_test2.shape)

# Define the options for RBF type, regression type, and clustering algorithm
rbf_types = ['gaussian', 'multiquadric', 'inverse_multiquadric', 'thin_plate_spline']
reg_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]

# Perform the comparisons
mse_values_train = []
mse_values_test = []
mse_values_train2 = []
mse_values_test2 = []
for rbf_type in rbf_types:
    for alpha in reg_values:
        # Train RBFN
        num_centers = 10
        model, centers1 = train_rbfn(x1, y1, num_centers, rbf_type, alpha=alpha)
        model2, centers2 = train_rbfn(x2, y2, num_centers, rbf_type, alpha=alpha)

        # Predict using RBFN
        y_pred_train = predict_rbfn(x1, model, centers1, rbf_type)
        y_pred2_train = predict_rbfn(x2, model2, centers2, rbf_type)
        y_pred = predict_rbfn(X_test, model, centers1, rbf_type)
        y_pred2 = predict_rbfn(X_test2, model2, centers2, rbf_type)

        # calculate mse
        mse = mean_squared_error(y1, y_pred_train)
        mse_values_train.append((rbf_type, alpha, mse))
        mse = mean_squared_error(y_test, y_pred)
        mse_values_test.append((rbf_type, alpha, mse))
        mse = mean_squared_error(y2, y_pred2_train)
        mse_values_train2.append((rbf_type, alpha, mse))
        mse = mean_squared_error(y_test2, y_pred2)
        mse_values_test2.append((rbf_type, alpha, mse))

        # Plot the results
        # region RBNF Approximation 
        plt.scatter(x1, y1, label='Sine Data (Train)')
        plt.scatter(X_test, y_test, label='Sine Data (Test)')
        plt.plot(X_test, y_pred, color='green', label='RBFN Approximation (Test)')
        plt.plot(x1, y_pred_train, color='red', label='RBFN Approximation (Train)')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Function: Sine, RBF Type: {rbf_type}, alpha: {alpha}')
        plt.legend()
        plt.show()
        
        plt.scatter(x2, y2, label='Exponential Decay Data (Train)')
        plt.scatter(X_test2, y_test2, label='Exponential Decay Data (Test)')
        plt.plot(X_test2, y_pred2, color='green', label='RBFN Approximation (Test)')
        plt.plot(x2, y_pred2_train, color='red', label='RBFN Approximation (Train)')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Function: Exponential Decay, RBF Type: {rbf_type}, alpha: {alpha}')
        plt.legend()
        plt.show()
        # endregion

        # region Residuals
        residuals = y_pred - y_test
        plt.scatter(X_test, residuals)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('X')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot - Function: Sine, RBF Type: {rbf_type}, alpha: {alpha}')
        plt.show()
        residuals = y_pred2 - y_test2
        plt.scatter(X_test2, residuals)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('X')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot - Function: Exponential Decay, RBF Type: {rbf_type}, alpha: {alpha}')
        plt.show()
        # endregion Residuals

        # region Error Distribution
        errors = y_pred - y_test
        plt.hist(errors, bins=20)
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution (Test) - Function: Exponential Decay, RBF Type: {rbf_type}, alpha: {alpha}')
        plt.show()
        errors = y_pred2 - y_test2
        plt.hist(errors, bins=20)
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution (Test) - Function: Exponential Decay, RBF Type: {rbf_type}, alpha: {alpha}')
        plt.show()
        # endregion Residuals

# Sort MSE values in ascending order
mse_values_train.sort(key=lambda x: x[2])
mse_values_test.sort(key=lambda x: x[2])
mse_values_train2.sort(key=lambda x: x[2])
mse_values_test2.sort(key=lambda x: x[2])

# Plot the MSE values
configurations = [f'{rbf}_{alpha}' for (rbf, alpha, _) in mse_values_train]
mse_scores = [mse for (_, _, mse) in mse_values_train]

plt.barh(configurations, mse_scores)
plt.xlabel('Mean Squared Error (MSE)')
plt.ylabel('Configuration')
plt.title('Mean Squared Error - Sine (Train)')
plt.show()

configurations = [f'{rbf}_{alpha}' for (rbf, alpha, _) in mse_values_test]
mse_scores = [mse for (_, _, mse) in mse_values_test]

plt.barh(configurations, mse_scores)
plt.xlabel('Mean Squared Error (MSE)')
plt.ylabel('Configuration')
plt.title('Mean Squared Error - Sine (Test)')
plt.show()

configurations = [f'{rbf}_{alpha}' for (rbf, alpha, _) in mse_values_train2]
mse_scores = [mse for (_, _, mse) in mse_values_train2]

plt.barh(configurations, mse_scores)
plt.xlabel('Mean Squared Error (MSE)')
plt.ylabel('Configuration')
plt.title('Mean Squared Error - Exponential Decay (Train)')
plt.show()

configurations = [f'{rbf}_{alpha}' for (rbf, alpha, _) in mse_values_test2]
mse_scores = [mse for (_, _, mse) in mse_values_test2]

plt.barh(configurations, mse_scores)
plt.xlabel('Mean Squared Error (MSE)')
plt.ylabel('Configuration')
plt.title('Mean Squared Error - Exponential Decay (Test)')
plt.show()