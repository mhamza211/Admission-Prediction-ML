import numpy as np
import pandas as pd
import math
import copy

# Sigmoid function to map predictions between 0 and 1
def sigmoid(z):
    return 1/(1+np.exp(-z))

# Computing the cost function with regularization to avoid overfitting
def compute_cost(x,y,w,b,lambda_=1):
    m,n = x.shape
    cost = 0
    # Calculate cost for each example
    for i in range(m):
        f_wb = sigmoid(np.dot(x[i],w) + b)  # Predicted value
        cost += -y[i]*np.log(f_wb) - (1-y[i])*np.log(1-f_wb)  # Log loss
    cost = cost/m
    
    # Regularization term
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j])**2
    reg_cost = (lambda_/(2*m)) * reg_cost
    
    # Total cost (including regularization)
    total_cost = cost + reg_cost
    return total_cost

# Computing gradients for updating weights (w) and bias (b) with regularization
def compute_gradient(x,y,w,b,lambda_):
    m,n = x.shape
    dj_dw = np.zeros((n,))  # Gradient for w
    dj_db = 0  # Gradient for b
    
    # Compute gradients for each example
    for i in range(m):
        f_wb = sigmoid(np.dot(x[i],w) + b)  # Prediction
        err = f_wb - y[i]  # Error term
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i,j]  # Gradient for w
        dj_db = dj_db + err  # Gradient for b
    
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    
    # Add regularization term to gradient of w
    for i in range(n):
        dj_dw[i] = dj_dw[i] + (lambda_/m)*w[i]
    
    return dj_dw,dj_db


# Gradient descent to update w and b using computed gradients
def compute_gradient_decent(x, y, w, b, compute_cost, compute_gradient, alpha, num_iters, lambda_):
    m = len(x)
    j_history = []  # Cost history
    w_history = []  # Weights history
    
    # Loop through iterations to update w and b
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b, lambda_)
        w = w - alpha * dj_dw  # Update weights
        b = b - alpha * dj_db  # Update bias
        
        # Store cost for first 100,000 iterations
        if i < 100000:
            cost = compute_cost(x, y, w, b, lambda_)
            j_history.append(cost)
        
        # Print cost at every 10% of iterations
        if i % math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w)
            print("iterations", i, " cost", cost)
    
    return w, b, j_history, w_history

# Predict function using learned weights and bias
def predict(x, w, b):
    m, n = x.shape
    p = np.zeros(m)  # Initialize predictions
    
    # Make prediction for each example
    for i in range(m):
        f_wb = sigmoid(np.dot(x[i], w) + b)  # Apply sigmoid
        p[i] = f_wb >= 0.5  # Classify based on threshold of 0.5
    return p




# Sample dataset for testing the gradient function
np.random.seed(1)
X_tmp = np.random.rand(5,3)  # Random feature matrix
y_tmp = np.array([0,1,0,1,0])  # Labels
w_tmp = np.random.rand(X_tmp.shape[1])  # Random weights
b_tmp = 0.5  # Bias
lambda_tmp = 0.7  # Regularization parameter

# Compute gradients using the test dataset
dj_dw_tmp, dj_db_tmp = compute_gradient(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

# Sample training data of student scores in 2 subjects
x_train = np.array([
    [85, 90],   # Scores in 2 subjects for student 1
    [78, 85],   # Scores for student 2
    [92, 88],   # Scores for student 3
    [60, 65],   # Scores for student 4
    [75, 80]    # Scores for student 5
])

# Labels for whether students got admission (1) or were rejected (0)
y_train = np.array([1, 1, 1, 0, 0])

# Initialize weights, bias, and hyperparameters for gradient descent
np.random.seed(1)
lambda_tmp = 0.7
initial_w = 0.01 * (np.random.rand(x_train.shape[1]) - 0.5)  # Small random weights
initial_b = -8  # Initial bias
iterations = 10000  # Number of iterations
alpha = 0.001  # Learning rate

# Run gradient descent to optimize weights and bias
w, b, J_history, _ = compute_gradient_decent(x_train, y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)

# Test data for predictions
tmp_w = w
tmp_b = b
tmp_x = np.array([
    [85, 90],   # Scores for student 1
    [78, 85],   # Scores for student 2
    [92, 88],   # Scores for student 3
    [60, 65]    # Scores for student 4
])

# Predict admission results based on optimized weights and bias
tmp_p = predict(tmp_x, tmp_w, tmp_b)
print("Output of predict ")
print("values ", tmp_p)  # 1: admitted, 0: rejected
