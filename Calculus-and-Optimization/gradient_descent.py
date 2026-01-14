# Math4AI: Calculus & Optimization - Assignment 5
# Starter Code Template
#
# Your file should be named:
# Math4AI_Calculus&Optimization_FirstName_LastName_Assignment5_code.py

import numpy as np
import matplotlib.pyplot as plt 

# ====================================================================
# --- Problem Setup ---
# ====================================================================

# We will train a linear model: y = m*x + b
# The parameters (theta) to be optimized are [m, b]

# Generate synthetic data for y = 2*x + 1 + noise
np.random.seed(42)
N = 100 # Number of data points
X = np.random.rand(N, 1) * 10 # X values from 0 to 10
y_true = 2 * X.squeeze() + 1
y = y_true + np.random.randn(N) * 1.5 # Add Gaussian noise

# --- Helper Functions (TO BE IMPLEMENTED) ---

def compute_loss(X, y, m, b):
    """
    Computes the Mean Squared Error (MSE) loss.
    L(m, b) = (1/N) * sum( (y_i - (m*x_i + b))^2 )

    Args:
        X (np.ndarray): Input features (N x 1).
        y (np.ndarray): True labels (N,).
        m (float): Current slope parameter.
        b (float): Current intercept parameter.

    Returns:
        float: The MSE loss, or None if not implemented.
    """
    # N = len(y)
    # --- YOUR CODE HERE ---

    # TODO 1: Calculate the predicted y values (y_pred = m*X + b).
    #         (Make sure to use X.squeeze() or handle array shapes)
    x_squeze=X.squeeze()
    y_pred=m*x_squeze + b

    # TODO 2: Calculate the error (y - y_pred).
    error=y-y_pred

    # TODO 3: Square the error.
    sqrt_error=error**2
    # TODO 4: Compute the mean of the squared errors.
    loss=np.mean(sqrt_error)

    return loss

def compute_gradient(X, y, m, b):
    """
    Computes the gradient of the MSE loss w.r.t. m and b.
    dL/dm = -(2/N) * sum( (y_i - (m*x_i + b)) * x_i )
    dL/db = -(2/N) * sum( (y_i - (m*x_i + b)) )

    Args:
        X (np.ndarray): Input features (N x 1).
        y (np.ndarray): True labels (N,).
        m (float): Current slope parameter.
        b (float): Current intercept parameter.

    Returns:
        (float, float): A tuple (grad_m, grad_b), or (None, None).
    """
    N = len(y)
    X_flat = X.squeeze()

    # --- YOUR CODE HERE ---
    predicted=m*X_flat + b
    # TODO 1: Calculate the error term (y_i - (m*x_i + b)).
    error=y-predicted

    # TODO 2: Calculate grad_m: -(2/N) * sum(error * x_i).
    sum_m=np.sum(error*X_flat)
    grad_m=-(2/N)*sum_m

    # TODO 3: Calculate grad_b: -(2/N) * sum(error).
    sum_b=np.sum(error)
    grad_b=-(2/N)*sum_b

    return grad_m, grad_b

def compute_hessian(X, y, m, b):
    """
    Computes the Hessian matrix of the MSE loss.
    H = [ [d2L/dm2,  d2L/dmdb],
          [d2L/dbdm, d2L/db2] ]

    d2L/dm2  = (2/N) * sum(x_i^2)
    d2L/db2  = 2
    d2L/dmdb = (2/N) * sum(x_i)

    Args:
        X (np.ndarray): Input features (N x 1).
        (y, m, b are not needed for MSE Hessian, but included for API consistency)

    Returns:
        np.ndarray: The 2x2 Hessian matrix, or None.
    """
    N = len(y)
    X_flat = X.squeeze()

    # --- YOUR CODE HERE ---

    # TODO 1: Calculate d2L/dm2.
    sum_dm2=np.sum(X_flat**2)
    hessian_dm2=(2/N)*sum_dm2

    # TODO 2: Calculate d2L/db2.
    hessian_db2=2
    # TODO 3: Calculate d2L/dmdb.
    sum_dmdb=np.sum(X_flat)
    hessian_dmdb=(2/N)*sum_dmdb

    hessian = np.array([
        [hessian_dm2, hessian_dmdb],
        [hessian_dmdb, hessian_db2]

    ])
    return hessian

# ====================================================================
# Part 2: Gradient-Based Optimization Methods
# ====================================================================

# --------------------------------------------------------------------
# Problem 2.1: Gradient Descent Variants
# --------------------------------------------------------------------

def batch_gradient_descent(X, y, lr, epochs):
    """
    Performs Batch Gradient Descent (BGD).

    Returns:
        (list, list): A tuple of (loss_history, param_history)
                      where param_history is a list of [m, b] pairs.
    """
    # Initialize parameters
    m, b = 0.0, 0.0
    loss_history = []
    param_history = [[m, b]]

    # --- YOUR CODE HERE ---
    # TODO: Loop from 0 to epochs-1
    for i in range(0,epochs):
        grad_m,grad_b=compute_gradient(X,y,m,b)
        m=m-lr*grad_m
        b=b-lr*grad_b

    #   1. Compute the gradient (grad_m, grad_b) using the *entire*
    #      dataset (X, y) by calling compute_gradient.
    #   2. Update m and b:
    #      m = m - lr * grad_m
    #      b = b - lr * grad_b
    #   3. Compute the loss for the new m, b using the *entire* dataset
    #      and append it to loss_history.
        loss_history.append(compute_loss(X,y,m,b))
    #   4. Append the new [m, b] to param_history.
        param_history.append([m,b])

    return loss_history, param_history

def stochastic_gradient_descent(X, y, lr, epochs):
    """
    Performs Stochastic Gradient Descent (SGD).

    Returns:
        (list, list): A tuple of (loss_history, param_history).
    """
    m, b = 0.0, 0.0
    N = len(y)
    loss_history = []
    param_history = [[m, b]]

    # --- YOUR CODE HERE ---
    # TODO: Loop from 0 to epochs-1
    for i in range(0,epochs):
        indices = np.random.permutation(N)#shuffe elemeycin
        x_shuffled = X[indices]
        y_shuffled = y[indices]

        for j in range(N):
            x_i=x_shuffled[j:j+1]
            y_i=y_shuffled[j:j+1]
            grad_m,grad_b=compute_gradient(x_i,y_i,m,b)
            m=m-lr*grad_m
            b=b-lr*grad_b # Fix: Changed grad_m to grad_b

        crnt_loss=compute_loss(X,y,m,b)
        loss_history.append(crnt_loss)
        param_history.append([m,b])
    #   1. (Optional, but recommended) Shuffle the data (X, y)
    #      at the start of *each* epoch.
    #   2. Start a *nested loop* that iterates through *each*
    #      data point (x_i, y_i) one by one.
    #   3. Inside the nested loop:
    #      a. Compute the gradient using *only* the single point (x_i, y_i).
    #         (Hint: You can call compute_gradient with X[i:i+1] and y[i:i+1])
    #      b. Update m and b.
    #   4. *After* the nested loop (at the end of the epoch):
    #      a. Compute the loss over the *entire* dataset
    #         and append it to loss_history.
    #      b. Append the current [m, b] to param_history.

    return loss_history, param_history

def minibatch_gradient_descent(X, y, lr, epochs, batch_size):
    """
    Performs Mini-Batch Gradient Descent.

    Returns:
        (list, list): A tuple of (loss_history, param_history).
    """
    m, b = 0.0, 0.0
    N = len(y)
    loss_history = []
    param_history = [[m, b]]

    # --- YOUR CODE HERE ---
    # TODO: Loop from 0 to epochs-1
    for i in range(0,epochs):
        indices = np.random.permutation(N)#suffle
        x_suffled = X[indices]
        y_suffled = y[indices]

        for j in range(0,N,batch_size):
            x_i=x_suffled[j:j+batch_size]
            y_i=y_suffled[j:j+batch_size]
            grad_m,grad_b=compute_gradient(x_i,y_i,m,b)
            m = m - lr * grad_m
            b = b - lr * grad_b

        loss=compute_loss(X,y,m,b)
        loss_history.append(loss)
        param_history.append([m,b])

    #   1. (Optional, but recommended) Shuffle the data (X, y)
    #      at the start of *each* epoch.
    #   2. Start a *nested loop* that iterates through the data
    #      in batches of 'batch_size'.
    #   3. Inside the nested loop:
    #      a. Get the current mini-batch (X_batch, y_batch).
    #      b. Compute the gradient using *only* this mini-batch.
    #      c. Update m and b.
    #   4. *After* the nested loop (at the end of the epoch):
    #      a. Compute the loss over the *entire* dataset
    #         and append it to loss_history.
    #      b. Append the current [m, b] to param_history.

    return loss_history, param_history

# --------------------------------------------------------------------
# Problem 2.2: Gradient Descent with Momentum
# --------------------------------------------------------------------

def minibatch_gd_with_momentum(X, y, lr, epochs, batch_size, beta):
    """
    Performs Mini-Batch GD with Momentum.

    Returns:
        (list, list): A tuple of (loss_history, param_history).
    """
    # Fix: Initialize m, b directly
    m, b = 0.0, 0.0
    N = len(y)
    loss_history = []
    param_history = [[m, b]]

    # Initialize velocity terms
    v_m, v_b = 0.0, 0.0

    # --- YOUR CODE HERE ---
    # TODO: Implement the Mini-Batch loop (as in 2.1).
    for i in range(0,epochs):
        # Fix: Use np.random.permutation for shuffling X and y together
        indices = np.random.permutation(N)
        x_shuffled = X[indices]
        y_shuffled = y[indices]
        for j in range(0,N,batch_size):
            x_batch=x_shuffled[j:j+batch_size]
            y_batch=y_shuffled[j:j+batch_size]
            grad_m,grad_b=compute_gradient(x_batch,y_batch,m,b)
            v_m=beta*v_m + (1-beta)*grad_m
            v_b=beta*v_b +(1-beta)*grad_b
            m=m - lr*v_m 
            b=b - lr*v_b
        param_history.append([m,b])
        loss=compute_loss(X,y,m,b)
        loss_history.append(loss)
    #   Inside the nested loop (for each batch):
    #   1. Compute the gradient (grad_m, grad_b) for the batch.
    #   2. Update the velocities (v_m, v_b) using the momentum formula:
    #      v_t = beta * v_(t-1) + (1 - beta) * gradient_t
    #   3. Update the parameters (m, b) using the velocities:
    #      param_t = param_(t-1) - lr * v_t

    return loss_history, param_history

# --------------------------------------------------------------------
# Problem 2.3: The Adam Optimizer
# --------------------------------------------------------------------

def adam_optimizer(X, y, lr, epochs, batch_size, beta1, beta2, epsilon):
    """
    Performs the Adam optimization algorithm.

    Returns:
        (list, list): A tuple of (loss_history, param_history).
    """
    m, b = 0.0, 0.0
    N = len(y)
    loss_history = []
    param_history = [[m, b]]

    # Initialize Adam's moment vectors
    m_m, m_b = 0.0, 0.0 # First moment (mean)
    v_m, v_b = 0.0, 0.0 # Second moment (variance)
    t = 0 # Timestep counter

    # --- YOUR CODE HERE ---
    # TODO: Implement the Mini-Batch loop (as in 2.1).
    for i in range(0,epochs):
        indices = np.random.permutation(N)
        x_shuffled = X[indices]
        y_shuffled = y[indices]

        for j in range(0,N,batch_size):
            x_i=x_shuffled[j:j+batch_size]
            y_i=y_shuffled[j:j+batch_size]
            t+=1
            grad_m,grad_b=compute_gradient(x_i,y_i,m,b)
            m_m=beta1*m_m +(1-beta1)*grad_m
            m_b=beta1*m_b +(1-beta1)*grad_b
            v_m=beta2*v_m +(1-beta2)*(grad_m**2)
            v_b=beta2*v_b +(1-beta2)*(grad_b**2)

            #bias
            m_hat=m_m/(1-beta1**t)
            b_hat=m_b/(1-beta1**t)
            vm_hat=v_m/(1-beta2**t)
            vb_hat=v_b/(1-beta2**t)

            m=m-lr*m_hat/(np.sqrt(vm_hat)+epsilon) 
            b=b-lr*b_hat/(np.sqrt(vb_hat)+epsilon)

            param_history.append([m,b])
        loss=compute_loss(X,y,m,b)
        loss_history.append(loss)
    #   Inside the nested loop (for each batch):
    #   1. Increment timestep t = t + 1.
    #   2. Compute gradient (grad_m, grad_b) for the batch.
    #   3. Update first moment (m_m, m_b):
    #      m_t = beta1 * m_(t-1) + (1 - beta1) * gradient
    #   4. Update second moment (v_m, v_b):
    #      v_t = beta2 * v_(t-1) + (1 - beta2) * (gradient^2)
    #   5. Compute bias-corrected moments (m_hat, v_hat):
    #      m_hat = m_t / (1 - beta1^t)
    #      v_hat = v_t / (1 - beta2^t)
    #   6. Update parameters (m, b):
    #      param = param - lr * m_hat / (sqrt(v_hat) + epsilon)

    return loss_history, param_history

# ====================================================================
# Part 3: Advanced Second-Order Methods
# ====================================================================

# --------------------------------------------------------------------
# Problem 3.1: Newton's Method
# --------------------------------------------------------------------

def newtons_method(X, y, epochs):
    """
    Performs Newton's method for optimization.

    Returns:
        (list, list): A tuple of (loss_history, param_history).
    """
    # theta = [m, b]
    theta = np.array([0.0, 0.0])
    loss_history = []
    param_history = [theta.copy()]

    # --- YOUR CODE HERE ---
    # TODO: Loop from 0 to epochs-1
    for i in range(0,epochs):
        m_crrnt,b_crrnt=theta[0],theta[1]
        grad_m,grad_b=compute_gradient(X,y,m_crrnt,b_crrnt)
        grad=np.array([grad_m,grad_b])
        hessian=compute_hessian(X,y,m_crrnt,b_crrnt)
        inv_hessian=np.linalg.inv(hessian)
        step=inv_hessian@grad
        theta=theta-step
        loss=compute_loss(X,y,theta[0],theta[1])
        loss_history.append(loss)
        param_history.append(theta.copy()) 
    #   1. Get current m and b from theta.
    #   2. Compute the gradient vector grad = [grad_m, grad_b]
    #      (Hint: Call compute_gradient).
    #   3. Compute the Hessian matrix H
    #      (Hint: Call compute_hessian).
    #   4. Compute the inverse of the Hessian (H_inv = np.linalg.inv(H)).
    #   5. Calculate the update step (H_inv @ grad).
    #   6. Update theta:
    #      theta = theta - update_step
    #   7. Compute and append the loss for the new theta.
    #   8. Append the new theta to param_history.

    return loss_history, param_history


# ====================================================================
# --- Main Execution & Verification ---
# ====================================================================

if __name__ == "__main__":

    print("=====================================================")
    print("Math4AI: Assignment 5 Verification")
    print("=====================================================")

    # --- Define Hyperparameters ---
    LR = 0.01
    EPOCHS = 100
    BATCH_SIZE = 16
    BETA = 0.9
    BETA1 = 0.9
    BETA2 = 0.999
    EPSILON = 1e-8

    # --- Run Optimizers ---

    # TODO:
    # 1. Call each of your optimizer functions (BGD, SGD, Mini-Batch,
    #    Momentum, Adam, Newton) and store their loss_history.
    #
    # Example:
    l_bgd, _ = batch_gradient_descent(X, y, LR, EPOCHS)
    l_sgd, _ = stochastic_gradient_descent(X, y, 0.001, EPOCHS)
    l_mbgd, _ = minibatch_gradient_descent(X, y, LR, EPOCHS, BATCH_SIZE)
    l_mom, _ = minibatch_gd_with_momentum(X, y, LR, EPOCHS, BATCH_SIZE, 0.9)
    l_adam, _ = adam_optimizer(X, y, 0.1, EPOCHS, BATCH_SIZE, 0.9, 0.999, 1e-8)
    l_newton, _ = newtons_method(X, y, EPOCHS)
    # loss_bgd, params_bgd = batch_gradient_descent(X, y, LR, EPOCHS)
    # loss_sgd, params_sgd = stochastic_gradient_descent(X, y, 0.001, EPOCHS) # Note: SGD often needs a smaller LR
    # loss_mbgd, params_mbgd = minibatch_gradient_descent(X, y, LR, EPOCHS, BATCH_SIZE)
    # ... and so on for Momentum, Adam, and Newton.


    # --- Plotting ---

    # TODO:
    plt.figure(figsize=(10, 6))
    plt.plot(l_bgd, label="Batch GD")
    plt.plot(l_sgd, label="Stochastic GD (LR=0.001)")
    plt.plot(l_mbgd, label="Mini-Batch GD")
    plt.plot(l_mom, label="Momentum")
    plt.plot(l_adam, label="Adam (LR=0.1)")
    plt.plot(l_newton, label="Newton's Method", linestyle='--')

    plt.yscale('log')
    plt.xlabel("Epochs")
    plt.ylabel("Loss ( Log Scale)")
    plt.title("Convergence Comparison of Different Optimizers")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig("optimizer_comparison.png")
    plt.show()
    # 1. Create a new figure (plt.figure).
    # 2. Plot the loss_history for each optimizer on the same graph.
    #    (e.g., plt.plot(loss_bgd, label="BGD"))
    # 3. Use a logarithmic scale for the y-axis (plt.yscale('log'))
    #    to better see the differences in convergence.
    # 4. Add a title, x-label ("Epochs"), y-label ("Loss (log scale)"),
    #    and a legend (plt.legend()).
    # 5. Save the plot (plt.savefig) and display it (plt.show).

    print("--- End of Verification ---")