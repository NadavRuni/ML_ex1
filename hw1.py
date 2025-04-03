###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################

# imports
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform Standardization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The Standardized input data.
    - y: The Standardized true labels.
    """
    ###########################################################################
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (n instances over p features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (n instances over p+1).
    """
    ###########################################################################
    if X.ndim == 1:
        X = X.reshape(-1, 1)

        # Create column of ones
    ones = np.insert(X, 0, 1, axis=1)
    X = ones
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X


def compute_loss(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the loss associated with the current set of parameters (single number).
    """

    J = 0  # We use J for the loss.
    ###########################################################################

    n = X.shape[0]

    # Linear model prediction
    y_hat = X @ theta  # shape (n,)

    # Compute MSE loss
    J = (1 / (2 * n)) * np.sum((y_hat - y) ** 2)

    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J


def gradient_descent(X, y, theta, eta, num_iters):
    """
    Learn the parameters of the model using gradient descent using
    the training set. Gradient descent is an optimization algorithm
    used to minimize some (loss) function by iteratively moving in
    the direction of steepest descent as defined by the negative of
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the loss value in every iteration
    ###########################################################################
    n = len(y)

    for i in range(num_iters):
        if i % 1000 == 0:
            print(f'Iteration {i} completed')
        y_hat = X @ theta

        diff = y_hat - y  #  The error (how much we missed by)
        grad = (X.T @ diff) / n  # gradient



        theta -= eta * grad

        J = (1 / (2 * n)) * np.dot(diff, diff)  # compute_loss
        J_history.append(J)



    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """

    pinv_theta = []
    ###########################################################################
    # X^T transpose
    X_transpose = np.transpose(X)

    # (X^T*X)^-1 = a  invers and mult
    invres_of_mult = np.linalg.inv(X_transpose @ X)

    # out = a * X^T * y
    pinv_theta = invres_of_mult @ X_transpose @ y

    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta


def gradient_descent_stop_condition(X, y, theta, eta, max_iter, epsilon=1e-8):
    """
    Learn the parameters of your model using the training set, but stop
    the learning process once the improvement of the loss value is smaller
    than epsilon. This function is very similar to the gradient descent
    function you already implemented.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - max_iter: The maximum number of iterations.
    - epsilon: The threshold for the improvement of the loss value.
    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the loss value in every iteration
    ###########################################################################
    n = X.shape[0]

    for iteration in range(max_iter):
        if iteration % 1000 == 0:
            print(f'Iteration {iteration} completed')

        y_hat = X @ theta
        diff = y_hat - y
        grad = (X.T @ diff) / n

        theta -= eta * grad

        loss = (1 / (2 * n)) * np.sum(diff ** 2)
        J_history.append(loss)

        # Early stopping
        if iteration > 0 and abs(J_history[-1] - J_history[-2]) < epsilon:
            print(f"Stopping early at iteration {iteration}")

            #  pad J_history for full length
            last_J = J_history[-1]
            J_history.extend([last_J] * (max_iter - len(J_history)))
            break

    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def find_best_learning_rate(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of eta and train a model using
    the training dataset. Maintain a python dictionary with eta as the
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part.

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - eta_dict: A python dictionary - {eta_value : validation_loss}
    """

    etas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    eta_dict = {}  # {eta_value: validation_loss}
    ###########################################################################
    features_num = X_train.shape[1] #how many features
    theta_vector = np.zeros(features_num) #build the theta vector

    #for each eta return the best theta and the min loss
    for eta in etas:
        print(f'eta: {eta}')
        # Train using gradient descent
        theta, _ = gradient_descent_stop_condition(X_train, y_train, theta_vector, eta, iterations )

        # Compute loss on validation set
        loss_with_specific_eta = compute_loss(X_val, y_val, theta)

        # Save the loss of each eta
        eta_dict[eta] = loss_with_specific_eta

    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return eta_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_eta, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to
    select the most relevant features for a predictive model. The objective
    of this algorithm is to improve the model's performance by identifying
    and using only the most relevant features, potentially reducing overfitting,
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part.

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_eta: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    total_features = X_train.shape[1]
    ###########################################################################
    while len(selected_features) < 5:
        the_best_feature = None
        lowest_loss = float('inf')

        for i in range(total_features):
            if i in selected_features:
                continue

            chosen_features = selected_features+[i]

            sub_X_train = X_train[:, chosen_features]
            sub_X_val = X_val[:, chosen_features]

            X_train_bias = apply_bias_trick(sub_X_train)
            X_val_bias = apply_bias_trick(sub_X_val)

            theta = np.zeros(X_train_bias.shape[1])
            theta ,_ = gradient_descent(X_train_bias, y_train, theta, best_eta, iterations)

            loss = compute_loss(X_val_bias, y_val, theta)

            if loss < lowest_loss:
                lowest_loss = loss
                the_best_feature = i

        selected_features.append(the_best_feature)


    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (n instances over p features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    squared_features = pd.DataFrame({
        f"{col}^2": df[col] ** 2 for col in df.columns
    }, index=df.index)

    # Build interaction features
    interaction_data = {}
    feature_names = df.columns
    for i in range(len(feature_names) - 1):
        for j in range(i + 1, len(feature_names)):
            col1 = feature_names[i]
            col2 = feature_names[j]
            interaction_data[f"{col1}*{col2}"] = df[col1] * df[col2]

    interaction_features = pd.DataFrame(interaction_data, index=df.index)

    # Concatenate everything into df_poly
    df_poly = pd.concat([df_poly, squared_features, interaction_features], axis=1)
    df_poly = df_poly.copy()
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly
