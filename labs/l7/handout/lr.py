from re import S

import numpy as np
import argparse


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (?,)
    X : np.ndarray,     # shape (?, ?)
    y : np.ndarray,     # shape (?,)
    num_epoch : int, 
    learning_rate : float
) -> None:
    n, d = X.shape

    # intercept term
    # X -> (n, d+1)
    X_aug = np.hstack([np.ones((n, 1)), X])

    # params = 0
    theta[:] = 0.0

    # epoch cycles
    for _ in range(num_epoch):
        for i in range(n):
            
            xi = X_aug[i]
            yi = y[i]

            # ---- forward ----
            z = np.dot(xi, theta)
            y_pred = sigmoid(z)

            # gradient
            loss = y_pred - yi
            grad = loss * xi  # shape (d+1,)

            # update params
            theta -= learning_rate * grad


def predict(
    theta : np.ndarray, # shape (?,)
    X : np.ndarray      # shape (?, ?)
) -> np.ndarray:
    n = X.shape[0]
    
    # intercept term
    X_aug = np.hstack([np.ones((n, 1)), X])   # (n, d+1)
    
    z = X_aug @ theta                         # (n,)
    y_pred = sigmoid(z)                       # (n,)
    
    y_pred = (y_pred >= 0.5).astype(float)  # thresholding at 0.5
    
    return y_pred


def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    """
    Binary Cross-Entropy Loss (Log Loss)
    """
    # TODO: Implement `compute_error` using vectorization
    # loss = -np.mean(
    #     y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
    # )

    loss = np.mean(y_pred != y)
    
    return float(loss)

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()
    """
    paths = [
        "my/test_formatted.tsv",
        "my/val_formatted.tsv",
        "my/train_formatted.tsv"
    ]

    sample_paths = [
        'smalloutput/sample_formatted_test_small.tsv',
        'smalloutput/sample_formatted_val_small.tsv',
        'smalloutput/sample_formatted_train_small.tsv'
    ]

    sample_lg_paths = [
        'largeoutput/sample_formatted_test_large.tsv',
        'largeoutput/sample_formatted_val_large.tsv',
        'largeoutput/sample_formatted_train_large.tsv'
    ]

    def read_tsv(path : str):
        with open(path, 'r') as f:
            lines = f.readlines()
            data = [[float(x) for x in line.strip().split('\t')] for line in lines]
            return np.array(data)
        
    test_data = read_tsv(paths[0])
    val_data = read_tsv(paths[1])
    train_data = read_tsv(paths[2])

    test_data = read_tsv(sample_lg_paths[0])
    val_data = read_tsv(sample_lg_paths[1])
    train_data = read_tsv(sample_lg_paths[2])


    y_te = test_data[:, 0] 
    X_te = test_data[:, 1:]
    y_val = val_data[:, 0]
    X_val = val_data[:, 1:]
    y_tr = train_data[:, 0]
    X_tr = train_data[:, 1:]

    epoch = 500
    learning_rate = 0.1
    theta = np.zeros(X_tr.shape[1] + 1)

    train(
        theta=theta, 
        X=X_tr, 
        y=y_tr, 
        num_epoch=epoch, 
        learning_rate=learning_rate
    )

    print("Trained theta:", len(theta), "\n", theta)

    y_tr_pred = predict(theta, X_tr)
    y_te_pred = predict(theta, X_te)

    print("Predicted probabilities on training set:", y_tr_pred)
    print("Predicted probabilities on test set:", y_te_pred)

    with open("my/train_labels.txt", 'w') as f:
        for y in y_tr_pred:
            f.write(f"{y}\n")

    with open("my/test_labels.txt", 'w') as f:
        for y in y_te_pred:
            f.write(f"{y}\n")

    tr_error = compute_error(y_tr_pred, y_tr)
    test_error = compute_error(y_te_pred, y_te)

    with open("my/metrics.txt", 'w') as f:
        f.write(f"error(train): {tr_error:.6f}\n")
        f.write(f"error(test): {test_error:.6f}\n")

    print(f"Validation Error: {tr_error:.6f}")
    print(f"Test Error: {test_error:.6f}")

    """