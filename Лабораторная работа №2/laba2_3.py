# ex2data3_logistic_one_vs_all.py
# Improved default parameters for higher accuracy (~95%+)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def rand_select_show(X, y, example_width=20, example_height=20, examples_per_row=10):
    m = X.shape[0]
    labels = np.unique(y)
    chosen_idx = []
    for lbl in labels:
        idxs = np.where(y.ravel() == lbl)[0]
        if idxs.size > 0:
            chosen_idx.append(np.random.choice(idxs))
    target = examples_per_row * examples_per_row
    while len(chosen_idx) < target and len(chosen_idx) < m:
        i = np.random.randint(0, m)
        if i not in chosen_idx:
            chosen_idx.append(i)
    chosen_idx = np.array(chosen_idx)
    sel = X[chosen_idx, :]
    sel_labels = y[chosen_idx]
    n_examples = sel.shape[0]
    cols = examples_per_row
    rows = int(np.ceil(n_examples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = axes.ravel()
    for i in range(rows * cols):
        ax = axes[i]
        ax.axis('off')
        if i < n_examples:
            img = sel[i].reshape(example_height, example_width, order='F')
            ax.imshow(img.T, cmap='gray')
            lbl = sel_labels[i].item()
            digit = 0 if lbl == 10 else int(lbl)
            ax.set_title(str(digit), fontsize=8)
    plt.tight_layout()
    plt.show()

def lr_cost_function(theta, X, y, lambda_reg):
    m = y.size
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    h = sigmoid(X.dot(theta))
    eps = 1e-12
    cost = (-1.0 / m) * (y.T.dot(np.log(h + eps)) + (1 - y).T.dot(np.log(1 - h + eps))).squeeze()
    reg = (lambda_reg / (2.0 * m)) * np.sum(np.square(theta[1:]))
    J = cost + reg
    grad = (1.0 / m) * (X.T.dot(h - y))
    grad[1:] = grad[1:] + (lambda_reg / m) * theta[1:]
    return J, grad.ravel()

def lr_cost_for_minimize(theta, X, y, lambda_reg):
    J, grad = lr_cost_function(theta, X, y, lambda_reg)
    return J, grad

def one_vs_all(X, y, num_labels, lambda_reg, maxiter=100):
    m, n = X.shape
    X_with_bias = np.concatenate([np.ones((m, 1)), X], axis=1)
    all_theta = np.zeros((num_labels, n + 1))
    for c in range(1, num_labels + 1):
        print(f"Training classifier for label {c}...")
        initial_theta = np.zeros(n + 1)
        y_c = np.array((y.ravel() == c).astype(int))
        res = minimize(
            fun=lambda t: lr_cost_for_minimize(t, X_with_bias, y_c, lambda_reg)[0],
            x0=initial_theta,
            jac=lambda t: lr_cost_for_minimize(t, X_with_bias, y_c, lambda_reg)[1],
            method='TNC',
            options={'maxfun': maxiter}
        )
        all_theta[c - 1, :] = res.x
    return all_theta

def predict_one_vs_all(all_theta, X):
    m = X.shape[0]
    X_with_bias = np.concatenate([np.ones((m, 1)), X], axis=1)
    probs = sigmoid(X_with_bias.dot(all_theta.T))
    preds = np.argmax(probs, axis=1) + 1
    return preds.reshape(-1, 1)

def main(mat_filename='ex2data3.mat', lambda_reg=0.1, maxiter=100):
    data = loadmat(mat_filename)
    if 'X' in data and 'y' in data:
        X = data['X']
        y = data['y']
    else:
        X = None
        y = None
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 2 and v.shape[1] == 400:
                    X = v
                if v.ndim in (1, 2) and v.size == 5000:
                    y = v.reshape(-1, 1)
        if X is None or y is None:
            raise ValueError('Could not find X and y in the .mat file.')
    print(f'Loaded X with shape {X.shape} and y with shape {y.shape}')
    np.random.seed(42)
    rand_select_show(X, y)
    num_labels = 10
    print('Training one-vs-all logistic regression...')
    all_theta = one_vs_all(X, y, num_labels=num_labels, lambda_reg=lambda_reg, maxiter=maxiter)
    preds = predict_one_vs_all(all_theta, X)
    readable_preds = np.where(preds == 10, 0, preds)
    readable_y = np.where(y == 10, 0, y)
    accuracy = np.mean(readable_preds.ravel() == readable_y.ravel()) * 100
    print(f'Training set accuracy: {accuracy:.2f}%')
    return {'X': X, 'y': y, 'all_theta': all_theta, 'preds': preds, 'accuracy': accuracy}

if __name__ == '__main__':
    results = main(mat_filename='ex2data3.mat', lambda_reg=0.1, maxiter=100)