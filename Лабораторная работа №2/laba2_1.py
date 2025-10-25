# logistic_ex2_unified.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

# -----------------------------
# 1) Загрузка данных
# -----------------------------
def load_data(path='ex2data1.txt'):
    data = pd.read_csv(path, header=None)
    if data.shape[1] < 3:
        raise ValueError("Файл должен содержать по крайней мере 3 колонки: exam1, exam2, admitted")
    data = data.iloc[:, :3]
    data.columns = ['Exam1', 'Exam2', 'Admitted']
    return data

# -----------------------------
# 2) Нормализация признаков
# -----------------------------
def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def apply_normalize(X, mu, sigma):
    X = np.array(X)
    return (X - mu) / sigma

# -----------------------------
# Вспомогательные (визуализация)
# -----------------------------
def plot_data(X, y, ax=None):
    if ax is None:
        ax = plt.gca()
    pos = y == 1
    neg = y == 0
    ax.scatter(X[pos,0], X[pos,1], marker='+', label='Admitted (1)', s=60, linewidths=1.2)
    ax.scatter(X[neg,0], X[neg,1], marker='o', label='Not admitted (0)', edgecolors='k', facecolors='none', s=50)
    ax.set_xlabel('Exam 1 score')
    ax.set_ylabel('Exam 2 score')
    ax.legend()
    return ax

def sigmoid(z):
    z = np.array(z)
    # стабильный сигмоид
    return 1.0 / (1.0 + np.exp(-z))

# -----------------------------
# Стоимость и градиент (работают на X с bias)
# -----------------------------
def cost_function(theta, X, y):
    m = y.size
    theta = theta.reshape(-1)
    h = sigmoid(X.dot(theta))
    eps = 1e-15
    J = -(1.0/m) * ( y.dot(np.log(h + eps)) + (1 - y).dot(np.log(1 - h + eps)) )
    return J

def gradient(theta, X, y):
    m = y.size
    theta = theta.reshape(-1)
    h = sigmoid(X.dot(theta))
    grad = (1.0/m) * X.T.dot(h - y)
    return grad

# -----------------------------
# Градиентный спуск (обучаем на X с bias)
# -----------------------------
def gradient_descent(X, y, theta_init=None, alpha=0.1, num_iters=10000, tol=1e-6, verbose=False):
    m, n = X.shape
    if theta_init is None:
        theta = np.zeros(n)
    else:
        theta = theta_init.copy()
    J_history = []
    for i in range(num_iters):
        grad = gradient(theta, X, y)
        theta = theta - alpha * grad
        J = cost_function(theta, X, y)
        J_history.append(J)
        grad_norm = np.linalg.norm(grad)
        if verbose and (i % max(1, num_iters // 10) == 0):
            print(f"Iter {i:5d} | Cost {J:.6f} | ||grad|| {grad_norm:.3e}")
        if grad_norm < tol:
            if verbose:
                print(f"Converged at iter {i}, grad_norm {grad_norm:.3e}")
            break
    return theta, np.array(J_history)

# -----------------------------
# SciPy optimize wrapper (на X с bias)
# -----------------------------
def scipy_optimize(X, y, method='BFGS', theta_init=None, options=None):
    if theta_init is None:
        theta_init = np.zeros(X.shape[1])
    if options is None:
        options = {'maxiter':4000}
    if method == 'Nelder-Mead':
        res = optimize.minimize(fun=lambda t: cost_function(t, X, y),
                                x0=theta_init,
                                method='Nelder-Mead',
                                options={'maxiter':2000, 'disp': False})
    else:
        res = optimize.minimize(fun=lambda t: cost_function(t, X, y),
                                x0=theta_init,
                                jac=lambda t: gradient(t, X, y),
                                method=method,
                                options=options)
    if not res.success:
        print(f"Warning: {method} did not converge: {res.message}")
    return res

# -----------------------------
# predict: автоматически нормализует входные данные (если нужно)
# -----------------------------
def predict_proba(theta, X_input, mu=None, sigma=None):
    """
    theta — обученные параметры, соответствующие X с bias, где X использовался после normalization.
    X_input — либо X_with_bias, либо raw features (m x 2) (без bias).
    Если mu, sigma заданы — считаем, что X_input в оригинальных координатах и нормализуем.
    Если X_input уже содержит bias и имеет размерность совпадающую с theta, используем как есть.
    """
    X = np.array(X_input)
    # если передали без bias и mu/sigma заданы -> нормализуем
    if (X.ndim == 2 and X.shape[1] == theta.size - 1) and (mu is not None and sigma is not None):
        Xn = apply_normalize(X, mu, sigma)
        X_with_bias = np.hstack([np.ones((Xn.shape[0],1)), Xn])
    elif X.ndim == 1 and X.size == theta.size - 1 and (mu is not None and sigma is not None):
        Xn = apply_normalize(X, mu, sigma)
        X_with_bias = np.hstack([1.0, Xn])
    elif X.ndim == 2 and X.shape[1] == theta.size:
        X_with_bias = X
    elif X.ndim == 1 and X.size == theta.size:
        X_with_bias = X.reshape(1,-1)
    else:
        # Попытка автоматически определить: если mu/sigma не заданы, предполагаем X уже содержит bias
        X_with_bias = X
    return sigmoid(X_with_bias.dot(theta))

def predict(theta, X_input, mu=None, sigma=None, threshold=0.5):
    return (predict_proba(theta, X_input, mu, sigma) >= threshold).astype(int)

# -----------------------------
# Построение разделяющей прямой в оригинальных координатах
# -----------------------------
def plot_decision_boundary_original(theta, mu, sigma, ax, X_raw, label=None, color=None):
    # x1 в оригинальных координатах
    x1_vals = np.linspace(X_raw[:,0].min()-2, X_raw[:,0].max()+2, 200)
    # нормализуем
    x1_norm = (x1_vals - mu[0]) / sigma[0]
    # вычисляем x2_norm = -(theta0 + theta1*x1_norm)/theta2
    theta0, theta1, theta2 = theta[0], theta[1], theta[2]
    x2_norm = -(theta0 + theta1 * x1_norm) / theta2
    # вернём в оригинальную систему
    x2_vals = x2_norm * sigma[1] + mu[1]
    ax.plot(x1_vals, x2_vals, linestyle='--', label=label, color=color)

# -----------------------------
# Main
# -----------------------------
def main():
    # Загрузка
    data = load_data('ex2data1.txt')
    X_raw = data[['Exam1','Exam2']].values  # m x 2 (оригинальные)
    y = data['Admitted'].values
    m = y.size

    # Нормализация (все алгоритмы будут обучаться на этой нормализации)
    X_norm, mu, sigma = feature_normalize(X_raw)
    # добавляем bias
    X = np.hstack([np.ones((m,1)), X_norm])  # m x 3

    # Параметры обучения для GD (на нормализованных данных можно брать больший alpha)
    alpha = 0.5
    num_iters = 20000

    # Визуализация исходных данных
    fig, ax = plt.subplots(figsize=(8,6))
    plot_data(X_raw, y, ax=ax)
    ax.set_title('Exam scores and admission (training data)')

    # 1) Gradient Descent (на нормализованных данных)
    theta_gd, J_hist = gradient_descent(X, y, alpha=alpha, num_iters=num_iters, tol=1e-6, verbose=True)
    print("\nTheta (GD):", theta_gd)
    print("Final cost (GD):", cost_function(theta_gd, X, y))

    # 2) SciPy Nelder-Mead (на тех же X)
    res_nm = scipy_optimize(X, y, method='Nelder-Mead')
    theta_nm = res_nm.x
    print("\nTheta (Nelder-Mead):", theta_nm)
    print("Final cost (Nelder-Mead):", res_nm.fun)

    # 3) SciPy BFGS
    res_bfgs = scipy_optimize(X, y, method='BFGS')
    theta_bfgs = res_bfgs.x
    print("\nTheta (BFGS):", theta_bfgs)
    print("Final cost (BFGS):", res_bfgs.fun)

    # Рисуем разделяющие прямые в ОРИГИНАЛЬНЫХ координатах
    plot_decision_boundary_original(theta_gd, mu, sigma, ax, X_raw, label='Decision (GD)', color='tab:blue')
    plot_decision_boundary_original(theta_nm, mu, sigma, ax, X_raw, label='Decision (NM)', color='tab:green')
    plot_decision_boundary_original(theta_bfgs, mu, sigma, ax, X_raw, label='Decision (BFGS)', color='tab:orange')

    ax.legend()
    plt.show()

    # График J(θ) для GD
    plt.figure(figsize=(6,4))
    plt.plot(J_hist)
    plt.title('Cost vs iterations (Gradient Descent)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost J(θ)')
    plt.grid(True)
    plt.show()

    # Точности (predict автоматически использует mu,sigma)
    for name, th in [('GD', theta_gd), ('Nelder-Mead', theta_nm), ('BFGS', theta_bfgs)]:
        preds = predict(th, X_raw, mu, sigma)
        acc = 100.0 * np.mean(preds.flatten() == y)
        print(f"{name} accuracy on training set: {acc:.2f}%")

    # Примеры предсказаний (сырые оценки — функция сама нормализует)
    examples = np.array([[45, 85], [60, 70]])
    for name, th in [('GD', theta_gd), ('BFGS', theta_bfgs)]:
        probs = predict_proba(th, examples, mu, sigma)
        print(f"{name} predicted probabilities for {examples.tolist()}: {probs}")

if __name__ == "__main__":
    main()