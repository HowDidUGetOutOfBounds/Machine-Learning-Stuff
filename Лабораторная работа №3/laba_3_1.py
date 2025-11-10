import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# =========================================================
# 1. Загрузка данных
# =========================================================
data = sio.loadmat('ex3data1.mat')
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

print("Размеры обучающей выборки:", X.shape)
print("Размеры валидационной выборки:", Xval.shape)
print("Размеры тестовой выборки:", Xtest.shape)

# =========================================================
# 2. График обучающей выборки
# =========================================================
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Training data')
plt.xlabel('X (уровень воды)')
plt.ylabel('y (объем)')
plt.title('График обучающей выборки')
plt.legend()
plt.grid(True)
plt.show()


# =========================================================
# 3. Функция стоимости с L2-регуляризацией
# =========================================================
def compute_cost(X, y, theta, reg_lambda):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    reg_term = (reg_lambda / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost + reg_term


# =========================================================
# 4. Градиентный спуск с L2-регуляризацией
# =========================================================
def gradient_descent(X, y, theta, alpha, num_iters, reg_lambda):
    m = len(y)
    for _ in range(num_iters):
        errors = X.dot(theta) - y
        gradient = (1 / m) * (X.T.dot(errors))
        gradient[1:] += (reg_lambda / m) * theta[1:]
        theta -= alpha * gradient
    return theta


# =========================================================
# 5. Линейная регрессия λ=0 (чистая)
# =========================================================
reg_lambda = 0

X_lin = np.c_[np.ones((X.shape[0], 1)), X]
Xval_lin = np.c_[np.ones((Xval.shape[0], 1)), Xval]
Xtest_lin = np.c_[np.ones((Xtest.shape[0], 1)), Xtest]

theta_lin = np.zeros((X_lin.shape[1], 1))
theta_lin = gradient_descent(X_lin, y, theta_lin, alpha=0.001, num_iters=5000, reg_lambda=reg_lambda)

y_pred_lin = X_lin.dot(theta_lin)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Training data')
plt.plot(X, y_pred_lin, 'r-', label='Linear regression λ=0')
plt.xlabel('X (уровень воды)')
plt.ylabel('y (объем)')
plt.title('Линейная регрессия λ=0 (пункт 5)')
plt.legend()
plt.grid(True)
plt.show()

print("Параметры линейной модели:", theta_lin.ravel())


# =========================================================
# 6. Функция добавления полиномиальных признаков
# =========================================================
def poly_features(X, p):
    X_poly = np.zeros((X.shape[0], p))
    for i in range(1, p + 1):
        X_poly[:, i - 1] = X[:, 0] ** i
    return X_poly


# =========================================================
# 7. Подготовка полиномиальных признаков с нормализацией
# =========================================================
def prepare_poly_features(X, p, mu=None, sigma=None):
    X_poly = poly_features(X, p)
    if mu is None or sigma is None:
        mu = np.mean(X_poly, axis=0)
        sigma = np.std(X_poly, axis=0)
    X_poly_norm = (X_poly - mu) / sigma
    X_poly_bias = np.c_[np.ones((X_poly_norm.shape[0], 1)), X_poly_norm]
    return X_poly_bias, mu, sigma


# =========================================================
# 8. Функция learning curves
# =========================================================
def learning_curve(X_train, y_train, X_val, y_val, alpha, num_iters, reg_lambda):
    m = len(y_train)
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    for i in range(1, m + 1):
        X_sub = X_train[:i, :]
        y_sub = y_train[:i]
        theta = np.zeros((X_train.shape[1], 1))
        theta = gradient_descent(X_sub, y_sub, theta, alpha, num_iters, reg_lambda)
        error_train[i - 1] = compute_cost(X_sub, y_sub, theta, 0)
        error_val[i - 1] = compute_cost(X_val, y_val, theta, 0)
    return error_train, error_val


# =========================================================
# 9–10. Полиномиальная регрессия p=8, λ=0
# =========================================================
p = 8
alpha = 0.001
num_iters = 5000

X_poly_train, mu_poly, sigma_poly = prepare_poly_features(X, p)
X_poly_val, _, _ = prepare_poly_features(Xval, p, mu_poly, sigma_poly)
X_poly_test, _, _ = prepare_poly_features(Xtest, p, mu_poly, sigma_poly)

theta0 = np.zeros((X_poly_train.shape[1], 1))
theta0 = gradient_descent(X_poly_train, y, theta0, alpha, num_iters, reg_lambda=0)

x_vals = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)
x_vals_poly = poly_features(x_vals, p)
x_vals_poly_norm = (x_vals_poly - mu_poly) / sigma_poly
x_vals_poly_bias = np.c_[np.ones((x_vals_poly_norm.shape[0], 1)), x_vals_poly_norm]
y_pred0 = x_vals_poly_bias.dot(theta0)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Training data')
plt.plot(x_vals, y_pred0, 'r-', label=f'λ=0')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Полиномиальная регрессия p=8, λ=0')
plt.legend()
plt.grid(True)
plt.show()

error_train0, error_val0 = learning_curve(X_poly_train, y, X_poly_val, yval, alpha, num_iters, 0)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(error_train0) + 1), error_train0, 'b-', label='Training error')
plt.plot(range(1, len(error_val0) + 1), error_val0, 'r-', label='Validation error')
plt.title('Learning Curves λ=0')
plt.xlabel('Количество примеров')
plt.ylabel('Ошибка')
plt.legend()
plt.grid(True)
plt.show()

# =========================================================
# 11. Модели λ=1 и λ=100
# =========================================================
for reg_lambda in [1, 100]:
    theta = np.zeros((X_poly_train.shape[1], 1))
    theta = gradient_descent(X_poly_train, y, theta, alpha, num_iters, reg_lambda)
    y_pred = x_vals_poly_bias.dot(theta)

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Training data')
    plt.plot(x_vals, y_pred, 'r-', label=f'λ={reg_lambda}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Полиномиальная регрессия p=8, λ={reg_lambda}')
    plt.legend()
    plt.grid(True)
    plt.show()

    e_train, e_val = learning_curve(X_poly_train, y, X_poly_val, yval, alpha, num_iters, reg_lambda)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(e_train) + 1), e_train, 'b-', label='Training error')
    plt.plot(range(1, len(e_val) + 1), e_val, 'r-', label='Validation error')
    plt.title(f'Learning Curves λ={reg_lambda}')
    plt.xlabel('Количество примеров')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.grid(True)
    plt.show()

# =========================================================
# 12. Подбор оптимального λ
# =========================================================
lambda_candidates = np.array([0, 0.01, 0.1, 1, 10, 100])
val_errors = []

for reg_lambda in lambda_candidates:
    theta = np.zeros((X_poly_train.shape[1], 1))
    theta = gradient_descent(X_poly_train, y, theta, alpha, num_iters, reg_lambda)
    cost_val = compute_cost(X_poly_val, yval, theta, 0)
    val_errors.append(cost_val)

plt.figure(figsize=(8, 6))
plt.plot(lambda_candidates, val_errors, 'bo-')
plt.xscale('log')
plt.xlabel('λ')
plt.ylabel('Ошибка на валидации')
plt.title('Подбор коэффициента регуляризации')
plt.grid(True)
plt.show()

best_lambda = lambda_candidates[np.argmin(val_errors)]
print("Оптимальный λ по валидационной выборке:", best_lambda)

# =========================================================
# 13. Ошибка на контрольной выборке
# =========================================================
theta_best = np.zeros((X_poly_train.shape[1], 1))
theta_best = gradient_descent(X_poly_train, y, theta_best, alpha, num_iters, best_lambda)
test_error = compute_cost(X_poly_test, ytest, theta_best, 0)
print("Ошибка на контрольной выборке:", test_error)
