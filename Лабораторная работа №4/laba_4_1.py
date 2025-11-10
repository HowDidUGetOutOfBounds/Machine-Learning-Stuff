# ============================================
#  Лабораторная работа: Нейронная сеть (ex4)
#  Полный скрипт с шагами 1–17
# ============================================

import scipy.io as sio
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ----------------------------------------------------------
# 1️⃣ Загрузка данных ex4data1.mat
# ----------------------------------------------------------
data = sio.loadmat('ex4data1.mat')
X = data['X']
y = data['y'].ravel()
num_labels = 10

print("Форма X:", X.shape)
print("Форма y:", y.shape)
print("Уникальные метки:", np.unique(y))

# ----------------------------------------------------------
# 2️⃣ Загрузка весов нейросети ex4weights.mat
# ----------------------------------------------------------
weights = sio.loadmat('ex4weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

print("\nФорма Theta1:", Theta1.shape)
print("Форма Theta2:", Theta2.shape)

input_layer_size = X.shape[1]
hidden_layer_size = Theta1.shape[0]

print(f"""
Структура нейронной сети:
- Входной слой: {input_layer_size} нейронов
- Скрытый слой: {hidden_layer_size} нейронов
- Выходной слой: {num_labels} нейронов (классы 1–10)
""")

# ----------------------------------------------------------
# 3️⃣ Сигмоида и её производная
# ----------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    s = sigmoid(z)
    return s * (1 - s)

# ----------------------------------------------------------
# 4️⃣ One-hot кодирование
# ----------------------------------------------------------
Y_onehot = np.zeros((y.size, num_labels))
Y_onehot[np.arange(y.size), y - 1] = 1

# ----------------------------------------------------------
# 5️⃣ Прямое распространение
# ----------------------------------------------------------
def predict_nn(Theta1, Theta2, X):
    m = X.shape[0]
    a1 = np.concatenate([np.ones((m,1)), X], axis=1)
    z2 = a1 @ Theta1.T
    a2 = sigmoid(z2)
    a2_bias = np.concatenate([np.ones((m,1)), a2], axis=1)
    z3 = a2_bias @ Theta2.T
    a3 = sigmoid(z3)
    pred = np.argmax(a3, axis=1) + 1
    return pred

pred_nn = predict_nn(Theta1, Theta2, X)
accuracy_nn = np.mean(pred_nn == y) * 100
print(f"\nТочность нейросети: {accuracy_nn:.2f}%")

# ----------------------------------------------------------
# 6️⃣ Логистическая регрессия для сравнения
# ----------------------------------------------------------
logreg = OneVsRestClassifier(LogisticRegression(max_iter=500, solver='lbfgs'))
logreg.fit(X, y)
pred_lr = logreg.predict(X)
accuracy_lr = np.mean(pred_lr == y) * 100
print(f"Точность логистической регрессии: {accuracy_lr:.2f}%")

# ----------------------------------------------------------
# 7️⃣ Функция стоимости с L2-регуляризацией
# ----------------------------------------------------------
def nn_cost_function(Theta1, Theta2, X, Y, lambda_=0):
    m = X.shape[0]
    epsilon = 1e-7  # для стабилизации логарифма

    X_bias = np.concatenate([np.ones((m,1)), X], axis=1)
    z2 = X_bias @ Theta1.T
    a2 = sigmoid(z2)
    a2_bias = np.concatenate([np.ones((m,1)), a2], axis=1)
    z3 = a2_bias @ Theta2.T
    a3 = sigmoid(z3)

    term1 = -Y * np.log(a3 + epsilon)
    term2 = (1 - Y) * np.log(1 - a3 + epsilon)
    J = np.sum(term1 - term2) / m
    if lambda_ > 0:
        reg = (lambda_ / (2*m)) * (np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))
        J += reg
    return J

cost_reg = nn_cost_function(Theta1, Theta2, X, Y_onehot, lambda_=1)
print(f"Стоимость нейросети с L2-регуляризацией λ=1: {cost_reg:.4f}")

# ----------------------------------------------------------
# 8️⃣ Инициализация весов случайными малыми числами
# ----------------------------------------------------------
def rand_initialize_weights(L_in, L_out, epsilon_init=0.12):
    return np.random.rand(L_out, L_in + 1) * 2 * epsilon_init - epsilon_init

Theta1_init = rand_initialize_weights(input_layer_size, hidden_layer_size)
Theta2_init = rand_initialize_weights(hidden_layer_size, num_labels)

# ----------------------------------------------------------
# 9️⃣ Backpropagation с регуляризацией
# ----------------------------------------------------------
def nn_backprop_regularized(Theta1, Theta2, X, Y, lambda_=0):
    m = X.shape[0]
    a1 = np.concatenate([np.ones((m,1)), X], axis=1)
    z2 = a1 @ Theta1.T
    a2 = sigmoid(z2)
    a2_bias = np.concatenate([np.ones((m,1)), a2], axis=1)
    z3 = a2_bias @ Theta2.T
    a3 = sigmoid(z3)
    delta3 = a3 - Y
    delta2 = (delta3 @ Theta2[:,1:]) * sigmoid_gradient(z2)
    Delta1 = delta2.T @ a1
    Delta2 = delta3.T @ a2_bias
    Theta1_grad = Delta1 / m
    Theta2_grad = Delta2 / m
    if lambda_ > 0:
        Theta1_grad[:,1:] += (lambda_/m) * Theta1[:,1:]
        Theta2_grad[:,1:] += (lambda_/m) * Theta2[:,1:]
    return Theta1_grad, Theta2_grad

Theta1_grad, Theta2_grad = nn_backprop_regularized(Theta1_init, Theta2_init, X, Y_onehot, lambda_=1)
print("Форма Theta1_grad:", Theta1_grad.shape)
print("Форма Theta2_grad:", Theta2_grad.shape)

# ----------------------------------------------------------
# 10️⃣ Gradient checking (на подмножестве)
# ----------------------------------------------------------
def flatten_params(Theta1, Theta2):
    return np.concatenate([Theta1.ravel(), Theta2.ravel()])

def reshape_params(params, input_size, hidden_size, num_labels):
    Theta1 = params[:hidden_size*(input_size+1)].reshape(hidden_size, input_size+1)
    Theta2 = params[hidden_size*(input_size+1):].reshape(num_labels, hidden_size+1)
    return Theta1, Theta2

def gradient_checking_regularized(Theta1, Theta2, X, Y, lambda_=1, epsilon=1e-4):
    params = flatten_params(Theta1, Theta2)
    num_grad = np.zeros_like(params)
    perturb = np.zeros_like(params)
    for i in range(len(params)):
        perturb[i] = epsilon
        Theta1_plus, Theta2_plus = reshape_params(params + perturb, X.shape[1], Theta1.shape[0], Theta2.shape[0])
        Theta1_minus, Theta2_minus = reshape_params(params - perturb, X.shape[1], Theta1.shape[0], Theta2.shape[0])
        loss1 = nn_cost_function(Theta1_plus, Theta2_plus, X, Y, lambda_)
        loss2 = nn_cost_function(Theta1_minus, Theta2_minus, X, Y, lambda_)
        num_grad[i] = (loss1 - loss2) / (2 * epsilon)
        perturb[i] = 0
    Theta1_grad, Theta2_grad = nn_backprop_regularized(Theta1, Theta2, X, Y, lambda_)
    grad = flatten_params(Theta1_grad, Theta2_grad)
    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
    print(f"Относительная разница (λ={lambda_}): {diff:e}")
    return diff

subset_size = 5
X_sub = X[:subset_size, :]
Y_sub = Y_onehot[:subset_size, :]
Theta1_check = rand_initialize_weights(input_layer_size, hidden_layer_size)
Theta2_check = rand_initialize_weights(hidden_layer_size, num_labels)

gradient_checking_regularized(Theta1_check, Theta2_check, X_sub, Y_sub, lambda_=1)

# ----------------------------------------------------------
# 14️⃣ Обучение нейросети с L-BFGS-B
# ----------------------------------------------------------
initial_params = flatten_params(Theta1_init, Theta2_init)

def cost_func(params, input_size, hidden_size, num_labels, X, Y, lambda_):
    Theta1, Theta2 = reshape_params(params, input_size, hidden_size, num_labels)
    J = nn_cost_function(Theta1, Theta2, X, Y, lambda_)
    Theta1_grad, Theta2_grad = nn_backprop_regularized(Theta1, Theta2, X, Y, lambda_)
    grad = flatten_params(Theta1_grad, Theta2_grad)
    return J, grad

res = minimize(fun=lambda p: cost_func(p, input_layer_size, hidden_layer_size, num_labels, X, Y_onehot, lambda_=1),
               x0=initial_params,
               method='L-BFGS-B',
               jac=True,
               options={'maxiter': 200})

Theta1_opt, Theta2_opt = reshape_params(res.x, input_layer_size, hidden_layer_size, num_labels)

# ----------------------------------------------------------
# 15️⃣ Точность на обучающей выборке
# ----------------------------------------------------------
pred_train = predict_nn(Theta1_opt, Theta2_opt, X)
accuracy_train = np.mean(pred_train == y) * 100
print(f"\nТочность обученной нейросети: {accuracy_train:.2f}%")

# ----------------------------------------------------------
# 16️⃣ Визуализация скрытого слоя
# ----------------------------------------------------------
def display_hidden_layer(Theta1):
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < Theta1.shape[0]:
            img = Theta1[i, 1:].reshape(20, 20).T
            ax.imshow(img, cmap='gray')
            ax.axis('off')
    plt.show()

display_hidden_layer(Theta1_opt)

# ----------------------------------------------------------
# 17️⃣ Подбор параметра регуляризации λ
# ----------------------------------------------------------
lambdas = [0, 0.1, 1, 10, 100]
for lam in lambdas:
    print(f"\nРегуляризация λ = {lam}")
    res_lam = minimize(fun=lambda p: cost_func(p, input_layer_size, hidden_layer_size, num_labels, X, Y_onehot, lam),
                       x0=initial_params,
                       method='L-BFGS-B',
                       jac=True,
                       options={'maxiter': 100})
    Theta1_lam, Theta2_lam = reshape_params(res_lam.x, input_layer_size, hidden_layer_size, num_labels)
    display_hidden_layer(Theta1_lam)
