import os
import re
import numpy as np
from scipy.io import loadmat
from sklearn import svm
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score
# ==============================
# 1. Предобработка писем
# ==============================
stemmer = PorterStemmer()


def preprocess_email(email_text):
    """Предобработка текста письма"""
    email_text = email_text.lower()
    email_text = re.sub(r'<[^<>]+>', ' ', email_text)  # HTML теги
    email_text = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_text)  # URL
    email_text = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_text)  # email
    email_text = re.sub(r'\d+', 'number', email_text)  # числа
    email_text = re.sub(r'\$', 'доллар', email_text)  # $

    words = re.split(r'\W+', email_text)
    words = [stemmer.stem(word) for word in words if word]
    return ' '.join(words)


# ==============================
# 2. Работа со словарем
# ==============================
def load_vocab(vocab_file='vocab.txt'):
    vocab = {}
    with open(vocab_file, 'r') as f:
        for line in f:
            index, word = line.strip().split()
            vocab[word] = int(index)
    print(f"Загружено слов: {len(vocab)}")
    return vocab


def words_to_indices(processed_text, vocab):
    words = processed_text.split()
    return [vocab[word] for word in words if word in vocab]


def email_to_feature_vector(processed_text, vocab):
    features = np.zeros(len(vocab))
    indices = words_to_indices(processed_text, vocab)
    for idx in indices:
        features[idx - 1] = 1
    return features


# ==============================
# 3. Загрузка данных spamTrain / spamTest
# ==============================
def load_spam_data(train_file='spamTrain.mat', test_file='spamTest.mat'):
    train_data = loadmat(train_file)
    X_train = train_data['X']
    y_train = train_data['y'].ravel()

    test_data = loadmat(test_file)
    X_test = test_data['Xtest']
    y_test = test_data['ytest'].ravel()

    return X_train, y_train, X_test, y_test


# ==============================
# 4. Обучение SVM
# ==============================
def train_svm(X_train, y_train, C=0.1):
    model = svm.SVC(C=C, kernel='linear', class_weight='balanced')
    model.fit(X_train, y_train)
    return model


def find_best_C(X_train, y_train, X_test, y_test, C_values=[0.01, 0.1, 1, 10]):
    best_C = None
    best_acc = 0
    for C in C_values:
        model = train_svm(X_train, y_train, C)
        acc = model.score(X_test, y_test)
        print(f"C={C}, точность: {acc * 100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_C = C
    print(f"Лучший C={best_C}, точность={best_acc * 100:.2f}%")
    return best_C, best_acc


def find_best_C_sigma(X_train, y_train, X_val, y_val,
                      C_values=[0.01, 0.1, 1, 10],
                      sigma_values=[0.01, 0.1, 1, 10]):
    best_score = 0
    best_C = None
    best_sigma = None

    for C in C_values:
        for sigma in sigma_values:
            gamma = 1 / (2 * sigma ** 2)
            model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            print(f"C={C}, σ²={sigma ** 2:.4f}, gamma={gamma:.5f}, accuracy={acc * 100:.2f}%")

            if acc > best_score:
                best_score = acc
                best_C = C
                best_sigma = sigma

    print(f"\n✅ Лучшие параметры: C={best_C}, σ²={best_sigma ** 2:.4f}, accuracy={best_score * 100:.2f}%")
    return best_C, best_sigma, best_score

# ==============================
# 5. Проверка классификации
# ==============================
def predict_email(text, model, vocab):
    processed = preprocess_email(text)
    features = email_to_feature_vector(processed, vocab).reshape(1, -1)
    pred = model.predict(features)[0]
    return "SPAM" if pred == 1 else "HAM"


# ==============================
# 6. Работа с собственным корпусом
# ==============================
def load_emails_from_folder(folder_path, label):
    emails, labels = [], []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if os.path.isfile(path):
            with open(path, 'r', encoding='latin1', errors='ignore') as f:
                content = f.read()
                emails.append(content)
                labels.append(label)
    return emails, labels


def build_vocab(emails, max_words=5000):
    all_words = []
    for email in emails:
        text = preprocess_email(email)
        all_words.extend(text.split())
    most_common = Counter(all_words).most_common(max_words)
    return {word: i + 1 for i, (word, _) in enumerate(most_common)}


def emails_to_features(emails, vocab):
    X = np.zeros((len(emails), len(vocab)))
    for i, email in enumerate(emails):
        X[i, :] = email_to_feature_vector(preprocess_email(email), vocab)
    return X


# ==============================
# 7. Основной блок
# ==============================
if __name__ == "__main__":
    # --- 13-16: spamTrain / spamTest ---
    X_train, y_train, X_test, y_test = load_spam_data()
    model_spam = train_svm(X_train, y_train, C=0.1)
    print(f"Train accuracy: {model_spam.score(X_train, y_train) * 100:.2f}%")
    print(f"Test accuracy: {model_spam.score(X_test, y_test) * 100:.2f}%")
    best_C, best_acc = find_best_C(X_train, y_train, X_test, y_test)

    best_C1, best_sigma1, best_acc1 = find_best_C_sigma(
        X_train, y_train, X_test, y_test,
        C_values=[0.01, 0.1, 1, 10],
        sigma_values=[0.01, 0.1, 1, 10]
    )

    # Обучим модель с найденными параметрами
    gamma = 1 / (2 * best_sigma1 ** 2)
    best_model = svm.SVC(C=best_C1, kernel='rbf', gamma=gamma)
    best_model.fit(X_train, y_train)

    print(f"Train accuracy: {best_model.score(X_train, y_train) * 100:.2f}%")
    print(f"Test accuracy: {best_model.score(X_test, y_test) * 100:.2f}%")

    vocab = load_vocab()
    sample_emails = ['emailSample1.txt', 'emailSample2.txt', 'spamSample1.txt', 'spamSample2.txt']
    for f in sample_emails:
        with open(f, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        label = predict_email(text, best_model, vocab)
        print(f"{f}: {label}")

    # --- 17-19: словарь и функции ---
    sample_emails = ['emailSample1.txt', 'emailSample2.txt', 'spamSample1.txt', 'spamSample2.txt']
    for f in sample_emails:
        with open(f, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        label = predict_email(text, model_spam, vocab)
        print(f"{f}: {label}")

    test_samples = [
            "Congratulations! You have won $1000. Claim your prize now!",
            """I am trying to rebuild the recently posted ALSA driver package for my 
    kernel.  Although I run Red Hat 7.3, I am not using a Red Hat kernel 
    package: my kernel is lovingly downloaded, configured, and built by 
    hand.  Call me old fashioned.""",
            "Get cheap meds at http://pharmacy.com now!",
            "Reminder: your subscription renewal is due next week."
        ]
    for i, text in enumerate(test_samples, 1):
        label = predict_email(text, model_spam, vocab)
        print(f"Sample {i}: {label}")

    # --- 21-24: собственный корпус ---
    ham_easy, labels_easy = load_emails_from_folder('./easy_ham', 0)
    ham_hard, labels_hard = load_emails_from_folder('./hard_ham', 0)
    spam_emails, labels_spam = load_emails_from_folder('./spam', 1)

    emails = ham_easy + ham_hard + spam_emails
    labels = labels_easy + labels_hard + labels_spam

    vocab_custom = build_vocab(emails)
    print(f"Custom vocab size: {len(vocab_custom)}")
    X_custom = emails_to_features(emails, vocab_custom)
    y_custom = np.array(labels)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_custom, y_custom, test_size=0.4, random_state=42,
                                                                stratify=y_custom)
    model_custom = svm.SVC(C=0.1, kernel='rbf', class_weight='balanced')
    model_custom.fit(X_train_c, y_train_c)

    train_acc = model_custom.score(X_train_c, y_train_c)
    test_acc = model_custom.score(X_test_c, y_test_c)
    print(f"Custom corpus train accuracy: {train_acc * 100:.2f}%")
    print(f"Custom corpus test accuracy: {test_acc * 100:.2f}%")

    # Проверка на примерах
    for i, text in enumerate(test_samples, 1):
        label = predict_email(text, model_custom, vocab_custom)
        print(f"Sample {i}: {label}")

