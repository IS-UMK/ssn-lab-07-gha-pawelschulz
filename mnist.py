import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from gha import GHA
from sklearn.utils import shuffle

# Wczytanie danych
mnist = fetch_openml('mnist_784', data_home='./dane/', parser='auto')
X = mnist.data.astype(np.float32) # konwersja na float32
y = mnist.target.astype(int)

# Normalizacja danych
X = StandardScaler().fit_transform(X)

# Wybierz 1000 losowych przykładów
X, y = shuffle(X, y, random_state=42)  # Mieszamy dane z ustalonym ziarniem losowości
X_sample = X[:1000]                     # wybieram 1000 przykładów
y_sample = y[:1000]

# Parametry modelu
n_components = 50 # neuronów
gha = GHA(n_components=n_components, eta=0.001, n_epochs=20)
gha.fit(X_sample) # Trenowanie modelu GHA na wybranym zbiorze

# Transformacja do przestrzeni o mniejszym wymiarze
X_transformed = gha.transform(X_sample)

# Odtworzenie oryginalnych danych
X_reconstructed = gha.inverse_transform(X_transformed)

# Wizualizacja kilku oryginalnych i odtworzonych obrazków
def show_images(original, reconstructed, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Oryginał
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.title("Oryginalny")
        plt.axis("off")

        # Rekonstrukcja
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray') # Zmieniamy 1D → 2D
        plt.title("Odtworzony")
        plt.axis("off")
    plt.show()

show_images(X_sample, X_reconstructed)

# Wizualizacja wektorów własnych (wag)
def show_components(W, n=10):
    plt.figure(figsize=(20, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(W[i].reshape(28, 28), cmap='gray')
        plt.title(f"PC {i+1}")
        plt.axis("off")
    plt.show()

show_components(gha.W, n=10)

# Wykres rozrzutu dla dwóch pierwszych składowych głównych
plt.figure(figsize=(8, 6))
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_sample, cmap='tab10', alpha=0.7, s=30)
plt.colorbar() # Pasek kolorów odpowiadający klasom cyfr
plt.title("Rozkład danych MNIST w 2D (GHA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
