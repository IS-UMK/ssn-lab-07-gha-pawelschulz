import numpy as np

class GHA(object):
    
    def __init__(self, n_components=1, eta=0.001, n_epochs=100):
        self.eta = eta
        self.n_components = n_components
        self.n_epochs = n_epochs
        
    def init(self, X):
        # TODO inicjuj macierz wag self.W o kształcie [n_components, ilość zmiennych ] 
        n_features = X.shape[1] # Liczba cech wejściowych
        # Losowa inicjalizacja wag
        self.W = np.random.randn(self.n_components, n_features) * 0.01 # Losowa inicjalizacja małymi wartościami
        return self

    def fit(self, X):
        # TODO algorytm uczenia GHA
        self.init(X)
        for epoch in range(self.n_epochs):
            for x in X:
                x = x.reshape(-1, 1)  # zmiana na wekor kolumnowy (784x1)
                y = self.W @ x        # Obliczenie aktywacji neuronów: y = W * x
                y = y.reshape(-1, 1)  # y na postać kolumnową
                
                #obliczenie zmiany wag według wzoru
                delta_W = self.eta * ((y @ x.T) - np.tril(y @ y.T) @ self.W)
                self.W += delta_W # aktualizacja wag
        return self
            
    def transform(self, X):
        # Rzutowanie danych X do przestrzeni głównych komponentów (nowe współrzędne w tej przestrzeni)
        return (self.W @ X.T).T # X.T: [cechy x próbki] → wynik transponujemy do [próbki x komponenty]

    def inverse_transform(self, Y):
        # Odtworzenie oryginalnych danych na podstawie współrzędnych Y w przestrzeni komponentów
        return Y @ self.W # Mnożenie współrzędnych przez macierz wag daje rekonstrukcję
