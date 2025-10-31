import numpy as np


class LinearRegressionRidge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.column_stack([np.ones(X.shape[0]), X])
        I = np.eye(X.shape[1])
        I[0, 0] = 0

        self.weights = np.linalg.solve(X.T @ X + self.alpha * I, X.T @ y)
        self.b = self.weights[0]
        self.w = self.weights[1:]

        return self

    def predict(self, X):
        return X @ self.w + self.b


class LinearClassifierRidge(LinearRegressionRidge):
    def __init__(self, alpha=1.0):
        super().__init__(alpha)

    def fit(self, X, y):
        if np.array_equal(np.unique(y), [-1, 1]):
            raise ValueError("Expected y to be in {-1, 1}")

        return super().fit(X, y)

    def predict(self, X):
        return np.where(super().predict(X) >= 0, 1, -1)


class LinearClassifierGD:
    def __init__(
        self,
        loss_type="logistic",
        l1_ratio=0.0,
        alpha=0.01,
        learning_rate=0.01,
        max_iter=1000,
    ):
        if l1_ratio < 0 or l1_ratio > 1:
            raise ValueError("l1_ratio must be between 0 and 1")
        if loss_type not in ["logistic", "hinge", "perceptron"]:
            raise ValueError("loss_type must be in {'logistic', 'hinge', 'perceptron'}")

        self.loss_type = loss_type
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.train_loss_history = []
        self.test_loss_history = []
        self.w = None
        self.b = None

    def _loss(self, margins):
        if self.loss_type == "logistic":
            data_loss = np.mean(np.log(1 + np.exp(-margins)))
        elif self.loss_type == "hinge":
            data_loss = np.mean(np.maximum(0, 1 - margins))
        elif self.loss_type == "perceptron":
            data_loss = np.mean(np.maximum(0, -margins))

        l1_penalty = self.alpha * self.l1_ratio * np.sum(np.abs(self.w))
        l2_penalty = self.alpha * (1 - self.l1_ratio) * np.sum(self.w**2)

        return data_loss + l1_penalty + l2_penalty

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        n_samples, n_features = X_train.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for i in range(self.max_iter):
            margins = y_train * (X_train @ self.w + self.b)

            self.train_loss_history.append(self._loss(margins))

            if self.loss_type == "logistic":
                dy = -y_train / (1 + np.exp(margins))
            elif self.loss_type == "hinge":
                dy = np.where(margins < 1, -y_train, 0)
            elif self.loss_type == "perceptron":
                dy = np.where(margins < 0, -y_train, 0)

            dw = X_train.T @ dy / n_samples
            dw += self.alpha * self.l1_ratio * np.sign(self.w)
            dw += 2 * self.alpha * (1 - self.l1_ratio) * self.w
            db = np.mean(dy)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if X_test is not None and y_test is not None:
                self.test_loss_history.append(
                    self._loss(y_test * (X_test @ self.w + self.b))
                )

        return self

    def predict(self, X):
        return np.where(X @ self.w + self.b >= 0, 1, -1)
