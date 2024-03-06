from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        shift = -self.loss_derivative(y, predictions)
        
        bootstrap_indices = np.random.choice(x.shape[0], int(x.shape[0] * self.subsample), replace=True)
        x_boot, shift_boot = x[bootstrap_indices], shift[bootstrap_indices]

        model = self.base_model_class(**self.base_model_params)
        model.fit(x_boot, shift_boot)
        model_predictions = model.predict(x)

        self.gammas.append(self.find_optimal_gamma(y, predictions, model_predictions))
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        best_valid_loss = 10000000000
        no_improve_rounds = 0
        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            train_predictions += self.models[-1].predict(x_train) * self.learning_rate * self.gammas[-1]
            valid_predictions += self.models[-1].predict(x_valid) * self.learning_rate * self.gammas[-1]
            train_loss = self.loss_fn(train_predictions, y_train)
            valid_loss = self.loss_fn(valid_predictions, y_valid)
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

            if self.early_stopping_rounds is not None:
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                else:
                    no_improve_rounds += 1

                if no_improve_rounds == self.early_stopping_rounds:
                    break

        if self.plot:
            plt.figure(figsize = (10, 6))
            plt.plot(self.history['train_loss'], label = 'train_loss')
            plt.plot(self.history['valid_loss'], label = 'valid_loss')
            plt.title("Boosting results")
            plt.ylabel("Loss")
            plt.xlabel("n_estimator")
            plt.legend()
            plt.show()

    def predict_proba(self, x):
      predictions = np.zeros(x.shape[0])
      for gamma, model in zip(self.gammas, self.models):
        predictions += model.predict(x) * self.learning_rate * gamma

      predictions = self.sigmoid(predictions)
      probs = np.zeros([x.shape[0], 2])
      probs[:, 0], probs[:, 1] = 1 - predictions, predictions
      return probs

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        feature_importances_avg = np.sum(model.feature_importances_ for model in self.models) / len(self.models)
        feature_importances_avg /= np.sum(feature_importances_avg)
        
        return feature_importances_avg
