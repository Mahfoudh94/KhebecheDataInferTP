from keras import Sequential
from keras.layers import Dense
from sklearn.utils import resample


from models.base import ModelBase, ModelFactory


@ModelFactory.register
class RegressionModel(ModelBase):
    @staticmethod
    def init_model(X_train):
        return Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ])


    def train(self, X_train, y_train):
        self.model = self.init_model(X_train)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history = self.model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)
        self.history = history
        return self.model, history

    def predict(self, X_test):
        return self.model.predict(X_test)

    def bootstrap(self, X_train, y_train, n_bootstraps=10):
        predictions = []
        histories = []

        for _ in range(n_bootstraps):
            X_sample, y_sample = resample(X_train, y_train)

            model = self.init_model(X_train)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            history = model.fit(X_sample, y_sample, epochs=50, validation_split=0.2, verbose=0)

            predictions.append(model.predict(X_train))
            histories.append(history)

        return predictions, histories
