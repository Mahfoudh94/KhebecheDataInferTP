from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

from models.base import ModelBase, ModelFactory


@ModelFactory.register
class ClassificationModel(ModelBase):
    @property
    def is_classifier(self):
        return True

    def train(self, X_train, y_train):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(len(set(y_train)), activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)
        self.history = history
        return self.model, history

    def predict(self, X_test):
        return self.model.predict(X_test)

    def bootstrap(self, X_train, y_train, n_bootstraps):
        pass
