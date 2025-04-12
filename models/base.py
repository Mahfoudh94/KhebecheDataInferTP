import abc


class ModelBase(abc.ABC):
    @property
    def is_classifier(self):
        return False

    def __init__(self):
        self.model = None
        self.history = None

    @abc.abstractmethod
    def train(self, X_train, y_train):
        pass
    @abc.abstractmethod
    def predict(self, X_test):
        pass

    @abc.abstractmethod
    def bootstrap(self, X_train, y_train, n_bootstraps):
        pass


class ModelFactory:
    _models_registry = {}

    @classmethod
    def register(cls, model_class: type[ModelBase]):
        cls._models_registry[model_class.__name__] = model_class

    @classmethod
    def get_models(cls):
        return cls._models_registry.items()

    @classmethod
    def get_keys(cls):
        return cls._models_registry.keys()

    @classmethod
    def create(cls, model_name: str):
        model_class = cls._models_registry[model_name]
        return model_class()
