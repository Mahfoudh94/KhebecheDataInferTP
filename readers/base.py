import abc
from typing import Dict, Optional, List, Tuple
import re

import pandas


class ReaderBase(abc.ABC):
    @property
    @abc.abstractmethod
    def _file_extensions(self) -> List[str]:
        pass

    @property
    @abc.abstractmethod
    def _file_type_name(self) -> str:
        pass

    @classmethod
    def get_file_types(cls) -> Tuple[str, Optional[str]]:
        return cls._file_type_name, "|".join(
            f'*.{ext}' for ext in cls._file_extensions
        )

    def __init__(self):
        self.dataframe = pandas.DataFrame()

    @staticmethod
    @abc.abstractmethod
    def read(file) -> pandas.DataFrame:
        pass

class ReaderFactory():
    _models_registry: Dict[str, type[ReaderBase]] = {}
    _models_ext_registry: Dict[str, type[ReaderBase]] = {}

    @classmethod
    def register(cls, reader_class: type[ReaderBase]):
        if not isinstance(reader_class, type):
            raise TypeError("Only classes can be registered.")

        class_name = re.sub(
            r'(?<!^)(?=[A-Z][^A-Z])',
            '_', reader_class.__name__,
        ).lower()
        
        cls._models_registry[class_name] = reader_class

        for ext in reader_class._file_extensions:
            cls._models_ext_registry[ext] = reader_class

        return reader_class

    @classmethod
    def get_instance(cls, class_name: str) -> Optional[ReaderBase]:
        try:
            model_class = cls._models_registry[class_name]
            return model_class()
        except KeyError:
            print(f"Reader '{class_name}' not found in registry.")
            return None

    @classmethod
    def get_instance_by_extension(cls, ext: str) -> Optional[ReaderBase]:
        try:
            model_class = cls._models_ext_registry[ext]
            return model_class()
        except KeyError:
            print(f"Reader for '{ext}' not found in registry.")
            return None

    @classmethod
    def get_extensions(cls):
        return [
            reader.get_file_types()
            for reader in cls._models_registry.values()
        ]

    def list_models(self):
        return list(self._models_registry.keys())
