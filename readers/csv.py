import pandas

from readers.base import ReaderBase, ReaderFactory


@ReaderFactory.register
class CSVReader(ReaderBase):
    _file_type_name = "CSV Files"
    _file_extensions = ['csv']

    @staticmethod
    def read(file) -> pandas.DataFrame:
        return pandas.read_csv(file)