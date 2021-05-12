import pandas as pd
from pandas.core.base import NoNewAttributesMixin



class Normalizer:
  csv_data = 'dataset/iris.csv'  # file from work data
  ordered_types = [
    'versicolor',
    'virginica',
    'setosa'
  ]

  def get_type_by_index(index) -> str:
    return Normalizer.ordered_types[index]

  def __init__(self) -> None:
    self.dataset = pd.read_csv(self.csv_data)
    self.__normalize_data__()
    self.__separate_species__()    

  '''
    @ remove the iris setosa from datase
    @ specifies the species as number
    @ reset the dataset index after droppes and convert all info to number
  '''
  def __normalize_data__(self) -> None:
    self.dataset.loc[self.dataset['Species'] == 'Iris-versicolor', 'Species'] = 0
    self.dataset.loc[self.dataset['Species'] == 'Iris-virginica', 'Species'] = 1
    self.dataset.loc[self.dataset['Species'] == 'Iris-setosa', 'Species'] = 2

    self.dataset.drop('Id', inplace=True, axis=1)

    self.dataset = self.dataset.reset_index()
    self.dataset = self.dataset.apply(pd.to_numeric)
    
  '''
    separates the dataset where clause for specie is setted
  '''
  def __separate_specie__(self, where):
    return self.dataset.loc[where]

  '''
    separates the dataset between species
  '''
  def __separate_species__(self):
    self.dataset_versicolor = self.__separate_specie__(self.dataset['Species'] == 0)
    self.dataset_virginica = self.__separate_specie__(self.dataset['Species'] == 1)
    self.dataset_setosa = self.__separate_specie__(self.dataset['Species'] == 2)
    self.dataset_versicolor = self.dataset_versicolor.reset_index()
    self.dataset_virginica = self.dataset_virginica.reset_index()
    self.dataset_setosa = self.dataset_setosa.reset_index()

