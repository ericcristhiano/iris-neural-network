import pandas as pd
from sklearn.neural_network import MLPClassifier

class PerceptronMultiLayer:
    csv_data = 'dataset/iris.csv'  # file from work data

    def __init__(self) -> None:
      self.__setup__()

    def __setup__(self) -> None:
      self.mlp = MLPClassifier(
        hidden_layer_sizes=10, 
        learning_rate_init=0.01, 
        max_iter=500, 
        random_state=105
      )
      self.dataset = pd.read_csv(self.csv_data)
      self.__normalize_data__()
      self.__separate_species__()    
      self.__set_data_train__()   

    '''
      @ remove the iris setosa from datase
      @ specifies the species as number
      @ reset the dataset index after droppes and convert all info to number
    '''
    def __normalize_data__(self) -> None:
      index_names = self.dataset[self.dataset['Species'] == 'Iris-setosa'].index
      self.dataset.drop(index_names, inplace=True)

      self.dataset.loc[self.dataset['Species'] == 'Iris-versicolor', 'Species'] = 0
      self.dataset.loc[self.dataset['Species'] == 'Iris-virginica', 'Species'] = 1
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
      self.dataset_versicolor = self.__separate_specie__(self.dataset.index < 50)
      self.dataset_virginica = self.__separate_specie__(self.dataset.index >= 50)
      self.dataset_versicolor = self.dataset_versicolor.reset_index()
      self.dataset_virginica = self.dataset_virginica.reset_index()

    '''
      set the first 35 of dataset to data for the network train
    '''
    def __set_data_train__(self):
      dataset_train_versicolor = self.dataset_versicolor.loc[self.dataset_versicolor.index < 35]
      dataset_train_virginica = self.dataset_virginica.loc[self.dataset_virginica.index < 35]
      self.dataset_train = pd.concat([dataset_train_versicolor, dataset_train_virginica])

    ''' 
      train the model
    '''
    def __train__(self):
      x_train = self.dataset_train.loc[:, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
      y_train = self.dataset_train['Species']
      self.mlp.fit(x_train, y_train)

    def get_classifier(self):
      return self.mlp

    def execute(self) -> None:
      return self.__train__()
