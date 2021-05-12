import pandas as pd
from normalizer import Normalizer 
from sklearn.linear_model import Perceptron

class PerceptronLinear:
    def __init__(self) -> None:
      self.__setup__()

    def __setup__(self) -> None:
      self.mlp = Perceptron()
      self.normalizer = Normalizer()
      self.__set_data_train__()   

    '''
      set the first 35 of dataset to data for the network train
    '''
    def __set_data_train__(self):
      dataset_train_setosa = self.normalizer.dataset_setosa.loc[self.normalizer.dataset_setosa.index < 35]
      dataset_train_virginica = self.normalizer.dataset_virginica.loc[self.normalizer.dataset_virginica.index < 35]

      self.dataset_train = pd.concat([dataset_train_setosa, dataset_train_virginica])

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
