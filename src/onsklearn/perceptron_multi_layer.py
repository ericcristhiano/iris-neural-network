import pandas as pd
from normalizer import Normalizer 
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
      self.normalizer = Normalizer()
      self.__set_data_train__()   

    '''
      set the first 35 of dataset to data for the network train
    '''
    def __set_data_train__(self):
      dataset_train_versicolor = self.normalizer.dataset_versicolor.loc[self.normalizer.dataset_versicolor.index < 35]
      dataset_train_virginica = self.normalizer.dataset_virginica.loc[self.normalizer.dataset_virginica.index < 35]
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
