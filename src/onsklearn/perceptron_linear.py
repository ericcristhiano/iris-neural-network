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

    def test_all(self, x_test, y_test):
      for (i, x) in enumerate(x_test.values):
        current = str(x)
        expected = str(Normalizer.get_type_by_index(y_test.values[i]))
        returned = str(Normalizer.get_type_by_index(self.mlp.predict([x])[0]))

        print(f'with {current} is expected "{expected}" and was returned "{returned}"')

    def execute(self) -> None:
      return self.__train__()

if __name__ == '__main__':
  network = PerceptronLinear()
  network.execute()
  classifier = network.get_classifier()
  normalizer = Normalizer()

  dataset_test_setosa = normalizer.dataset_setosa.loc[normalizer.dataset_setosa.index >= 35]
  dataset_test_virginica = normalizer.dataset_virginica.loc[normalizer.dataset_virginica.index >= 35]

  dataset_test = pd.concat([dataset_test_setosa, dataset_test_virginica])
  x_test = dataset_test.loc[:, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
  y_test = dataset_test['Species']

  network.test_all(x_test, y_test)