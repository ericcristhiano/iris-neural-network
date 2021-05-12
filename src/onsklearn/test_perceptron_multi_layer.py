import unittest
import pandas as pd

from perceptron_multi_layer import PerceptronMultiLayer

class TestPerceptronMultiLayer(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.network = PerceptronMultiLayer()
    cls.network.execute()
    cls.classifier = cls.network.get_classifier()
    
    dataset = pd.read_csv('dataset/iris.csv')
    index_names = dataset[dataset['Species'] == 'Iris-setosa'].index
    dataset.drop(index_names, inplace=True)
    dataset.loc[dataset['Species'] == 'Iris-versicolor', 'Species'] = 0
    dataset.loc[dataset['Species'] == 'Iris-virginica', 'Species'] = 1
    dataset.drop('Id', inplace=True, axis=1)
    dataset = dataset.reset_index()
    dataset = dataset.apply(pd.to_numeric)
    dataset_versicolor = dataset.loc[dataset.index < 50]
    dataset_virginica = dataset.loc[dataset.index >= 50]
    dataset_versicolor = dataset_versicolor.reset_index()
    dataset_virginica = dataset_virginica.reset_index()
    dataset_test_versicolor = dataset_versicolor.loc[dataset_versicolor.index >= 35]
    dataset_test_virginica = dataset_virginica.loc[dataset_virginica.index >= 35]
    cls.dataset_test = pd.concat([dataset_test_versicolor, dataset_test_virginica])
    cls.x_test = cls.dataset_test.loc[:, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    cls.y_test = cls.dataset_test['Species']

    print('[score]: ' + str(cls.classifier.score(cls.x_test, cls.y_test)))

  def test_must_return_specie_correct(self):        
    '''Must return Versicolor specie''' 
    print(self.shortDescription())

    self.assertEqual([0], self.classifier.predict([self.x_test.values[14]]))
  
  def test_must_return_specie_uncorrect(self):   
    '''Must return virginica specie'''
    print(self.shortDescription())
    
    self.assertEqual([1], self.classifier.predict([self.x_test.values[20]]))

if __name__ == '__main__':
    unittest.main()