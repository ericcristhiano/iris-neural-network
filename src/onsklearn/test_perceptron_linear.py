import unittest
import pandas as pd

from normalizer import Normalizer 
from perceptron_linear import PerceptronLinear

class TestPerceptronMultiLayer(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.network = PerceptronLinear()
    cls.network.execute()
    cls.classifier = cls.network.get_classifier()
    cls.normalizer = Normalizer()

    dataset_test_setosa = cls.normalizer.dataset_setosa.loc[cls.normalizer.dataset_setosa.index >= 35]
    dataset_test_virginica = cls.normalizer.dataset_virginica.loc[cls.normalizer.dataset_virginica.index >= 35]

    cls.dataset_test = pd.concat([dataset_test_setosa, dataset_test_virginica])
    cls.x_test = cls.dataset_test.loc[:, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    cls.y_test = cls.dataset_test['Species']

    print('[score]: ' + str(cls.classifier.score(cls.x_test, cls.y_test)))

  def test_must_return_specie_correct(self):        
    '''Must return setosa specie''' 
    print(self.shortDescription())

    self.assertEqual([2], self.classifier.predict([self.x_test.values[14]]))
  
  def test_must_return_specie_uncorrect(self):   
    '''Must return virginica specie'''
    print(self.shortDescription())
    
    self.assertEqual([1], self.classifier.predict([self.x_test.values[20]]))

if __name__ == '__main__':
    unittest.main()