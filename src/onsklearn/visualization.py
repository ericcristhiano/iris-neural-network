import matplotlib.pyplot as plt
from normalizer import Normalizer 

class Visualization:
  virginica = 'virginica'
  versicolor = 'versicolor'
  setosa = 'setosa'

  def __init__(self, dataset_types = [virginica, versicolor, setosa]) -> None:
    self.dataset_types = dataset_types
    self.normalizer = Normalizer()

  def __configure_graphic_data__(self, x):
    if self.setosa in self.dataset_types:
      plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')

    if self.versicolor in self.dataset_types:
      plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='o', label='versicolor')
    
    if self.virginica in self.dataset_types:
      plt.scatter(x[100:150, 0], x[100:150, 1], color='green', marker='o', label='virginica')

  def __visualize__(self, x, xlabel, ylabel):
    self.__configure_graphic_data__(x)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.show()

  def visualize_length_diff(self):
    self.__visualize__(self.normalizer.dataset.iloc[0:150, [1, 3]].values, 'petal length', 'sepal length')

  def visualize_width_diff(self):
    self.__visualize__(self.normalizer.dataset.iloc[0:150, [2, 5]].values, 'petal width', 'sepal width')

if __name__ == '__main__':
  visualization = Visualization()
  visualization.visualize_length_diff()
  visualization.visualize_width_diff()

