import matplotlib.pyplot as plt
from normalizer import Normalizer 

class Visualization:
  def __init__(self) -> None:
    self.normalizer = Normalizer()

  def __visualize__(self, x, xlabel, ylabel):
    plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='o', label='versicolor')
    plt.scatter(x[100:150, 0], x[100:150, 1], color='green', marker='o', label='virginica')
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

