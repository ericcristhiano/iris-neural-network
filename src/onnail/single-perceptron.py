import csv


def step_function(result):
    return 1 if result >= 0 else 0


class Perceptron:
    def __init__(self, number_of_inputs, learning_rate):
        self.__number_of_inputs = number_of_inputs
        self.__learning_rate = learning_rate
        self.__weights = [0] * self.__number_of_inputs
        self.__bias_input = 1
        self.__bias_weight = 0

    def train(self, inputs):
        epochs = 0
        has_error = True

        while has_error:
            has_error = False
            for x_group, expected in inputs:
                y = self.__calculate_y(x_group)
                total_error = expected - y

                if y != expected:
                    has_error = True

                self.__calculate_weights(total_error, x_group)
            epochs += 1

    def fit(self, x_group):
        summation = self.__bias_input * self.__bias_weight
        for index, x in enumerate(x_group):
            summation += x * self.__weights[index]
        return step_function(summation)

    def __calculate_weights(self, total_error, x_group):
        for i in range(self.__number_of_inputs):
            self.__weights[i] += total_error * x_group[i] * self.__learning_rate
            self.__bias_weight += total_error * self.__bias_input * self.__learning_rate

    def __calculate_y(self, x_group):
        summation = self.__bias_input * self.__bias_weight
        for index, x in enumerate(x_group):
            summation += x * self.__weights[index]
        return step_function(summation)


with open('dataset/iris.csv', 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

IRIS_SETOSA_VALUE = 0
IRIS_VIRGINICA_VALUE = 1

iris_setosa_virginica = [item for item in rows[1:] if int(item[0]) <= 50 or int(item[0]) > 100]
iris_setosa_virginica_dataset_normalized = [([float(prop) for prop in item[1:5]], IRIS_SETOSA_VALUE if item[5] == 'Iris-setosa' else IRIS_VIRGINICA_VALUE) for item in iris_setosa_virginica]


if __name__ == '__main__':
    iris_setosa_dataset = iris_setosa_virginica_dataset_normalized[:50]
    iris_virginica_dataset = iris_setosa_virginica_dataset_normalized[50:]

    training_set = iris_setosa_dataset[:35] + iris_virginica_dataset[:35]
    perceptron = Perceptron(4, 1)
    perceptron.train(training_set)

    test_dataset = iris_setosa_dataset[35:] + iris_virginica_dataset[35:]

    for iris, expected in test_dataset:
        fit = perceptron.fit(iris)
        iris_name = "Iris-setosa" if expected == IRIS_SETOSA_VALUE else "Iris-virginica"
        print("Entrada: {}\t | Esperado: {} ({})\t | Obtido: {}\t Acerto: {}".format(iris, expected, iris_name, fit, fit == expected))
