import pandas as pd
from sklearn.neural_network import MLPClassifier


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

dataset_train_versicolor = dataset_versicolor.loc[dataset_versicolor.index < 35]
dataset_train_virginica = dataset_virginica.loc[dataset_virginica.index < 35]

dataset_test_versicolor = dataset_versicolor.loc[dataset_versicolor.index >= 35]
dataset_test_virginica = dataset_virginica.loc[dataset_virginica.index >= 35]

dataset_train = pd.concat([dataset_train_versicolor, dataset_train_virginica])
dataset_test = pd.concat([dataset_test_versicolor, dataset_test_virginica])

# training_set = dataset.values
#
X_train = dataset_train.loc[:, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y_train = dataset_train['Species']

X_test = dataset_test.loc[:, ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y_test = dataset_test['Species']

# # X_train, X_test, y_train, y_test = train_test_split(training_set[:, :4],
# #                                                     training_set[:, 4],
# #                                                     test_size=0.2)

mlp = MLPClassifier(hidden_layer_sizes=10,
                    learning_rate_init=0.01,
                    max_iter=500,
                    random_state=105)


# Train the model
mlp.fit(X_train, y_train)
# Test the model
print(mlp.score(X_test, y_test))
# print(X_test)
print(X_test.values[14])
print(y_test.values[14])
print(mlp.predict([X_test.values[14]]))

# print()
