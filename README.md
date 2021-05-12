# Iris Neural Network

## Installation
For the first step it's necessary install all python requirements, using:
```
pip install -r requirements.txt
```

## Tests
For the run tests, just run the file with suffix test_, e.g. `src/onsklearn/test_perceptron_multi_layer.py`:
```
py src/onsklearn/test_<file>.py
```

If you wish run the visual tests, just run the file without test_ suffix:
```
py src/onsklearn/<file>.py
```

## Visual tests
After runs the file perceptron, the return must be as:

### Setosa from Virginica 
This example utilizes the Linear Perceptron, according this show belown.
![Visual tests](./samples/linear_tests.png)

### Virginica from Versicolor
This example utilizes the Multilayer Perceptron, according this show belown too.
![Visual tests](./samples/multilayer_tests.png)

## Graphic visualization
If you wish generate the graphic of decision region from dataset specified, just run:
```
py src/onsklearn/visualization.py
```

## Explanation
The first step is get the graphic of decision region from dataset. And got this:
![Decision Region Graphic](./samples/decision_region.png)

### Setosa from Virginica
![Decision Region Graphic](./samples/linear_graphic.png)
### Virginica from Versicolor
![Decision Region Graphic](./samples/not_linear_graphic.png)

With the graphic generated, it's possible notice that the dataset diff between setosa and virginica is not linear, instead of virginica from versicolor that according we can see is a linear graphic. It's importante notice that the diff from virginica and versicolor are more difficult to distinguish. 

### Linear Perceptron
The [Linear Perceptron](https://en.wikipedia.org/wiki/Perceptron) is an algorithm linear classifier, therefore the dataset fitted must be linearly separable. According to shown below, the algorithm function work with binary options, and nothing besides that. Thus, we can observe that a possible implementation for "Setosa from Virginica" can be this classifier.

### MultiLayer Perceptron
Different of Linear Perceptron the [MultiLayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) is a feedforward artificial neural network, that way this classifier uses multiple layers of the perceptrons.

This perceptron normally has three node's layer, a layer for input, hidden and output. The nodes (except for the input) uses a non linear function and uses a supervisioned technique (backpropagation for training). With this, the Multilayer Perceptron can distinguish data even when they are not linearly separated.