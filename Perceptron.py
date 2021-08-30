import numpy as np

def my_training(input, weights, correct):
  threshold = 0
  num_iter = 100
  for _ in range(num_iter):
      for i in range(len(input)):
        sum = np.dot(input[i], weights)
        output = sum > threshold
        error = correct[i] - output
        weights = weights + error*input[i]
  return weights, threshold

class MyPerceptron:
  weights = []
  threshold = 0
  def __init__(self, dataset, expected, name):
    self.weights = np.zeros(len(dataset[0]))
    self.weights, self.threshold = my_training(dataset, self.weights, expected)
    self.name = name

  def predict(self, input):
    return np.dot(input,self.weights)>self.threshold

import numpy as np

Inputs = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
OR_Expected = np.array([0,1,1,1])
AND_Expected = np.array([0,0,0,1])
NOR_Expected = np.array([1,0,0,0])
NAND_Expected = np.array([1,1,1,0])

OR = MyPerceptron(Inputs, OR_Expected, "OR")
AND = MyPerceptron(Inputs, AND_Expected, "AND")
NOR = MyPerceptron(Inputs, NOR_Expected, "NOR")
NAND = MyPerceptron(Inputs, NAND_Expected, "NAND")

Logic_Gates = [OR, AND, NOR, NAND]


for i in Logic_Gates:
  for j in Inputs:
    print(j[1], i.name, j[2], "=", i.predict(j))
  print()