import random
import matplotlib.pyplot as plt
import numpy as np
"""
     Perceptron.py -- simulate a neural net Perceptron
"""

class Perceptron:
    def __init__(self, nInputs, learningRate):
        self.nInputs = nInputs
        self.learningRate = learningRate
    #def 

""" TRAINING FUNCTION """


def train(inputs, weights, desired, bias, learningRate = 0.00001):
    """Sometimes, if both inputs are zero, the perceptron might produce an incorrect output.
    To avoid this, we give the perceptron an extra input with the value of 1.
    This is called a bias."""
    inputs.append(bias)

    guess = activate(inputs, weights)
    error = desired - guess
    if (error != 0):
        for i, input in enumerate(inputs):
            print("Weight Before: ", weights[i])
            print("Err: ", error)
            print("Input: ", input)
            weights[i] += learningRate * error * input
            print("weight After: ", weights[i]);

""" ASSIGNS RANDOM VALUE BETWEEN -1 & 1 FOR EACH INPUT """
def initRandomWeights(numOfInputs):
    """ create random weights between -1 and 1 for each input """
    weights = []
    i = 0
    while i <= numOfInputs:
        weights.append(round(random.uniform(-1, 1), 1))
        i += 1

    return weights

""" CREATES A PERCEPTRON by multiplying the weights against the inputs to check if it passes the activation function """
def activate(inputs, weights):
    sum = 0

    """ Loop through the weights, adding all of their sums -- 
        ([1 x 0.7] + [0 x 0.6] + [1 x 0.5], + [0 x 0.3] + [1 x 0.4])
    """
    for index, input in enumerate(inputs):
        print(input, index)
        sum += input * weights[index]
    
    """ The perceptron is activated if the sum > 1.5 """
    print("Sum of weight calculations: ", sum)
    if (sum > 1.5):
        return 1
    else:
        return 0

""" COMPUTE THE DESIRED ANSWERS BASED ON THE LINE FUNCTION y = x * 1.2 + 50 """
def calculateDesiredOutput(xPoints, yPoints):
    desired = []
    for i, x in enumerate(xPoints):
        desired.append(0)
        # if this condition is true, then the point is above the line graph
        if (yPoints[i] > f(xPoints[i])):
            desired[i] = 1
    return desired

""" CREATE THE LINE FUNCTION (ACTIVATION FUNCTION)"""
def f(x):
    return x * 1.2 + 50

""" Generate n amount of random points between xMax and yMax """
def createRandomPoints(n, xMax, yMax):
    xPoints = []
    yPoints = []

    i = 0
    while (i <= n):
        xPoints.append(random.randint(0, xMax))
        yPoints.append(random.randint(0, yMax))
        i += 1
    
    return {
        "xPoints": xPoints,
        "yPoints": yPoints
    }

"""************************** MAIN CODE *********************************************"""
# Get 500 random points between x = 100 and y = 100 to use as initial values
n = 500
learningRate = 0.00001
points = createRandomPoints(n, 100, 100)
xPoints = points['xPoints']
yPoints = points['yPoints']

# Calculate the desired output 
desired = calculateDesiredOutput(xPoints, yPoints)

# Seperate the "desired output" from the "not desired" output, this is to see what passes the activation function
xDesired = []
yDesired = []
xNot = []
yNot = []
i = 0
while (i < n):
    if (desired[i] == 1):
        xDesired.append(xPoints[i])
        yDesired.append(yPoints[i])
    else: 
        xNot.append(xPoints[i])
        yNot.append(yPoints[i])
    i += 1

# scatter plot the non desired output (under the line graph)
x0 = np.array(xNot)
y0 = np.array(yNot)
plt.scatter(x0, y0, color = 'hotpink')

# scatter plot the desired output (above the line)
x1 = np.array(xDesired)
y1 = np.array(yDesired)
plt.scatter(x1, y1)

# plot the line graph between x = -10 and x = 100
x = np.linspace(-10, 100)
y = x * 1.2 + 50
plt.plot(x, y)

# display the matplotlib
plt.show()


i = 0
j = 0
# Train the perceptron
while (i < 10000):
    while (j < n):
        train()
        j += 1
    i += 1
"""
weights = initRandomWeights(len(inputs))
desired = calculateDesiredOutput(inputs)
print("Weights: ", weights)
train(inputs, weights, 1.5, 1)
"""