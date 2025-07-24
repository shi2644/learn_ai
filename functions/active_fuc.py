import math


def sigmoid(inputs):
    z = 0
    for i in inputs:
        z = z + i

    a = 1 / ( 1 + (math.e ** -z))
    return a