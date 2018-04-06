
import numpy as np
from random import random
import math

# sigmoide assim para evitar error de overflow
def sigmoid(gamma):
  if gamma < 0:
    return 1 - 1/(1 + math.exp(gamma))
  else:
    return 1/(1 + math.exp(-gamma))


#x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y = np.array([0, 1, 1, 1]) # porta OR
#y = np.array([[0, 0, 0, 1]]).T # porta AND

x = np.array([ [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1] ])

y = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]).T

print("x: \n", x)
print("y: \n", y)

D = x.shape[1]
w = [2*random() - 1 for i in range(D)] # [1xD]
b = 0.5

print("w: \n", w)
print("D: \n", D)

#print(b)

learning_rate = 0.01

print("zip: \n", list(zip(x, y)))

for step in range(1000):
    cost = 0
    for x_i, y_i in zip(x, y):

        y_pred = sum([x_i[d]*w[d] for d in range(D)]) + b
        
        y_pred = sigmoid(y_pred)

        print("y: ", y_i)
        print("y_pred: ", y_pred)

        error = y_i - y_pred

        print("Error: ", error)
        print("\n")

        w = [ w[d] + learning_rate*error*x_i[d] for d in range(D) ]

        #print("w\n", w)

        b = b + learning_rate*error
        
        cost += error**2
        
    #if step%10 == 0:
        #print('step {0}: {1}'.format(step, cost))

print('w: ', w)
print('b: ', b)
print('y_pred: {0}'.format(np.dot(x, np.array(w))+b))







