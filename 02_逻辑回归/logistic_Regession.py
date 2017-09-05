import numpy as np
import matplotlib.pyplot as plt

raw_X = [1, 1, 1.5, 2, 2.5, 1.75]
raw_Y = [2, 2.5, 3, 1.7, 1.9, 2.5]
labels = [1, 1, 1, 0, 0, 0]
color_set = ['green', 'red']
colors = []
for i in labels:
    if i == 1:
        colors.append(color_set[0])
    else:
        colors.append(color_set[1])
for i in range(len(raw_X)):
    plt.scatter(raw_X[i], raw_Y[i], c=colors[i])
# plt.show()

'''规范到-1 -- 1之间'''
X = np.array(raw_X)
Y = np.array(raw_Y)
X = (X - np.average(X))
X = X * (1/X.max())
Y = (Y - np.average(Y))
Y = Y * (1/Y.max())
plt.clf()
for i in range(X.size):
    plt.scatter(X[i], Y[i], c=colors[i])
plt.show()

def sigmod(x):
    return 1 / (1+np.e**(-x))

data_X = np.vstack((X, Y))
data_label = np.array([labels])
W = np.array(np.zeros((1, 2)))
b = 0.
# print("data_X:")
# print(data_X)
# print("data_label:")
# print(data_label)
# print("W:")
# print(W)
# print("b:")
# print(b)

plt.ion()
def draw(pred):
    plt.clf()
    for i in range(len(raw_X)):
        if pred[0][i] <= 0.5:
            plt.scatter(X[i], Y[i], c='red')
        else:
            plt.scatter(X[i], Y[i], c='green')
    plt.show()
    plt.pause(0.01)

Costs = []
training_times = 20
for step in range(training_times):
    Z = np.dot(W, data_X) + b
    A = sigmod(Z)
    draw(A)
    # print(A)

    J = data_label*np.log(A) + (1-data_label)*np.log(1-A)
    J = 1/len(raw_X)*J.sum(axis=1)*10
    Costs.append(abs(J))

    dZ = A - data_label
    dW = 1/len(raw_X)*(data_X*dZ).sum(axis=1)
    db = 1/len(raw_X)*np.sum(dZ)
    W -= dW
    b -= db

    print("Cost: ", J)
plt.ioff()
plt.show()
plt.plot(range(training_times), Costs, '-')
plt.show()
