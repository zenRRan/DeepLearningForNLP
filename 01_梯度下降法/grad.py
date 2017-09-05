import numpy as np
import matplotlib.pyplot as plt
import math
'''散点数据'''
raw_X = [2104, 1416, 1534, 852]
raw_Y = [460, 232, 315, 178]

size = len(raw_X)


'''数据处理 -1 - 1之间'''
X = np.array(raw_X)
Y = np.array(raw_Y)
X = X - (1 / size) * X.sum()
Y = Y - (1 / size) * Y.sum()
X = 1 / X.max() * X
Y = 1 / Y.max() * Y
plt.scatter(X, Y)
raw_X = X
raw_Y = Y

plt.show()

'''初始化拟合函数 y = aX + b'''
a = 0.01
b = 0
X = np.vstack((np.array([X.data]), np.ones([1, size]))).T
Y = np.array([Y.data]).T
learning_rate = 0.1
parameters = np.array([[a],
                       [b]])
# print(X,'\n',Y)
# print(parameters)
def Prediction(X, paras):
    return X.dot(paras)

# plt.ion()
def draw(a, b, s):
    if s == 1:
        plt.ioff()
    plt.clf()
    plt.scatter(raw_X, raw_Y)
    plt.plot([-1, (1-b)/a], [-a+b, 1], color='g', linestyle='-')
    plt.pause(0.0005)
    plt.show()

Costs = []

maxStep = 70
for step in range(maxStep):
    print('Step: ', step, end='')

    '''梯度下降'''
    P = Prediction(X, parameters)
    # print(P)
    a_ = 1/size * np.sum((P - Y) * X)
    b_ = 1/size * np.sum(P - Y)

    a = a - learning_rate * a_
    b = b - learning_rate * b_
    parameters = np.array([[a],
                           [b]])
    last_P = Prediction(X, parameters)
    cost = abs(np.sum(last_P - Y))
    # if abs(last_P - np.sum(P)) < 10**-6:
    #     break
    Costs.append(cost*(10**16))
    print(' Cost:', cost, end='   ')
    print(parameters[0], parameters[1])
    # draw(parameters[0], parameters[1], maxStep-step)

plt.clf()
plt.plot(range(maxStep), Costs, linestyle='-')
plt.ylim(0, 10)
plt.show()







