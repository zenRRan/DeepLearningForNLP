import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-1, 1, 100)
y = 2.3*x*x + 3.5*x + 0.04

y_ = y + np.random.rand(len(x)) - 0.5

A = []
times = 2
for i in range(times+1):
    A.append(x**(times-i))

A = np.array(A).T
B = y_.reshape(y_.shape[0], 1)

w = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(B)

pred_y = A.dot(w)
print(w)

plt.scatter(x, y_)
plt.plot(x, y, 'k-')
plt.plot(x, pred_y, 'r-')
plt.show()