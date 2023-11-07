import numpy as np

x = np.array([[0, 1, 2], [3, 4, 5]])
#x = x.reshape((2, 3))
y = np.ones_like(x)
x = x[:,0]
print(x)
z = x
print(z.reshape(-1,1))