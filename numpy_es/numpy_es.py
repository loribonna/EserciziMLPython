import numpy as np
from numpy import ndarray

a = np.array([1, 2, 3, 4, 5, 6])
print(a)
even = a[a % 2 == 0]
print(even)
odd = a[a % 2 != 0]
print(odd)
b = even+odd
print(b)
c = np.arange(5)
print(c)
d: ndarray = np.triu(np.ones((3, 3)), 1)
print(d)
print(d.T)
d = d+d.T
print(d)
print("###")
x = np.array([[[1, -1], [2, -2]], [[3, -3], [4, -4]]])
print(x)
print(x[:, 0])
print(x[0, 1:])
print(x.sum(axis=2))
