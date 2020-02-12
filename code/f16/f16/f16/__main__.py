import numpy as np
from f16 import sum_as_string, test, axpy, mult

print(sum_as_string(1, 2))
test()

x = np.array([1., 2., 3.])
y = np.array([4., 5., 6.])
print(axpy(7, x, y))
print(mult(8, x))
print(x)
