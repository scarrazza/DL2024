#!/usr/bin/env python
import numpy as np

a = np.array([[0.5, -1], [-1, 2]], dtype=np.float32)

print("a:", a)
print("shape:", a.shape)
print("ndim:", a.ndim)

# create copy
b = a.flatten().copy()
print("deep copy:", b)

# assign 0 to even entries
b[0::2] = 0
print("deep copy:", b)

# check a not modified
print("a:", a)
