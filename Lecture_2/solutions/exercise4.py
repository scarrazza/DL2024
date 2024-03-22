#!/usr/bin/env python

# import libraries
import numpy as np
import matplotlib.pyplot as plt

# load data
data = np.loadtxt("../data4.dat")

# plot
plt.plot(data[:,0], data[:, 1], 'or')
plt.title("Charged particles")
plt.xlabel("x-coordinate")
plt.ylabel("y-coordinate")

# save file to disk
plt.savefig("output.png")

# open canvas
plt.show()
