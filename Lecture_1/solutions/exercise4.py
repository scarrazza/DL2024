#!/usr/bin/env python

# point 1
v = []
for i in range(0,16):
    v.append(i)
print(f"v = {v}")

# point 2
w = []
w += [1.5, 4.5, 7.5]
print(f"w = {w}")

# point 3
m = {
    'name': 'neuralnet',
    'loss': 0.12,
    'weights': [10, 25, 5]
    }
print(f"m = {m}")

# point 4
mylist = [2, 6, 3, 8, 9, 11, -5]
mean = 0
for i in mylist:
    mean += i
mean /= len(mylist)
print(f"mean = {mean}")

# point 5
l = [2 ** n for n in range(10)]
print(f"l = {l}")
