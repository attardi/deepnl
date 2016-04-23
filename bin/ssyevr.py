#!/usr/bin/python

from __future__ import print_function
import numpy as np
from scipy.linalg.lapack import ssyevr

A = np.array([[ 0.67, -0.20,  0.19, -1.06,  0.46],
              [-0.20,  3.82, -0.13,  1.06, -0.48],
              [ 0.19, -0.13,  3.27,  0.11,  1.10],
              [-1.06,  1.06,  0.11,  5.86, -0.98],
              [ 0.46, -0.48,  1.10, -0.98,  3.54]
])

n = np.linalg.norm(A, axis=1)

print(A)

w,z,info = ssyevr(A, range='I', il=3, overwrite_a=1)

print(w)
print(z)
# z = (5 x 3)
print(A.dot(z))


