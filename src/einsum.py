import torch
import numpy as np

A = np.random.rand(3, 5)
B = np.random.rand(5, 2)
M = np.empty((3, 2))

# Combining multiple calls its even faster
# matrix multiplicationS
# A * B = M
# m(i,j) = sum(k)(a(i,k) * b(k,j))
# with einsum :
# M(i,j) = A(i,k) * B(k,j)( implicilty we know that
# we need to sum over k)
M = np.einsum("ik,kj -> ij", A, B)
# it measn than we summ over k as it does not appear in the output ij
print(M)

# outer product
a = np.random.rand(5)
b = np.random.rand(3)
outer = np.empty((5, 3))

outer = np.einsum("i,j -> ij", a, b)
print(outer)


# summing a vector

sum_v = np.einsum("i->", a)
print(sum_v)

# return unsumaxis in any order , change dimensions
A = np.ones((3, 5, 4))
B = np.einsum("ijk -> kij", A)
print(B.shape)


# EXAMPLESSSSS
X = torch.tensor(2, 3)

# PERMUTATION OF VECTORS
torch.einsum("ij -> ji", X)
# SUMMATION
torch.einsum("ij->", X)
torch.einsum("ij->j", X)
torch.einsum("ij->i", X)

# MATRIX VECTOR MULTIPLICATION

v = torch.tensor((1, 3))
torch.einsum("ij,kj->ik", X, v)

# MATRIX MATRIX multlication

torch.einsum("ij,kj -> ik", X, X)  # 2x3 X 2x3 = 2x2

# dot product vector

torch.einsum("i,i ->", X[0], X[1])

# dot product matrix

torch.einsum("ij,ij->", X, X)

# elemwise multiplication ( hadamard product)

torch.einsum("ij,ij->ij", X, X)

# BATCH MATRIX MULTIPLICATION

a = torch.rand((3, 2, 5))
b = torch.rand((3, 5, 3))

torch.einsum("ijk,ikl -> ijl")

# Diagonal
x = torch.rand((3, 3))
torch.einsum("ii->i", x)

# trace

torch.einsum("ii->", x)
