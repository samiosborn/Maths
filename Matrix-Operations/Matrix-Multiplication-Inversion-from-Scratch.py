# Matrix multiplication and inversion from scratch (without numpy)
import numpy as np

# Hardcode two 2x2 matrix as a list (mutable)
A = [
    [5.0,3.0],
    [1.0,3.0]
]

B = [
    [4.0,2.0],
    [6.0,1.0]
]

# Initialise C (the product)
C = [
    [0.0,0.0],
    [0.0,0.0]
]

# Multiply A and B
# Looping through the rows of C
for i in range(len(C)):
    # Looping through the columns of C
    for j in range(len(C)):
        # Looping through the rows of A and columns of B
        for k in range(len(A)):
            C[i][j] += A[i][k] * B[k][j]

print("The product of A and B is C: ",C)

# Inverting C

# Calculating the Determinant of a 2x2 matrix
detC = C[0][0]*C[1][1] - C[0][1]*C[1][0]
print("The determinant of C is: ",detC)

# Initialising the inverse of C
invC = [
    [0.0, 0.0],
    [0.0, 0.0]
]

# Looping through the rows of invC
for x in range(len(invC)):
    # Looping through the columns of invC
    for y in range(len(invC)):
        if x == y: 
            invC[x][y] = C[(1-x)][(1-y)] / detC
        else: 
            invC[x][y] = - C[x][y] / detC

print("The inverse of C is: ",invC)
