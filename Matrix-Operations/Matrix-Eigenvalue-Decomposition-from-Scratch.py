# Matrix operations from scratch (without numpy)

# Hardcode 2x2 matrix as a list (mutable)
A = [
    [5.0,3.0],
    [1.0,3.0]
]

# Applying a*x^2 + b*x + c = 0 using Determinant (A - lambda * Identity) = 0
a = 1.0
b = -A[0][0] - A[1][1]
c = A[0][0]*A[1][1] -A[0][1]*A[1][0]

# Calculating eigenvalues from quadratic equation
lambda1 = (-b-((b**2 - 4*a*c)**0.5)) / 2*a 
lambda2 = (-b+((b**2 - 4*a*c)**0.5)) / 2*a
print("The eigenvalues are: ",lambda1," and ",lambda2)

# Calculating eigenvectors v1, v2 given (A - lambda * identity)*V = 0
v1 = [
    [0],
    [0]
]

v2 = [
    [0],
    [0]
]

# (A[0][0] - lambda1)*v1[0] + A[0][1]*v1[1] = 0
# A[1][0]*v1[0] + (A[1][1]-lambda1)*v1[1] = 0
# v1[0] = (-A[0][1] / (A[0][0] - lambda1)) * v1[1]
# v1[1] = (- A[1][0] / (A[1][1]-lambda1)) * v1[0]
# v1[0] = (-A[0][1] / (A[0][0] - lambda1)) * (- A[1][0] / (A[1][1]-lambda1)) * v1[0]
v1[0][0] = 1
v1[1][0] = (- A[1][0] / (A[1][1]-lambda1)) * v1[0][0]

v2[0][0] = 1
v2[1][0] = (- A[1][0] / (A[1][1]-lambda2)) * v2[0][0]

# Initialised normalised eigenvectors v1norm, v2norm
v1norm = [
    [0],
    [0]
]

v2norm = [
    [0],
    [0]
]

# Normalised v1, v2
v1norm[0][0] = v1[0][0] / ((v1[0][0]**2 + v1[1][0]**2)**0.5)
v1norm[1][0] = v1[1][0] / ((v1[0][0]**2 + v1[1][0]**2)**0.5)

v2norm[0][0] = v2[0][0] / ((v2[0][0]**2 + v2[1][0]**2)**0.5)
v2norm[1][0] = v2[1][0] / ((v2[0][0]**2 + v2[1][0]**2)**0.5)

print("The normalised eigenvectors are: ",v1norm," and ",v2norm)

# Multiply A by v1 to make X1, and by v2 to make X2
X1 = [
    [0],
    [0]
]

X2 = [
    [0],
    [0]
]

# Iterating through the rows of A for calculating X1, X2
for i in range(len(A)):
    # Iterating through the rows of v1norm
    for j in range(len(v1norm)):
        X1[i][0] += A[i][j] * v1norm[j][0]

for z in range(len(A)):
    # Iterating through the rows of v2norm
    for y in range(len(v2)):
        X2[z][0] += A[z][y] * v2norm[y][0]

# Confirming lambda1, lambda2
result1 = 0
for k in range(len(X1)):
    result1 = result1 + abs(X1[k][0] - (lambda1 * v1norm[k][0]))
if int(result1 * 10**5) == 0: 
    print("Lambda1 is verified")
else:
    print("Lambda1 is incorrect")

result2 = 0
for l in range(len(X2)):
    result2 = result2 + abs(X2[l][0] - (lambda2 * v2norm[l][0]))
if int(result2 * 10**5) == 0: 
    print("Lambda2 is verified")
else:
    print("Lambda2 is incorrect")
