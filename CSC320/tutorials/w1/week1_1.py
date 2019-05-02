import numpy as np

# Create a 2x2 matrix
mat_1 = np.array([[1, 2], [3, 4]])
print(mat_1)
# Get the dimension of mat_1
print(mat_1.shape)
# Get the transpose of mat_1
print(mat_1.T)
# Unlike python list, we need a more specific value type for the data
print(mat_1.dtype)
mat_another_type = mat_1.astype(np.uint8)
print(mat_another_type.dtype)

# Create a 3x3 identity matrix
mat_2 = np.eye(3, 3)
# Create a non-square identity matrix (3x5)
mat_3 = np.eye(3, 5)
# Create a 4x3 matrix with all zero entries
mat_4 = np.zeros((4, 3))
# Create a 2x5 matrix with all one entries
mat_5 = np.ones((2, 5))
# Create a 3x1 random matrix (every entry is between 0 and 1)
mat_6 = np.random.random((3, 1))

# matrix concatenation
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
print(np.concatenate((a, b), axis=0))
print(np.concatenate((a, b.T), axis=1))

# Do matrix slicing on mat_1, to get the lower left element from mat_1
print(mat_1[1, 0])
# Do matrix slicing on mat_1, to get the second row from mat_1
print(mat_1[1, :])

# Loop over matrices
# Loop over matrices is slow, should consider using matrix operations, e.g. sum between matrices, dot product
# Loop over rows
for row in mat_1:
    print(row)
# Loop over every entry:
for row in mat_1:
    for entry in row:
        print(entry)
# Add one to every entry:
r, c = mat_1.shape
for row in range(r):
    for col in range(c):
        mat_1[row, col] += 1

# In the last example, an alternative way (and is a better way) is:
mat_1 = mat_1 + 1
# or equivalently:
mat_1 += 1
# Adding a number with a matrix will do addition on every entry of the matrix,
# and the same for subtraction, multiplication, division, etc.

# Operation between two matrices is also point-wise operation.
# Note: the two matrices must have same shape
print(mat_1 * mat_1)
# Thus the "multiplication" between matrices is not what we define mathematically
# To do the linear algebra style multiplication, we need to use function np.matmul
print(np.matmul(mat_1, mat_1))
# And we can use np.dot for dot product
print(np.dot(mat_1, mat_1))
# Furthermore, we can do cross product
print(np.cross(np.array([1, 2, 3]), np.array([4, 5, 6])))

# Filtering
# Get a boolean matrix showing if the entries can satisfy some condition
print(mat_1 > 2)
# Filter out values that doesn't satisfy the condition
mat_1[mat_1 > 2] 
