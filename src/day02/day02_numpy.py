import numpy as np


ratings_sample = np.array([4.5,3.0,5.0,2.5,4.0])

# vectorized operations

print("Ratings sample:", ratings_sample*2)
print("Mean:", ratings_sample.mean())
print("Standard Deviation:", ratings_sample.std())
print("Maximum:", ratings_sample.max())

high_ratings = ratings_sample[ratings_sample >= 4.0]
print("High ratings:", high_ratings)

matrix_a = np.array([[1,2,3],[4,5,6]])
matrix_b = np.array([[7,8,9],[10,11,12]])
print("Matrix A shape:", matrix_a.shape)
print("Column sums of Matrix A:", matrix_a.sum(axis=0))
print("Row sums of Matrix A:", matrix_a.sum(axis=1))

print("Matrix B shape:", matrix_b.shape)
print("Column sums of Matrix B:", matrix_b.sum(axis=0))
print("Row sums of Matrix B:", matrix_b.sum(axis=1))

print("Matrix A:")
print(matrix_a)
print("Matrix B:")
print(matrix_b)
print("Matrix A + Matrix B:")
print(matrix_a + matrix_b)  


####
# broadcasting
prices = np.array([10, 20, 30])
discounted = prices - (prices * 0.1)
print(f"Discounted prices: {discounted}")