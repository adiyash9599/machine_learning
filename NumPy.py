import numpy as np

# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# b = np.array([[1, 2], [3, 4]])
# print(a)
# print(b)

# c= np.zeros
# d = np.ones
# print(c)
# print(d)
# #get dimension
# print(a.ndim)
# print(b.ndim)

# #get shape
# print(a.shape)  
# print(b.shape)

# #get type 
# print(a.dtype)
# print(b.dtype)

# # we can also specify the type of array
# e = np.array([1, 2, 3, 4, 5], dtype='int16')
# f = np.array([1, 2, 3, 4, 5], dtype='int64')
# print(e.dtype)

# #get size
# print(e.itemsize)
# print(f.itemsize)

# #get total size
# print(e.nbytes) # or print(e.size * e.itemsize)
# print(e.size)

a = np.array([[1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20]])
# print(a.shape)
# get a element [r, c]
# print(a[1, -5])

# get a row
# print(a[0, :])

# get a column  
# print(a[:, 2])

# [startIdx, endIdx, step]
# print(a[0, 0:-2:2])

a[1, 5] = 20
# print(a)

#change column
a[:, 9] = 30
# print(a)

#change column in sequence
a[:, 8] = [1, 2]
# print(a)

b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print(b)

#get specific element (work outside in)
# print(b[0, 1, 1]) #b[0, 1, 1]
b[:, 1, :] = [[9, 10], [11, 12]]#b[0, 1, 1]]

# print(b)

c = np.zeros((2, 3, 4))
print(c)
d = np.ones((2, 3, 4))
print(d)

e = np.full((2, 2), 99)
print(e)

# any other number
f = np.full_like(a, 4)# a.shape
print(f)

# random decimal numbers
g = np.random.rand(4, 2)    
print(g)


# random integer values
h = np.random.randint(1, 100, size=(3, 3))
print(h)

i = np.random.random_sample(a.shape)
print(i)

j = np.identity(5)
print(j)

k = np.eye(4, 4, k=1)
print(k)

l = np.eye(4, 4, k=-1)
print(l)

m = np.array([[1, 2, 3]])
r1 = np.repeat(m, 3, axis = 0)
print(r1)

# n = np.array([1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 9, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1])
# print(n)

n = np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 9, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]])
print(n)

output = np.ones((5, 5))
print(output)

z = np.zeros((3, 3))
z[1, 1] = 9
print(z)

output[1: -1, 1: -1] = z
print(output)

#be careful when copying arrays
a = np.array([1, 2, 3])
b = a
b[0] = 100

print(b)
print(a)
# both values will change

c = a.copy()
c[0] = 200
print(c)
print(a)

# mathematics
x = np.array([1, 2, 3, 4])
print(x)
# x += 2
# print(x)
# x -= 1
# print(x)
# x *= 2
# print(x)
# x /= 2
# print(x)

b = np.array([1, 0, 1, 0])
x += b
print(x)

x **= 2
print(x)

y = np.array([1, 2, 3, 4])
print(np.sin(y))
print(np.cos(y))
print(np.tan(y))

# https://docs.scipy.org/doc/numpy/reference/routines.math.html

# Linear Algebra
t = np.ones((2, 3))
print(t)

l = np.full((3, 2), 2)
print(l)

print(np.matmul(t, l))

# find the determinant
c = np.identity(3)
print(c)
print(np.linalg.det(c))

# reference docs https://docs.scipy.org/doc/numpy/reference/routines.linalg.html

# stastics
arr = np.array([[1, 6, 7], [4, 2, 3]])
print(np.min(arr)) # min value
print(np.min(arr, axis = 0)) # min value -> got the value of min of 1st row and min of second row if axis = 1 and if axis = 0 then we got min of each column
print(np.max(arr)) # max value
print(np.sum(arr)) # sum of all values
print(np.sum(arr, axis = 1)) # sum of each row
print(np.sum(arr, axis = 0)) # sum of each column
print(np.mean(arr)) # mean of all values
print(np.std(arr)) # standard deviation

arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.min(arr2, axis = 1)) # min value of each row
print(np.min(arr2, axis = 0)) # min value of each column

before = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(before)
after = before.reshape((2, 2, 2))
print(after)

# vertically stacking vectors
v1 = np.array([1, 2, 3, 4])
v2 = np.array([5, 6, 7, 8])
print(np.vstack((v1, v2))) # vertically stacking vectors

# Horizontally stacking vectors
h1 = np.ones((2, 4))
h2 = np.zeros((2, 2))
print(np.hstack((h1, h2)))

fildata = np.genfromtxt('data.txt', delimiter = ',')
filedata = fildata.astype('int32')
print(fildata)

# advanced indexing and boolean masking
print(filedata > 50)
print(filedata[filedata > 50])

#you can index with a list in NumPy
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])# 1D array
print(a[[1, 3, 5]]) # indexing with a list

print(np.any(filedata > 50, axis = 0)) # axis = 0 means column wise
print(np.all(filedata > 50, axis = 0)) # axis = 0 means column wise

print(((filedata > 50) & (filedata < 100)))
print(~((filedata > 50) & (filedata < 100)))