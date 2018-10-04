import numpy as np
import numpy.random as rnd

print ("\n\nQuestion 1")
print ("-----------------------------------------------------")

print ("\nQuestion 1(a): ")
A = rnd.rand(4, 3)
print(A)
print ("-----------------------------------------------------")

print ("\nQuestion 1(b): ")
x = rnd.rand(4, 1)
print(x)
print ("-----------------------------------------------------")

print ("\nQuestion 1(c): ")
B = A.reshape((2, 6))
print(B)
print ("-----------------------------------------------------")

print ("\nQuestion 1(d): ")
C = np.add(x, A)
print(C)
print ("-----------------------------------------------------")

print ("\nQuestion 1(e): ")
y = x.reshape(4)
print(y)
print ("-----------------------------------------------------")

print ("\nQuestion 1(f): ")
A[:, 0] = y
print(A)
print ("-----------------------------------------------------")

print ("\nQuestion 1(g): ")
A[:, 0] = np.add(A[:, 2], y)
print(A)
print ("-----------------------------------------------------")

print ("\nQuestion 1(h): ")
print(A[:, :2])
print ("-----------------------------------------------------")

print ("\nQuestion 1(i): ")
print(A[1, :])
print(A[3, :])
print ("-----------------------------------------------------")

print ("\nQuestion 1(j): ")
print(np.sum(A, axis=0))
print ("-----------------------------------------------------")

print ("\nQuestion 1(k): ")
print(np.max(A, axis=1))
print ("-----------------------------------------------------")

print ("\nQuestion 1(l): ")
print(np.mean(A))
print ("-----------------------------------------------------")

print ("\nQuestion 1(m): ")
print(np.log(np.square(A)))
print ("-----------------------------------------------------")

print ("\nQuestion 1(n): ")
print(np.dot(np.transpose(A), x))
print ("-----------------------------------------------------")
