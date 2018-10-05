import numpy as np
import numpy.random as rnd
import time
from numpy import linalg as la
import pickle
import math
import matplotlib.pyplot as plt
#Question 1

"""
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



#Question 2
print("-----------------------------------------------------------------------")
print("Question 2 a)")


def cube_helper(A, B):

    answer = np.zeros(A.shape)

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A)):
                answer[i][j] += A[i][k] * B[k][j]
    return answer


def cube(A):

    return cube_helper(cube_helper(A,A),A)

print("-----------------------------------------------------------------------")
print("Question 2 b)")

def mymeasure(N):


    A = rnd.rand(N,N)
    start_for_cube1 = time.time()
    Cube1 = np.dot(np.dot(A,A), A)
    print(str(time.time()-start_for_cube1)+' seconds for Cube1 using numpy')

    start_for_cube2 = time.time()
    Cube2 = cube(A)
    print(str(time.time()-start_for_cube2)+' seconds for Cube2 using loops')

    print('Magnitude of the difference is'+str(la.norm(np.subtract(Cube1, Cube2), np.inf)))


print("-----------------------------------------------------------------------")
print("Question 2 c) N=200")
mymeasure(200)
print("-----------------------------------------------------------------------")
print("Question 2 c) N=2000")
mymeasure(2000)
print("-----------------------------------------------------------------------")
"""


#Question 4

with open('data1.pickle','rb') as f:
    dataTrain, dataTest = pickle.load(f)


#Question 4a)

def K_n(X, S, sigma):

            return math.exp((math.pow((X - S), 2)/-(2*sigma)))

def kernelMatrix(X, S, sigma):

    new_X = np.repeat(X[None], S.shape[0], axis=0).T

    form_K = np.vectorize(K_n)

    ones = np.ones(X.shape[0])

    K_without_ones = form_K(new_X, S, sigma)


    return np.insert(K_without_ones, 0, ones, axis=1)

# Question 4b)

def plotBasis(S, sigma):


    x = np.linspace(-1, 1, 1000)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.suptitle('Question 4(b): some basis functions with sigma = 0.2')
    plt.plot(x, kernelMatrix(x, S, sigma))
    plt.show()

print(dataTrain[:, 0])
plotBasis(dataTrain[:, 0][:3], sigma=0.2)

def check_plot(X, S,sigma):

    pass

