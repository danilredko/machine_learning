import numpy as np
import numpy.random as rnd
import time
from numpy import linalg as la
import pickle
import math
import matplotlib.pyplot as plt
import numpy.linalg as lin
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

            return math.exp((math.pow((X - S), 2)/-(2*sigma*sigma)))

def kernelMatrix(X, S, sigma):

    new_X = np.repeat(X[None], S.shape[0], axis=0).T

    form_K = np.vectorize(K_n)

    ones = np.ones(X.shape[0])

    K_without_ones = form_K(new_X, S, sigma)

    finalK = np.insert(K_without_ones, 0, ones, axis=1)

    return finalK


# Question 4b)

def plotBasis(S, sigma):

    x = np.linspace(0, 1, 1000)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.suptitle('Question 4(b): some basis functions with sigma = 0.2')
    K = kernelMatrix(x, S, sigma)
    plt.plot(x, K)
    plt.show()

plotBasis(dataTrain[:, 0][:5], sigma=0.2)

# Question 4 c)



def myfit(S,sigma):

    t_n = dataTrain[:, 1][:S.shape[0]]

    X = dataTrain[:, 0][:S.shape[0]]

    K = kernelMatrix(X , S, 0.2)

    w = lin.lstsq(K, t_n)[0]

    Y = K.dot(w)

    err_train = np.sum(np.divide(np.power(np.subtract(t_n, Y ),2), dataTrain.shape[0]))

    err_test = np.sum(np.divide(np.power(np.subtract(t_n, Y ),2), dataTest.shape[0]))

    return w, err_train, err_test

# Question 4 d)


def plotY(w,S,sigma):

    t_n = dataTrain[:, 1][:S.shape[0]]
    x = np.linspace(0,1,1000)
    plt.xlabel('x')
    plt.ylabel('t')
    K = kernelMatrix(x, S, sigma)
    Y = K.dot(w)
    plt.plot(x, Y, color='red')
    plt.scatter(S, t_n)
    plt.ylim(-15, 15)



# Question 4 e)
'''
plt.suptitle('Question 4 d)')
plotY(myfit(dataTrain[:, 0][:5], 0.2), dataTrain[:, 0][:5], 0.2)
plt.show()
'''

# Question 4 f)

def bestM(sigma):

    f = plt.figure()

    for M in range(1,17):
        plt.subplot(4, 4, M)
        S = dataTrain[:, 0]
        plt.title('M = {}'.format(M-1))
        plt.subplots_adjust(wspace=0.5, hspace =1)

        w, err_train, err_test = myfit(S[:M], 0.2)

        plotY(w, S[:M], sigma)





    plt.suptitle('Question 4 (f) Best-Fitting functions with 0-15 basis functions')
    plt.show()





bestM(0.2)



'''
S = dataTrain[:, 0]
t_n = dataTrain[:, 1]

print(kernelMatrix(S, t_n, 0.2) )
print("___________________________________________")
print()

plotY(myfit(S , 0.2) , S, sigma=0.2)


plt.show()

'''


