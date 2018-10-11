import numpy as np
import numpy.random as rnd
import time
from numpy import linalg as la
import pickle
import math
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import sys
import sklearn.linear_model as lin
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

    if S.shape[0] == 0:
        return np.ones([X.shape[0], 1])

    if X.shape[0] == 0:
        return np.array([])

    new_X = np.repeat(X[None], S.shape[0], axis=0).T

    form_K = np.vectorize(K_n)

    ones = np.ones(X.shape[0])

    K_without_ones = form_K(new_X, S , sigma)

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

#plotBasis(dataTrain[:, 0][:5], sigma=0.2)

# Question 4 c)


def myfit(S, sigma):


    #work for bestM
    #t_n = dataTrain[:, 1][:S.shape[0]]

    #works for errors


    #work for bestM
    #X = dataTrain[:, 0][:S.shape[0]]

    #works for errors

    # Training data

    tTrain = dataTrain[:, 1]
    tTrain = tTrain.reshape(tTrain.shape[0], 1)

    #print('tTrain: '+str(tTrain.shape))

    #print('S shape: '+str(S.shape))

    Xtrain = dataTrain[:, 0]

    K_train = kernelMatrix(Xtrain, S, sigma)

    #print("K shape: "+str(K_train.shape))

    w = linalg.lstsq(K_train, tTrain, rcond=None)[0]

    #print("W shape: "+str(w.shape))

    #print(w)

    Ytrain = K_train.dot(w)

    err_train = np.divide(np.power(np.subtract(tTrain, Ytrain), 2), tTrain.shape[0])

    # Testing data

    tTest = dataTest[:, 1]
    tTest = tTest.reshape(tTest.shape[0], 1)

    Xtest = dataTest[:, 0]

    K_test = kernelMatrix(Xtest, S, sigma)

    Y_test = K_test.dot(w)

    err_test = np.divide(np.power(np.subtract(tTest, Y_test), 2), dataTest.shape[0])

    return w, np.sum(err_train), np.sum(err_test)

# Question 4 d)


def plotY(w, S, sigma):

    x = np.linspace(0, 1, 1000)
    plt.xlabel('x')
    plt.ylabel('t')
    K = kernelMatrix(x, S, sigma)
    #print(K)
    Y = K.dot(w)
    plt.plot(x, Y, color='red')
    plt.scatter(dataTrain[:, 0], dataTrain[:, 1])
    plt.ylim(-15, 15)
    plt.xlim(0, 1)



# Question 4 e)
'''
plt.suptitle('Question 4 d)')
plotY(myfit(dataTrain[:, 0][:5], 0.2), dataTrain[:, 0][:5], 0.2)
plt.show()
'''

# Question 4 f)

def bestM(sigma):

    for M in range(0, 16):
        plt.subplot(4, 4, M+1)
        S = dataTrain[:, 0]
        plt.title('M = {}'.format(M))
        plt.subplots_adjust(wspace=0.5, hspace =1)
        w, err_train, err_test = myfit(S[:M], 0.2)
        plotY(w, S[:M], sigma)

    plt.suptitle('Question 4 (f) Best-Fitting functions with 0-15 basis functions')
    plt.show()



    errTrain = np.array([])
    errTest = np.array([])
    M = range(16)
    for m in M:

        w, err_train, err_test = myfit(dataTrain[:m, 0], 0.2)

        errTrain = np.append(errTrain, err_train)

        errTest = np.append(errTest, err_test)

    print("Train Errors : "+str(errTrain))
    print("")
    print("Test Errors : "+str(errTest))
    plt.plot(M, errTrain, 'blue', label='Train Error')
    plt.plot(M, errTest, 'red', label='Test Error')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.suptitle("Question 4(f): training and test error")
    plt.ylim(0, 250)
    plt.show()


bestM(0.2)

#print(myfit(dataTrain[0:5, 0], 0.2))
#print(dataTrain[0:0, 0].shape)
#print(kernelMatrix(dataTrain[0:0, 0], dataTrain[0:3, 0], 0.2))

# Question 5

with open('data2.pickle','rb') as f:
    dataVal, dataTest = pickle.load(f)


def regFit(S, sigma, alpha):

    t = dataTest[:, 1]

    K = kernelMatrix(t, S, sigma)

    ridge = lin.Ridge(alpha)

    ridge.fit(K, t)

    w = ridge.coef_

    w[0] = ridge.intercept_