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
#mymeasure(200)
print("-----------------------------------------------------------------------")
print("Question 2 c) N=2000")
#mymeasure(2000)
print("-----------------------------------------------------------------------")


# Question 4

print("-----------------------------------------------------------------------")
print("Question 4")

with open('data1.pickle','rb') as f:
    dataTrain, dataTest = pickle.load(f)

# Question 4a)

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


    # Training data

    tTrain = dataTrain[:, 1]
    tTrain = tTrain.reshape(tTrain.shape[0], 1)

    Xtrain = dataTrain[:, 0]

    K_train = kernelMatrix(Xtrain, S, sigma)

    w = linalg.lstsq(K_train, tTrain, rcond=None)[0]

    Ytrain = K_train.dot(w)

    train_error = np.divide(np.power(np.subtract(tTrain, Ytrain), 2), tTrain.shape[0])

    # Testing data

    tTest = dataTest[:, 1]
    tTest = tTest.reshape(tTest.shape[0], 1)

    Xtest = dataTest[:, 0]

    K_test = kernelMatrix(Xtest, S, sigma)

    Y_test = K_test.dot(w)

    test_error = np.divide(np.power(np.subtract(tTest, Y_test), 2), dataTest.shape[0])

    return w, np.sum(train_error), np.sum(test_error)

# Question 4 d)

def plotY(w, S, sigma):

    x = np.linspace(0, 1, 1000)
    plt.xlabel('x')
    plt.ylabel('t')
    K = kernelMatrix(x, S, sigma)
    Y = K.dot(w)
    plt.plot(x, Y, color='red')
    plt.scatter(dataTrain[:, 0], dataTrain[:, 1])
    plt.ylim(-15, 15)
    plt.xlim(0, 1)


# Question 4 e)

def q4e():

    plt.suptitle('Question 4 (e): the fitted function (5 basis functions) ')
    w, train_error, test_error = myfit(dataTrain[:, 0][:5], 0.2)
    plotY(w, dataTrain[:, 0][:5], 0.2)
    plt.show()

# Question 4 f)

def bestM(sigma):

    for M in range(0, 16):
        plt.subplot(4, 4, M+1)
        S = dataTrain[:, 0]
        plt.title('M = {}'.format(M))
        plt.subplots_adjust(wspace=0.5, hspace =1)
        w, train_error, test_error = myfit(S[:M], sigma)
        plotY(w, S[:M], sigma)

    plt.suptitle('Question 4 (f) Best-Fitting functions with 0-15 basis functions')
    plt.show()

    TrainErrors = np.array([])
    TestErrors = np.array([])
    xAxis = range(16)

    for m in xAxis:

        w, train_error, test_error = myfit(dataTrain[:m, 0], sigma)

        TrainErrors = np.append(TrainErrors, train_error)

        TestErrors = np.append(TestErrors, test_error)

    plt.plot(xAxis, TrainErrors, 'blue', label='Train Error')
    plt.plot(xAxis, TestErrors, 'red', label='Test Error')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.suptitle("Question 4(f): training and test error")
    plt.ylim(0, 250)
    plt.show()

    # Best fitting function
    M = np.argmin(TestErrors)
    S = dataTrain[:, 0][:M]
    w, train_error, test_error = myfit(S, sigma)
    plotY(w, S, sigma)
    plt.suptitle("Question 4(f): best-fitting function ({} basis functions)".format(M))
    print("Best Fitting Function: ")
    print("Optimal value of M = {}".format(M))
    print("_______________________________________________________")
    print("Optimal value of w = {}".format(w))
    print("_______________________________________________________")
    print("Training Error of Optimal Value of M and w :{} ".format(train_error))
    print("_______________________________________________________")
    print("Testing Error of Optimal Value of M and w :{} ".format(test_error))
    print("_______________________________________________________")
    if train_error < test_error:
        print("Training error is indeed less than test error.  ")
    print("_______________________________________________________")
    plt.show()

print("-----------------------------------------------------------------------")
#bestM(0.2)


# Question 5


print("Question 5")

with open('data2.pickle','rb') as f:
    dataVal, dataTest = pickle.load(f)


def regFit(S, sigma, alpha):

    tTrain = dataTrain[:, 1]
    xTrain = dataTrain[:, 0]

    K_train = kernelMatrix(xTrain, S, sigma)

    ridge = lin.Ridge(alpha)

    ridge.fit(K_train, tTrain)

    w = ridge.coef_

    w[0] = ridge.intercept_

    Ytrain = K_train.dot(w)

    train_error = np.divide(np.power(np.subtract(tTrain, Ytrain), 2), tTrain.shape[0])

    tVal = dataVal[:, 1]

    xVal = dataVal[:, 0]

    K_val = kernelMatrix(xVal, S, sigma)

    Yval = K_val.dot(w)

    validation_error = np.divide(np.power(np.subtract(tVal, Yval), 2), dataVal.shape[0])

    return w, np.sum(train_error), np.sum(validation_error)

def Q5b():

    w, train_error, validation_error = regFit(dataTrain[:, 0], 0.2, 1)
    plotY(w,dataTrain[:, 0], 0.2)
    plt.suptitle("Question 5(b): the fitted function (alpha=1)")
    plt.show()

def bestAlpha(S, sigma):

    alphas = np.power(10.0, np.arange(-12, 4, 1, dtype=int))

    TrainErrors = np.array([])
    ValidationErrors = np.array([])

    for M in range(0, 16):
        plt.subplot(4, 4, M+1)
        S = dataTrain[:, 0]
        plt.title('alpha = {}'.format(alphas[M]))
        plt.subplots_adjust(wspace=0.5, hspace =1)
        w, train_error, validation_error = regFit(S, sigma, alphas[M])

        TrainErrors = np.append(TrainErrors, train_error)

        ValidationErrors = np.append(ValidationErrors, validation_error)


        plotY(w, S, sigma)

    plt.suptitle('Question 5 (c) Best-Fitting functions for log(alpha)= -12, -11,...,0,1,2,3')
    plt.show()

    plt.semilogx(alphas, TrainErrors, 'blue', label='Train Error')
    plt.semilogx(alphas, ValidationErrors, 'red', label='Validation Error')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel("error")
    plt.xlabel("alpha")
    plt.suptitle("Question 5(c): training and validation error")
    plt.show()

    # Best fitting function
    i = np.argmin(ValidationErrors)
    S = dataTrain[:, 0]
    w, train_error, validation_error = regFit(S, sigma, alphas[i])
    plotY(w, S, sigma)
    plt.suptitle("Question 5(c): best-fitting function (alpha = {})".format(alphas[i]))
    plt.show()
    print("Best Fitting Function: ")
    print("Optimal value of alpha = {}".format(alphas[i]))
    print("_______________________________________________________")
    print("Optimal value of w = {}".format(w))
    print("_______________________________________________________")
    print("Training Error of Optimal Value of alpha and w :{} ".format(train_error))
    print("_______________________________________________________")
    print("Validation Error of Optimal Value of alpha and w :{} ".format(validation_error))
    print("_______________________________________________________")

    # Getting Test Error
    tTest = dataTest[:, 1]
    Xtest = dataTest[:, 0]
    K_test = kernelMatrix(Xtest, S, sigma)
    Y_test = K_test.dot(w)
    test_error = np.divide(np.power(np.subtract(tTest, Y_test), 2), dataTest.shape[0])
    test_error = np.sum(test_error)

    print("Testing Error of Optimal Value of alpha and w : {} ".format(test_error))
    print("_______________________________________________________")

    print("_______________________________________________________")
    if train_error < validation_error < test_error:
        print("The errors are indeed training error < validation error < test error ")
    print("_______________________________________________________")
    plt.show()

#Q5b()
#bestAlpha(dataTrain[:, 0], 0.2)


# Question 6
print("-----------------------------------------------------------------------")
print("Question 6")

# Question 6 a)

def q6a():

    np.random.shuffle(dataVal)
    plt.scatter(dataVal[:, 0], dataVal[:, 1])
    plt.suptitle("Question 6(a): Training data for Question 6")
    plt.xlabel(" x ")
    plt.ylabel(" y ")
    plt.show()




#q6a()


def cross_val(K, S, sigma, alpha, X, Y):


    X_k_fold = np.split(X, K)
    Y_k_fold = np.array_split(Y, K)

    trainErrors = np.array([])
    valErrors = np.array([])

    for i in range(0, K):

        tVal = Y_k_fold[i]

        xVal = X_k_fold[i]

        tTrain = np.delete(Y_k_fold, i, axis=0)

        tTrain = tTrain.reshape(tTrain.size)

        xTrain = np.delete(X_k_fold, i, axis=0)

        xTrain = xTrain.reshape(xTrain.size)

        K_train = kernelMatrix(xTrain, S, sigma)

        ridge = lin.Ridge(alpha)

        ridge.fit(K_train, tTrain)

        w = ridge.coef_

        w[0] = ridge.intercept_

        Ytrain = K_train.dot(w)

        train_error = np.divide(np.power(np.subtract(tTrain, Ytrain), 2), tTrain.shape[0])

        trainErrors = np.append(trainErrors, np.sum(train_error))

        K_val = kernelMatrix(xVal, S, sigma)

        Yval = K_val.dot(w)

        validation_error = np.divide(np.power(np.subtract(tVal, Yval), 2), xVal.shape[0])

        valErrors = np.append(valErrors, np.sum(validation_error))

    return trainErrors, valErrors


def q6c():

    trainErros, valErrors = cross_val(5, dataTrain[:, 0][:10], 0.2, 1.0, dataVal[:, 0], dataVal[:, 1])

    x = np.linspace(1, 5, 5)

    plt.plot(x, trainErros, 'b', label='Training Errors')
    plt.plot(x, valErrors, 'r', label='Validation Errors')
    plt.xlabel('fold')
    plt.ylabel('error')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.suptitle("Question 6(c): training and validation errors during cross validation")
    plt.show()
    print("Mean of training Errors: {}".format(np.mean(trainErros)))
    print("Mean of Validation Errors: {}".format(np.mean(valErrors)))
    if np.mean(valErrors) > np.mean(trainErros):
        print("Mean Validation Error is indeed greater than the mean training error")

#q6c()

def bestAlphaCV(K, S, sigma, X , Y):

    alphas = np.power(10.0, np.arange(-11, 5, 1, dtype=int))

    TrainErrors = np.array([])
    valErrors = np.array([])

    for i in range(0, 16):

        trainErr, ValErr = cross_val(K, S, sigma, alphas[i], X, Y)
        TrainErrors = np.append(TrainErrors, np.mean(trainErr))
        valErrors = np.append(valErrors, np.mean(ValErr))

    plt.semilogx(alphas, TrainErrors, 'b', label='Training Error')
    plt.semilogx(alphas, valErrors, 'r', label='Validation Error')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.suptitle("Question 6(d): training and validation error")
    plt.xlabel("alpha")
    plt.ylabel("error")
    plt.show()

    index_of_best_alpha = np.argmin(valErrors)

    best_aplha = alphas[index_of_best_alpha]

    print("Optimal value of alpha: {}".format(best_aplha))

    tTrain = Y
    xTrain = X

    K_train = kernelMatrix(xTrain, S, sigma)

    ridge = lin.Ridge(best_aplha)

    ridge.fit(K_train, tTrain)

    w = ridge.coef_

    Ytrain = K_train.dot(w)

    train_error = np.sum(np.divide(np.power(np.subtract(tTrain, Ytrain), 2), tTrain.shape[0]))

    tTest = dataTest[:, 1]
    Xtest = dataTest[:, 0]
    K_test = kernelMatrix(Xtest, S, sigma)

    Y_test = K_test.dot(w)

    test_error = np.sum(np.divide(np.power(np.subtract(tTest, Y_test), 2), dataTest.shape[0]))

    print("Optimal value of w: {}".format(w))

    print("Testing Error: {}".format(test_error))

    print("Training Error: {}".format(train_error))

    print("Mean Validation Error: {}".format(np.mean(valErrors)))


#bestAlphaCV(5, dataTrain[:, 0][:15], 0.2, dataVal[:, 0], dataVal[:, 1])


# Question 7

print("-----------------------------------------------------------------------")
print("Question 7")

def der_of_loss(w, S, sigma):

    X = dataTrain[:, 0]

    t = dataTrain[:, 1]

    K = kernelMatrix(X, S, sigma)

    Y = K.dot(w)

    K_w_t = np.subtract(Y, t)

    Two_K_ = np.add(np.transpose(K), np.transpose(K))

    train_error = np.sum(np.divide(np.power(np.subtract(t, Y), 2), t.shape[0]))

    tTest = dataTest[:, 1]
    Xtest = dataTest[:, 0]

    K_test = kernelMatrix(Xtest, S, sigma)

    Y_test = K_test.dot(w)
    test_error = np.sum(np.divide(np.power(np.subtract(tTest, Y_test), 2), dataTest.shape[0]))

    return (np.dot(Two_K_, K_w_t), train_error, test_error)


def fitRegGD(S, sigma, alpha, lrate):

    w = np.random.randn(16)

    fours = np.power(4, range(0, 9))
    counter = 1

    TrainErrors = np.array([])
    TestErrors = np.array([])

    for i in range(0, 100000):

        w_1 = w[1:]

        w_0 = w[0]

        two_alpha_w_1 = np.add(np.dot(alpha, w_1), np.dot(alpha, w_1))

        term_1, train_error, test_error = der_of_loss(w, S, sigma)

        TrainErrors = np.append(TrainErrors, train_error)
        TestErrors = np.append(TestErrors, test_error)

        w_1 = np.subtract(w_1, np.dot(lrate, np.add(term_1[1:], two_alpha_w_1)))

        term_0, train_error, test_error = der_of_loss(w, S, sigma)

        w_0 = np.subtract(w_0, np.dot(lrate, term_0[0]))

        w = np.insert(w_1, 0, w_0)

        if i in fours:
            plt.subplot(3, 3, counter)
            plt.title("$4^{}$ iterations".format(counter-1))
            counter += 1
            plt.subplots_adjust(wspace=0.5, hspace=1)
            plotY(w, S, sigma)

    plt.suptitle("Question 7: fitted function as iterations increase")
    plt.show()

    # Q7 fitted function
    plotY(w, S, sigma)
    plt.suptitle("Question 7: fitted function")
    plt.show()

    # Plot of Testing and Training Errors

    x = np.arange(100000)

    plt.plot(x, TrainErrors, 'b', label='Training Error')
    plt.plot(x, TestErrors, 'r', label='Testing Error')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('error')
    plt.xlabel('iterations')
    plt.suptitle("Question 7: training and test error v.s. iterations")
    plt.show()

    # Test and Train Error log Scale

    plt.semilogx(x, TrainErrors, 'b', label='Training Error')
    plt.semilogx(x, TestErrors, 'r', label='Testing Error')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('error')
    plt.xlabel('iterations')
    plt.suptitle("Question 7: training and test error v.s. iterations (log scale)")
    plt.show()

    # last 10000 training errors

    x = np.arange(10000)

    plt.plot(x, TrainErrors[-10000:], 'b')
    plt.ylabel('error')
    plt.xlabel('x')
    plt.suptitle("Question 7: last 10000 training errors")
    plt.show()


    print('')
    print("Optimal w:{} ".format(w))
    print("")
    print("Training error: "+str(TrainErrors[-1]))
    print("")
    print("Test Error: "+str(TestErrors[-1]))
    w2, test_error, val_error = regFit(S, sigma, alpha)
    print('')
    print("w2: {}".format(w2))
    print('')
    print("Manitude of the difference: {}".format(np.sum(np.square(np.subtract(w, w2)))))
    print('')
    print("Learning Rate: {}".format(lrate))
    print('')
    print("Value of alpha: {}".format(alpha))


fitRegGD(dataTrain[:, 0], 0.2, 0.01, 0.01)
