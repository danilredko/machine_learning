import numpy as np
import sklearn as skl
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

from bonnerlib2 import dfContour

print("Question 1")
print('')


def genData(mu0, mu1, Sigma0, Sigma1, N):

    cluster0 = np.random.multivariate_normal(mu0, Sigma0, N)

    cluster1 = np.random.multivariate_normal(mu1, Sigma1, N)

    t = np.ones((2*N,))
    t[:N] = np.zeros((N,))

    X = np.zeros((2*N, 2))
    X[:N, :] = cluster0
    X[-N:, :] = cluster1

    X, t = skl.utils.shuffle(X, t)

    return X, t


def generateData():

    mu0 = np.array([0, -1])
    Sigma0 = np.array([[2.0, 0.5], [0.5, 1.0]])
    mu1 = np.array([-1, 1])
    Sigma1 = np.array([[1.0, -1.0], [-1.0, 2.0]])
    Xtrain, tTrain = genData(mu0, mu1, Sigma0, Sigma1, 1000)
    Xtest, tTest = genData(mu0, mu1, Sigma0, Sigma1, 10000)

    return Xtrain, tTrain, Xtest, tTest


def q1b():

    mlp = MLPClassifier(hidden_layer_sizes=(1,), max_iter=10000,
                        tol=1e-10, solver='sgd', learning_rate_init=.01, activation='tanh')

    Xtrain, tTrain, Xtest, tTest = generateData()
    mlp.fit(Xtrain, tTrain)
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], s=2, color=np.where(tTrain == 0.0, 'r', 'b'))
    plt.title('Question 1(b): Neural net with 1 hidden unit.')
    dfContour(mlp)
    plt.show()

#q1b()


def q1c():

    mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=10000,
                        tol=1e-10, solver='sgd', learning_rate_init=.01, activation='tanh')

    Xtrain, tTrain, Xtest, tTest = generateData()

    accuracies = []
    models = []

    for i in range(0, 9):

        model = mlp.fit(Xtrain, tTrain)
        models.append(model)
        acc = mlp.score(Xtest, tTest)
        accuracies.append(acc)
        plt.subplot(3, 3, i+1)
        plt.scatter(Xtrain[:, 0], Xtrain[:, 1], s=2, color=np.where(tTrain == 0.0, 'r', 'b'))
        dfContour(model)
        print('Test accuracy: {:.4%}'.format(acc))

    plt.suptitle('Question 1(c): Neural net with 2 hidden unit.')
    bestNN = np.argmax(accuracies)
    print('Best NN is a graph #{}'.format(bestNN+1))
    plt.show()
    accuracies = np.array(accuracies)
    models = np.array(models)
    print('Best NN Test accuracy: {:.4%}'.format(accuracies[bestNN]))
    mlp.score(Xtest, tTest)
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], s=2, color=np.where(tTrain == 0.0, 'r', 'b'))
    dfContour(models[bestNN])
    plt.title('Question 1(c): Best neural net with 2 hidden units.')
    plt.show()


q1c()


