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
xmin = -5
xmax = 6

def plotDB(w, w0):

    y1 = -(w0+w[0]*xmin)/w[1]
    y2 = -(w0+w[0]*xmax) /w[1]
    plt.plot([xmin, xmax], [y1, y2], linestyle='dashed', color='k', linewidth=1)

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


def q1_3_by_3_graphs(hidden_units, question_letter):

    print('Question 1 ({})'.format(question_letter))
    print('')
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_units,), max_iter=10000,
                        tol=10e-10, solver='sgd', learning_rate_init=.01, activation='tanh')

    Xtrain, tTrain, Xtest, tTest = generateData()

    accuracies = []
    models = []
    weights = []

    for i in range(0, 9):

        model = mlp.fit(Xtrain, tTrain)

        w = mlp.coefs_[0]
        w0 = mlp.intercepts_[0]

        weights.append((w0, w))

        models.append(model)
        acc = mlp.score(Xtest, tTest)
        accuracies.append(acc)
        plt.subplot(3, 3, i+1)
        plt.scatter(Xtrain[:, 0], Xtrain[:, 1], s=2, color=np.where(tTrain == 0.0, 'r', 'b'))
        dfContour(mlp)

        print('Test accuracy: {:.4%}'.format(acc))

    plt.suptitle('Question 1({}): Neural net with {} hidden unit.'.format(question_letter, hidden_units))
    bestNN = np.argmax(accuracies)
    print('Best NN is a graph #{}'.format(bestNN+1))
    plt.show()
    accuracies = np.array(accuracies)
    print(accuracies)
    print(bestNN)
    models = np.array(models)
    print('Best NN Test accuracy: {:.4%}'.format(accuracies[bestNN]))
    mlp.score(Xtest, tTest)
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], s=2, color=np.where(tTrain == 0.0, 'r', 'b'))
    dfContour(models[bestNN])
    plt.title('Question 1({}): Best neural net with {} hidden units.'.format(question_letter, hidden_units))
    print('')
    plt.show()

    return models[bestNN], weights[bestNN][0], weights[bestNN][1], Xtrain, tTrain


bestNN_c, w0_c, w_c, Xtrain_c, tTrain_c = q1_3_by_3_graphs(2, 'c')
bestNN_d, w0_d, w_d, Xtrain_d, tTrain_d = q1_3_by_3_graphs(3, 'd')
bestNN_e, w0_e, w_e, Xtrain_e, tTrain_e = q1_3_by_3_graphs(4, 'e')


def decision_boundaries(hidden_units, question_letter, bestNN, w0, w, Xtrain, tTrain):

    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], s=2, color=np.where(tTrain == 0.0, 'r', 'b'))
    plt.xlim((-5, 6))
    plt.ylim((-5, 6))
    plotDB(w, w0)
    plt.title('Question 1({}): Decision boundaries for {} hidden units'.format(question_letter, hidden_units))
    dfContour(bestNN)
    plt.show()

decision_boundaries(3, 'g', bestNN_d, w0_d, w_d, Xtrain_d, tTrain_d)
decision_boundaries(2, 'h', bestNN_c, w0_c, w_c, Xtrain_c, tTrain_c)
decision_boundaries(4, 'i', bestNN_e, w0_e, w_e, Xtrain_e, tTrain_e)








