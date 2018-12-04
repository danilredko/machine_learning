import numpy as np
import sklearn as skl
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy.random as rnd
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
xmax = 7

def plotDB(w, w0):

    y1 = -(w0+w[0]*xmin)/w[1]
    y2 = -(w0+w[0]*xmax) /w[1]
    plt.plot([xmin, xmax], [y1, y2], linestyle='dashed', color='k', linewidth=1)

def generateData(TrainSize, TestSize):

    mu0 = np.array([0, -1])
    Sigma0 = np.array([[2.0, 0.5], [0.5, 1.0]])
    mu1 = np.array([-1, 1])
    Sigma1 = np.array([[1.0, -1.0], [-1.0, 2.0]])
    Xtrain, tTrain = genData(mu0, mu1, Sigma0, Sigma1, TrainSize)
    Xtest, tTest = genData(mu0, mu1, Sigma0, Sigma1, TestSize)

    return [Xtrain, tTrain, Xtest, tTest]

def q1b():

    mlp = MLPClassifier(hidden_layer_sizes=(1,), max_iter=10000,
                        tol=1e-10, solver='sgd', learning_rate_init=.01, activation='tanh')

    #Xtrain, tTrain, Xtest, tTest = generateData()
    mlp.fit(Xtrain, tTrain)
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], s=2, color=np.where(tTrain == 0.0, 'r', 'b'))
    plt.title('Question 1(b): Neural net with 1 hidden unit.')
    dfContour(mlp)
    plt.show()

#q1b()


def q1_3_by_3_graphs(hidden_units, question_letter, Xtrain, tTrain, Xtest, tTest):

    print('Question 1 ({})'.format(question_letter))
    print('')
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_units,), max_iter=10000,
                        tol=10e-10, solver='sgd', learning_rate_init=.01, activation='tanh')

    accuracies = []
    models = []
    weights = []
    predic_prob = []

    for i in range(0, 9):

        model = mlp.fit(Xtrain, tTrain)
        prob = mlp.predict_proba(Xtrain)
        predic_prob.append(prob)
        w = mlp.coefs_

        w0 = mlp.intercepts_

        weights.append((w0, w))

        models.append(model)
        acc = mlp.score(Xtest, tTest)
        accuracies.append(acc)
        plt.subplot(3, 3, i+1)
        plt.scatter(Xtrain[:, 0], Xtrain[:, 1], s=2, color=np.where(tTrain == 0.0, 'r', 'b'))
        dfContour(mlp)

        print('Test accuracy: {:.4%}'.format(acc))

    plt.suptitle('Question 1({}): Neural net with {} hidden units.'.format(question_letter, hidden_units))
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

    return models[bestNN], weights[bestNN][0], weights[bestNN][1], np.array(accuracies), np.array(predic_prob[bestNN])


#DATA = generateData(1000, 10000)

#bestNN_c, w0_c, w_c, acc_c, predic_prob_c = q1_3_by_3_graphs(2, 'c', DATA[0], DATA[1], DATA[2], DATA[3])
#bestNN_d, w0_d, w_d, acc_d, predic_prob_d = q1_3_by_3_graphs(3, 'd', DATA[0], DATA[1], DATA[2], DATA[3])
#bestNN_e, w0_e, w_e, acc_e, predic_prob_e = q1_3_by_3_graphs(4, 'e', DATA[0], DATA[1], DATA[2], DATA[3])


def decision_boundaries(hidden_units, question_letter, bestNN, w0, w, Xtrain, tTrain):

    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], s=2, color=np.where(tTrain == 0.0, 'r', 'b'))
    plt.xlim((-5, 6))
    plt.ylim((-5, 6))
    plotDB(w, w0)
    plt.title('Question 1({}): Decision boundaries for {} hidden units'.format(question_letter, hidden_units))
    dfContour(bestNN)
    plt.show()

#decision_boundaries(3, 'g', bestNN_d, w0_d[0], w_d[0], DATA[0], DATA[1])
#decision_boundaries(2, 'h', bestNN_c, w0_c[0], w_c[0],  DATA[0], DATA[1])
#decision_boundaries(4, 'i', bestNN_e, w0_e[0], w_e[0],  DATA[0], DATA[1])


def precision_recall_curve(tTest, predic_prob):

    M = 1000
    t = np.linspace(-4, 7, M)
    Precision = np.zeros([M])
    Recall = np.zeros([M])
    Pos = tTest.astype(int)
    numPos = np.sum(Pos)

    for n in range(M):
        PP = (predic_prob[:, 1]>=t[n]).astype(int)
        TP = Pos & PP
        numPP = np.sum(PP)
        numTP = np.sum(TP)
        Precision[n] = numTP/np.float(numPP)
        Recall[n] = numTP/np.float(numPos)
    plt.plot(Recall, Precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.suptitle('Question 1(k): Precision/Recall curve')
    plt.show()


#precision_recall_curve(DATA[1], predic_prob_d)





print("")
print("Question 3")
print("")




def sigmoid(z):

    return 1.0/(1.0 + np.exp(-z))

def q3_set_up():

    DATA = generateData(10000, 10000)
    nn3, w0_d, w_d, acc_d, predic_prob = q1_3_by_3_graphs(3, 'd', DATA[0], DATA[1], DATA[2], DATA[3])
    output_2 = predic_prob[:, 1]
    V = w_d[0]
    v0 = w0_d[0]
    W = w_d[1]
    w0 = w0_d[1]

    return DATA, V, v0, W, w0, output_2.reshape(-1)


def forward(X, V, v0, W, w0):

    U = np.dot(X, V) + v0

    H = np.tanh(U)

    Z = np.dot(H, W) + w0

    O = sigmoid(Z)

    return U, H, Z, O


def diff_of_outputs():

    X, V, v0, W, w0, output2 = q3_set_up()
    output1 = forward(X[0], V, v0, W, w0)[3].reshape(-1) # FIX THIS
    print("The difference between outputs: {}".format(np.sum(np.square(output2 - output1))))

#diff_of_outputs()


def gradient(H, O, T, U, X, Z, W):

    O = O.reshape(-1)
    T = T.reshape(-1)

    dZ = O-T
    DW = (np.dot(np.transpose(H), dZ)) / float(X.shape[0])
    dw0 = ((np.sum(dZ, axis=0)) / float(X.shape[0]))
    dU = (np.dot(Z, W.T)*(1-np.power(H, 2)))
    DV = ((np.dot(X.T, U))/float(X.shape[0]))
    dv0 = ((np.sum(dU, axis=0))/float(X.shape[0]))

    return DW.reshape(3, 1), dw0, DV, dv0


def loss(O, T):

    O = O.reshape(-1)
    T = T.reshape(-1)

    cross_ent = T * np.log(O) + (1-T) * np.log(1-O)

    loss = -np.sum(cross_ent) / float(T.shape[0])

    return loss


DATA = generateData(10000, 10000)

def predict(X, V, v0, W, w0):

    U, H, Z, O = forward(X, V, v0, W, w0)

    return (O > 0.5).astype(int).reshape(X.shape[0], )

def bgd(J, K, lrate):

    '''
    J is the number of units in the hidden layer
    K is the number of epochs of training
    lrate is the learning rate
    '''

    Xtrain = DATA[0]
    yTrain= DATA[1]

    Xtest = DATA[2]
    yTest = DATA[3]



    #input dimension

    '''
    Ntrain, I = np.shape(Xtrain)
    Ntest, I = np.shape(Xtest)
    K = int(np.max(yTrain)+1)

    Ttrain = np.zeros([Ntrain, K])
    Ttest = np.zeros([Ntest, K])
    Ttrain[range(Ntrain), yTrain] = 1
    Ttest[range(Ntest), yTest] = 1
    '''
    sigma = 1.0
    W = sigma*rnd.randn(3, 1)
    #W = W.reshape(-1)
    w0 = np.zeros((1, 1))

    V = sigma*rnd.randn(2, 3)
    v0 = np.zeros((1, 3))

    lossTrainList = []
    accTrainList = []
    accTestList = []
    epoches= []

    for i in range(1, K+1):

        U, H, Z, O = forward(Xtrain, V, v0, W, w0)

        DW, dw0, DV, dv0 = gradient(H, O, yTrain, U, Xtrain, Z, W)

        W = W - lrate*DW
        w0 = w0 - lrate*dw0

        V = V - lrate*DV
        v0 = v0 - lrate*dv0

        if i % 10 == 0:
            print("Step : {}".format(i))
            epoches.append(i)
            trainloss = loss(O, yTrain)
            lossTrainList.append(trainloss)

            #train_accuracy = ((predict(Xtrain, V, v0, W, w0) == yTrain).sum() / float(yTrain.shape[0])) * 100.0
            #test_accuracy = ((predict(yTest, V, v0, W, w0) == yTest).sum() / float(yTest.shape[0])) * 100.0
            #print("Test Accuracy: {}".format(test_accuracy))
            #print("Train Accuracy: {}".format(train_accuracy))
            #print((predict(Xtrain, V, v0, W, w0) & yTrain.astype(int)).sum())
            print("loss : {}" .format(loss(O, yTrain)))

    return np.array(lossTrainList), accTrainList, accTestList, np.array(epoches)

lossTrain, accTrain, accTestList, epoches = bgd(3, 1000, 0.0001)


def plot_loss(ep,lossTrain):

    plt.semilogx(ep, lossTrain)
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.suptitle('Training Loss for bgd')
    plt.show()


plot_loss(epoches, lossTrain)

