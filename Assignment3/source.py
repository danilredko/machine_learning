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

    mu0 = np.array([0.0, -1.0])
    Sigma0 = np.array([[2.0, 0.5], [0.5, 1.0]])
    mu1 = np.array([-1.0, 1.0])
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

    #print('Question 1 ({})'.format(question_letter))
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
# Question 3
#-----------------------------------------------------------------------------

def sigmoid(z):

    return 1.0/(1.0 + np.exp(-z))


def q3_set_up():

    nn3, w0_d, w_d, acc_d, predic_prob = q1_3_by_3_graphs(3, 'd', DATA[0], DATA[1], DATA[2], DATA[3])
    output_2 = predic_prob[:, 1]
    V = w_d[0]
    v0 = w0_d[0]
    W = w_d[1]
    w0 = w0_d[1]

    return V, v0, W, w0, output_2.reshape(-1)


def forward(X, V, v0, W, w0):

    U = np.dot(X, V) + v0

    H = np.tanh(U)

    Z = np.dot(H, W) + w0

    O = sigmoid(Z)

    return U, H, Z, O


def diff_of_outputs():

    V, v0, W, w0, output2 = q3_set_up()
    U, H, Z, output1 = forward(DATA[0], V, v0, W, w0)
    output1 = output1.reshape(-1)
    print("The difference between outputs: {}".format(np.sum(np.square(output2 - output1))))


def gradient(H, O, T, U, X, Z, W):

    O = O.reshape(-1)
    T = T.reshape(-1)

    dZ = O-T
    dZ=dZ.reshape(O.shape[0], 1)
    DW = (np.dot(np.transpose(H), dZ)) / float(X.shape[0])
    dw0 = ((np.sum(dZ, axis=0)) / float(X.shape[0]))
    dU = (np.dot(dZ, W.T)*(1-np.power(H, 2)))
    DV = ((np.dot(X.T, dU))/float(X.shape[0]))
    dv0 = ((np.sum(dU, axis=0))/float(X.shape[0]))

    return DW.reshape(3, 1), dw0, DV, dv0


def loss(O, T):

    O = O.reshape(-1)
    T = T.reshape(-1)

    cross_ent = T * np.log(O) + (1-T) * np.log(1-O)

    loss = -np.sum(cross_ent) / float(T.shape[0])

    return loss


def predict(X, V, v0, W, w0):

    U, H, Z, O = forward(X, V, v0, W, w0)

    return (O > 0.5).astype(int).reshape(X.shape[0], )


def accuracy(T, O):

    O = O.reshape(-1)
    T = T.reshape(-1)

    return np.mean(np.equal(T, O))


def bgd(J, K, lrate):

    Xtrain = DATA[0]
    yTrain= DATA[1]

    Xtest = DATA[2]
    yTest = DATA[3]

    sigma = 1.0
    W = sigma*rnd.randn(3, 1)
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
            #print("Step : {}".format(i))
            epoches.append(i)
            trainloss = loss(O, yTrain)
            lossTrainList.append(trainloss)

            predic_train = predict(Xtrain, V, v0, W, w0)
            train_accuracy = accuracy(yTrain, predic_train)
            accTrainList.append(train_accuracy)

            predic_test = predict(Xtest, V, v0, W, w0)
            test_accuracy = accuracy(yTest, predic_test)

            accTestList.append(test_accuracy)
            #print("Test Accuracy: {}".format(test_accuracy))
            #print("Train Accuracy: {}".format(train_accuracy))

            #print("loss : {}" .format(loss(O, yTrain)))
    print("Final Test Accuracy : {}".format(accTestList[-1]))

    return np.array(lossTrainList), accTrainList, accTestList, epoches


def plot_loss(ep, lossTrain, question_letter, learning_type):

    plt.semilogx(ep, lossTrain, 'blue')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.suptitle('Question 3({}): Training Loss for {}'.format(question_letter, learning_type))
    plt.show()


def plot_acc(ep, accTrain, accTest, question_letter, learning_type):

    plt.semilogx(ep, accTrain, 'blue', label='Train Accuracy')
    plt.semilogx(ep, accTest, 'red', label='Test Accuracy')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.suptitle('Question 3({}): training and test accuracy for {}'.format(question_letter, learning_type))
    plt.show()


def plot_final(ec, X, title, K):

    K = int(np.floor(K/2))

    ec = ec[-K:]
    X = X[-K:]
    plt.plot(ec, X, 'blue')
    plt.suptitle(title)
    plt.show()


def sgd(J, K, lrate):

    Xtrain = DATA[0]
    yTrain= DATA[1]

    Xtest = DATA[2]
    yTest = DATA[3]

    Ntrain = Xtrain.shape[0]

    sigma = 1.0
    W = sigma*rnd.randn(3, 1)
    w0 = np.zeros((1, 1))

    V = sigma*rnd.randn(2, 3)
    v0 = np.zeros((1, 3))

    lossTrainList = []
    accTrainList = []
    accTestList = []
    epoches= []

    batchSize = 50
    numEpochs = K

    for i in range(1, numEpochs):

        N1 = 0

        while N1 < Ntrain:


            N2 = np.min([N1+batchSize, Ntrain])

            X = Xtrain[N1:N2]
            Y = yTrain[N1:N2]
            N1 = N2

            U, H, Z, O = forward(X, V, v0, W, w0)

            DW, dw0, DV, dv0 = gradient(H, O, Y, U, X, Z, W)

            W = W - lrate*DW
            w0 = w0 - lrate*dw0

            V = V - lrate*DV
            v0 = v0 - lrate*dv0

        U, H, Z, O = forward(Xtrain, V, v0, W, w0)

        #print("Step : {}".format(i))
        epoches.append(i)
        trainloss = loss(O, yTrain)
        lossTrainList.append(trainloss)

        predic_train = predict(Xtrain, V, v0, W, w0)
        train_accuracy = accuracy(yTrain, predic_train)
        accTrainList.append(train_accuracy)

        predic_test = predict(Xtest, V, v0, W, w0)
        test_accuracy = accuracy(yTest, predic_test)

        accTestList.append(test_accuracy)
        #print("Test Accuracy: {}".format(test_accuracy))
        #print("Train Accuracy: {}".format(train_accuracy))

        #print("loss : {}" .format(loss(O, yTrain)))

    print("Final Test Accuracy : {}".format(accTestList[-1]))

    return np.array(lossTrainList), accTrainList, accTestList, epoche


'''
print("")
print("Question 3")
DATA = generateData(10000, 10000)
print("")
print("Question 3 (a)")
diff_of_outputs()
print("")
print('Question 3 (b)')
K = 1000
lossTrain, accTrainList, accTestList, epoches = bgd(3, K, 1)
plot_loss(epoches, lossTrain, 'b', 'bgd')
plot_acc(epoches, accTrainList, accTestList, 'b','bgd')
plot_final(epoches, accTestList, 'Question 3(b): final test accuracy for bgd', K)
plot_final(epoches, lossTrain, 'Question 3(b): final training loss for bgd', K)
print("")
print("Question 3(c): ")
K = 20
lossTrain, accTrainList, accTestList, epoches = sgd(3, K, 1)
plot_acc(epoches, accTrainList, accTestList, 'c', 'sgd')
plot_loss(epoches, lossTrain, 'c', 'sgd')
plot_final(epoches, accTestList, 'Question 3(c): final test accuracy for sgd', K)
plot_final(epoches, lossTrain, 'Question 3(c): final training loss for sgd', K)
'''