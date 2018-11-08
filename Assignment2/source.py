import numpy as np
import sklearn as sklrn
import sklearn.linear_model as lmodel
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


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

    X, t = sklrn.utils.shuffle(X, t)

    return X, t


# Question 1 b)
mu0 = np.array([0, -1])
Sigma0 = np.array([[2.0, 0.5], [0.5, 1.0]])
mu1 = np.array([-1, 1])
Sigma1 = np.array([[1.0, -1.0], [-1.0, 2.0]])
X, t = genData(mu0, mu1, Sigma0, Sigma1, 10000)

# Question 1 c)


def q1c(X, t):

    plt.scatter(X[:, [0]], X[:, [1]], s=2, c=np.where(t == 0.0, 'r', 'b'))
    plt.suptitle("Question 1(c): sample cluster data (10,000 points per cluster)")
    plt.xlim(-5, 6)
    plt.ylim(-5, 6)
    plt.show()

    return 0

#q1c(X, t)

print("Question 2")
print('')

# Question 2 a)

mu0 = np.array([0, -1])
Sigma0 = np.array([[2.0, 0.5], [0.5, 1.0]])
mu1 = np.array([-1, 1])
Sigma1 = np.array([[1.0, -1.0], [-1.0, 2.0]])
X, t = genData(mu0, mu1, Sigma0, Sigma1, 1000)

# Question 2 b)

logisticReg = lmodel.LogisticRegression(multi_class='ovr')

logisticReg.fit(X, t)

w_0 = logisticReg.intercept_

weights = logisticReg.coef_[0]

mean_accuracy = logisticReg.score(X, t)


print("w_0 : {}". format(w_0))
print("")
print("weight vector: {}".format(weights))
print("")
print("Accuracy: {}".format(mean_accuracy))

print("")
print("Question 2 c)")
print("")


def q2c(X, t):

    plt.scatter(X[:, [0]], X[:, [1]], s=2, c=np.where(t == 0.0, 'r', 'b'))
    x = np.linspace(-5, 6)
    plt.plot(x, np.add(np.dot(-weights[0]/weights[1], x), (-w_0)/weights[1]), 'k')
    plt.xlim(-5, 6)
    plt.ylim(-5, 6)
    plt.suptitle("Question 2(c): training data and decision boundary")
    plt.show()

#q2c(X, t)

# Question 2 d)

print("Question 2 d)")

def q2d(X, t):

    thres = np.arange(-3, 4)
    colors = np.where(thres > 0, 'b', 'red')
    colors[np.argwhere(thres == 0).reshape(1)[0]] = 'k'
    x = np.linspace(-5, 6)
    plt.xlim(-5, 6)
    plt.ylim(-5, 6)
    for i in range(7):
        plt.plot(x, np.add(np.dot(-weights[0]/weights[1], x), (-w_0+thres[i])/weights[1]), c=colors[i])
    plt.scatter(X[:, [0]], X[:, [1]], s=2, c=np.where(t == 0.0, 'r', 'b'))
    plt.suptitle("Question 2(d): decision boundaries for seven thresholds")
    plt.show()

#q2d(X, t)


# Question 2 g)

X, t = genData(mu0, mu1, Sigma0, Sigma1, 10000)


# Question 2 h)


def sigmoid(z):

    return np.divide(1, np.add(1, np.exp(np.negative(z))))


def predict(weights, X, threshold):

    term1 = np.add(w_0, np.dot(weights[0], X[:, [0]]))

    term2 = np.dot(weights[1], X[:, [1]])

    z = np.add(term1, term2)

    p_1 = sigmoid(z)

    return np.where(p_1 > sigmoid(threshold), 1.0 ,0.0)


def q2h(threshold, X, t):

    predictions = predict(weights, X, threshold)

    predicted_positives = np.count_nonzero(predictions, axis=0)

    predicted_negatives = len(predictions) - predicted_positives

    class_1_index = np.argwhere(t == 1).reshape(-1)

    true_positives = np.sum(np.ones(predictions[class_1_index, :].shape) == predictions[class_1_index, :] , axis=0)

    false_positives = predicted_positives - true_positives

    class_0_index = np.argwhere(t == 0).reshape(-1)

    true_negatives = np.sum(np.zeros(predictions[class_0_index, :].shape) == predictions[class_0_index, :] , axis=0)

    false_negatives = predicted_negatives - true_negatives

    recall = np.true_divide(true_positives, (np.add(true_positives, false_negatives)))

    precision = np.true_divide(true_positives, np.add(true_positives, false_positives))

    return predicted_positives, predicted_negatives, true_positives, false_positives, true_negatives, false_negatives, recall, precision


def graph(X, t, threshold):

    plt.scatter(X[:, [0]], X[:, [1]], s=2, c=np.where(t == 0.0, 'r', 'b'))
    x = np.linspace(-5, 6)
    plt.xlim(-5, 6)
    plt.ylim(-5, 6)
    plt.plot(x, np.add(np.dot(-weights[0]/weights[1], x), np.divide(np.add(np.negative(w_0), threshold), weights[1])), c='k')

    plt.show()


def summary(PP, PN, TP, FP, TN, FN, R, P):

    print("Predicted Positives: {}".format(PP))
    print("Predicted Negatives: {}".format(PN))
    print("True Positives: {}".format(TP))
    print("False Positives: {}".format(FP))
    print("True Negatives: {}".format(TN))
    print("False Negatives: {}".format(FN))
    print("Recall : {}".format(R))
    print("Precision: {}".format(P))
    print("____________________________________________________________")


PP, PN, TP, FP, TN, FN, R, P = q2h(threshold=1, X=X, t=t)
#summary(PP, PN, TP, FP, TN, FN, R, P)
#graph(X, t, 1)

# Question 2 i)

def q2i(X, t):

    thresholds = np.linspace(-3, 9, 1000)
    PP, PN, TP, FP, TN, FN, R, P = q2h(np.array(thresholds), X, t)
    plt.plot(R, P)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.suptitle('Question 2(i) precision/recall curve')
    plt.show()

#q2i(X, t)

def q2k(X, t):

    thresholds = np.linspace(-3, 9, 1000)
    PP, PN, TP, FP, TN, FN, R, P = q2h(np.array(thresholds), X, t)
    area = np.trapz(R, P)
    print("Area under the graph: {}".format(area))

q2k(X, t)




#q2k(X, t)


# Question 3


with open('../../mnist.pickle', 'rb') as f:
    Xtrain, Ytrain, Xtest, Ytest = pickle.load(f)


# Question 3 a


def show36Images(X):

    plt.figure()
    for i in range(0, 36):
        #v = X[:, i]
        w = np.reshape(X[i], [28,28])
        plt.subplot(6, 6, i+1)
        plt.axis('off')
        plt.imshow(w, cmap='Greys', interpolation='nearest')
        plt.suptitle("Question 3(a): 16 random MNIST images")
    plt.show()

def q3a():

    my_random = np.random.choice(np.arange(0, 60000, dtype=int), 36, replace=False)

    show36Images(Xtrain[my_random])

#q3a()

def q3b():


    clf = lmodel.LogisticRegression(multi_class='multinomial', solver='lbfgs')

    clf.fit(Xtrain, Ytrain)

    train_score = clf.score(Xtrain, Ytrain)

    test_score = clf.score(Xtest, Ytest)


    print('')
    print("Train score: {} %".format(train_score*100))
    print('')
    print('Test score: {} %'.format(test_score*100))

#q3b()

def q3c():

    all_accuracy = []

    K = range(1, 21)

    for k in K:

        clf = sklrn.neighbors.KNeighborsClassifier(n_neighbors=k, algorithm='brute')

        clf.fit(Xtrain, Ytrain)

        test_score = clf.score(Xtest, Ytest)

        all_accuracy.append(test_score*100)

    all_accuracy = np.array(all_accuracy)

    plt.plot(np.array(K), all_accuracy)
    plt.suptitle("Figure 3(c): KNN test accuracy")
    plt.show()

    bestK = K[np.argmax(all_accuracy)]

    print('')

    print('Best K: {}'.format(bestK))

q3c()





