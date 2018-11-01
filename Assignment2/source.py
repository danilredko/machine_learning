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

logisticReg = lmodel.LogisticRegression()

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
    x = np.linspace(-5, 5, 50)
    plt.plot(x, np.add(np.dot(weights[1].T, x), w_0), 'k')
    plt.suptitle("Question 2(c): training data and decision boundary")
    plt.show()

#q2c(X, t)

# Question 2 d)

print("Question 2 d)")

def q2d(X, t):

    thres = np.arange(-3, 4)
    colors = np.where(thres > 0, 'b', 'red')
    colors[np.argwhere(thres == 0).reshape(1)[0]] = 'k'
    x = np.linspace(-5, 5, 10)
    for i in range(7):
        plt.plot(x, np.add(np.dot(weights[1].T, x), w_0-thres[i]), c=colors[i])

    plt.scatter(X[:, [0]], X[:, [1]], s=2, c=np.where(t == 0.0, 'r', 'b'))
    plt.suptitle("Question 2(d): decision boundaries for seven thresholds")
    plt.show()

q2d(X, t)



# Question 2 g)

X, t = genData(mu0, mu1, Sigma0, Sigma1, 10000)

# Question 2 h)

predictions = logisticReg.predict(X)

predicted_positives = np.count_nonzero(predictions)

predicted_negatives = len(predictions) - predicted_positives

class_1_index = np.argwhere(t == 1).reshape(-1)
true_positives = np.equal(t[class_1_index], predictions[class_1_index]).sum()
false_positives = (len(t[class_1_index]) - true_positives)


class_0_index = np.argwhere(t == 0).reshape(-1)
true_negatives = np.equal(t[class_0_index], predictions[class_0_index]).sum()
false_negatives = (len(t[class_0_index]) - true_negatives)

recall = float(true_positives) / (true_positives+false_negatives)

precision = float(true_positives) / (true_positives+false_positives)


print("Predicted Positives: {}".format(predicted_positives))
print("Predicted Negatives: {}".format(predicted_negatives))
print("True Positives: {}".format(true_positives))
print("False Positives: {}".format(false_positives))
print("True Negatives: {}".format(true_negatives))
print("False Negatives: {}".format(false_negatives))
print("Recall : {}".format(recall))
print("Precision: {}".format(precision))

# Question 2 i)


plt.scatter(X[:, [0]], X[:, [1]], s=2, c=np.where(t == 0.0, 'r', 'b'))
x = np.linspace(-5, 5, 10)
plt.plot(x, np.add(np.dot(weights[1].T, x), w_0-1), c='k')
plt.show()