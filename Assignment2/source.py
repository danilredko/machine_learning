import numpy as np
import sklearn as sklrn
import matplotlib.pyplot as plt

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

q1c(X, t)

