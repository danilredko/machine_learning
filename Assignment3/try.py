import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters



def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1.0/(1.0 + np.exp(-1.0 * z))

    return s


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
            W1 -- weight matrix of shape (n_h, n_x)
            b1 -- bias vector of shape (n_h, 1)
            W2 -- weight matrix of shape (n_y, n_h)
            b2 -- bias vector of shape (n_y, 1)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    #print(W1.shape, X.shape)
    #print(np.matmul(W1, X).shape, b1.shape)
    Z1 = np.add(np.matmul(W1, X), b1)
    A1 = np.tanh(Z1)
    Z2 = np.add(np.matmul(W2, A1), b2)
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """

    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
    #print(logprobs.shape)
    cost = (-1.0/m) * np.sum(logprobs)

    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect.
                                # E.g., turns [[17]] into 17
    assert(isinstance(cost, float))

    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = (1.0/m) * np.matmul(dZ2, np.transpose(A1))
    db2 = (1.0/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.matmul(np.transpose(W2), dZ2) * (1 - np.power(A1, 2))
    dW1 = (1.0/m) * np.matmul(dZ1, np.transpose(X))
    db1 = (1.0/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


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

def generateData(TrainSize, TestSize):

    mu0 = np.array([0, -1])
    Sigma0 = np.array([[2.0, 0.5], [0.5, 1.0]])
    mu1 = np.array([-1, 1])
    Sigma1 = np.array([[1.0, -1.0], [-1.0, 2.0]])
    Xtrain, tTrain = genData(mu0, mu1, Sigma0, Sigma1, TrainSize)
    Xtest, tTest = genData(mu0, mu1, Sigma0, Sigma1, TestSize)

    return [Xtrain, tTrain, Xtest, tTest]

DATA = generateData(10000,10000)

x_1, x_2 = DATA[0].shape

DATA[1] = DATA[1].reshape(1, 20000)

X = DATA[0].reshape(x_2, x_1)
Y = DATA[1]

n_x = X.shape[0]
n_y = Y.shape[0]

parameters = initialize_parameters(n_x, 3, n_y)
W1 = parameters["W1"]
b1 = parameters["b1"]
W2 = parameters["W2"]
b2 = parameters["b2"]

costs = []
epoches = []
for i in range(0, 1000):
    print("step: {}".format(i))
    # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
    A2, cache = forward_propagation(X, parameters)

    # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
    cost = compute_cost(A2, Y, parameters)
    print(cost)
    costs.append(cost)
    epoches.append(i)

    # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
    grads = backward_propagation(parameters, cache, X, Y)

    # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
    parameters = update_parameters(parameters, grads, learning_rate=0.01)

def plot_loss(ep,lossTrain):

    plt.semilogx(ep, lossTrain)
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.suptitle('Training Loss for bgd')
    plt.show()
plot_loss(epoches, costs)