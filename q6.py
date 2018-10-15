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