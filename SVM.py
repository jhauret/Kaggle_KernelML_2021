import numpy as np

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

cvxopt_solvers.options["show_progress"] = False


def SVM(K, Y, C=1.0):

    assert np.allclose(K, K.T, rtol=1e-5)


    N = K.shape[0]
    G = np.vstack([np.eye(N, N), -np.eye(N, N)])

    h = np.concatenate([np.repeat(C, N), np.repeat(0, N)]).astype(np.double)
    label = -1.0 * np.logical_not(Y) + 1.0 * Y
    P = np.diag(label) @ K @ np.diag(label)
    np.testing.assert_array_equal(P, P.transpose())

    A = np.matrix(label)
    b = np.zeros(1)
    q = -np.ones((N, 1))

    P = cvxopt_matrix(P)
    G = cvxopt_matrix(G)
    h = cvxopt_matrix(h)
    b = cvxopt_matrix(b)
    A = cvxopt_matrix(A)
    q = cvxopt_matrix(q)

    sol = cvxopt_solvers.qp(
        P=P,
        q=q,
        G=G,
        h=h,
        A=A,
        b=b,
    )
    alphas = np.array(sol["x"]).flatten()

    S = 1e-15 < alphas
    M = np.logical_and(S, (alphas < C - 1e-15))

    b = np.mean(
        [label[n] - (alphas * label).dot(K[n, :]) for n in np.where(M)[0]]
    )

    return alphas, b


def pred(gram, Y, alphas, b):
    label = -1.0 * np.logical_not(Y) + 1.0 * Y
    return gram.dot(alphas * label) + b


def fit_SVM_and_predict(K, gram, Y, C=1, get_proba=True, ):
    """
    fit the SVM on the training data, and predict

    Args:
        K (matrix): the gram matrix from the training data
            for the linear kernel : K = np.dot(X_train , X_train.T)
        gram (matrix): the gram matrix used for prediction]
            for the linear kernel : gram = np.dot(X_test , X_train.T)
        Y (boolean array): the labels of the training set
        C (int, optional): the regularization parameter, the higher it is
        the less regularized the solution is
        get_proba (bool, optional): when True, the algorithm outputs an array
        of float, which signs are the prediction. If False, this array is
        passed through the sign function

    Returns:
        array: prediction
    """

    alphas, b = SVM(K, Y, C=C)
    y = pred(gram, Y, alphas, b)
    if get_proba:
        return y
    else:
        return np.sign(y)
