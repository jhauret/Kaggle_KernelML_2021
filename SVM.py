import numpy as np

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

cvxopt_solvers.options["show_progress"] = False


def SVM(K, Y, C=1.0):
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
    alphas = np.array(sol["x"])

    return alphas


def pred(gram, Y, alphas):
    label = -1.0 * np.logical_not(Y) + 1.0 * Y
    return gram.dot(alphas * label)


def fit_SVM_and_predict(K, gram, Y, C=1, get_proba=True):
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
    # TODO : implement the intercept

    alphas = SVM(K, Y, C=C)
    alphas = np.squeeze(alphas)
    y = pred(gram, Y, alphas)
    if get_proba:
        return y
    else:
        return np.sign(y)

