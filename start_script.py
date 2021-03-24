import numpy as np

# to read/write csv
import pandas as pd

# for SVM
from SVM import fit_SVM_and_predict

# for other features
from tqdm import tqdm
from kernels import phi, Kernel_func

from multiprocessing import Pool, cpu_count


# Load and standardize features by removing the mean and scaling to unit variance

path_Y = "./data/Ytr0.csv"
path_Xtrain = "./data/Xtr0.csv"
path_Xtest = "./data/Xte0.csv"


def construct_features(X_ATGC, k=8):
    Features = np.zeros([len(X_ATGC) - 1, 4 ** k])

    for idx, sequence in enumerate(
        X_ATGC[0][1:]
    ):  # [1:] pour ne pas prendre la ligne 'Id,seq'
        x = sequence.split(",")[1]
        Features[idx, :] = phi(x, k, kernel="spectrum_efficient")

    return Features


def construct_KERNELmat(XATGC):
    XATGC = XATGC[0]
    Kmat = np.zeros([len(XATGC), len(XATGC)])

    for idx1, sequence1 in enumerate(XATGC):
        x1 = sequence1.split(",")[1]  # keep only the ATGC sequence
        for idx2, sequence2 in enumerate(XATGC):
            x2 = sequence2.split(",")[1]
            if (
                idx1 >= idx2
            ):  # because the matrice is symmetric ==> reduce computation time
                Kmat[idx1, idx2] = Kernel_func(x1, x2, kernel="bio")
                Kmat[idx2, idx1] = Kmat[idx1, idx2]

    return Kmat


def construct_Grammat(XTrainATGC, XTestATGC):

    XTestATGC = XTestATGC[0]
    XTrainATGC = XTrainATGC[0]
    Gmat = np.zeros([len(XTestATGC), len(XTrainATGC)])

    for idx1, sequence1 in enumerate(XTestATGC):
        x1 = sequence1.split(",")[1]
        for idx2, sequence2 in enumerate(XTrainATGC):
            x2 = sequence2.split(",")[1]
            Gmat[idx1, idx2] = Kernel_func(x1, x2, kernel="bio")

    return Gmat


def train_and_compute_pred(path_Y, path_Xtrain, path_Xtest, lambdaa):

    print("path ; ", path_Y, path_Xtrain, path_Xtest)

    YTrain = pd.read_csv(path_Y, usecols=["Bound"])
    XTrain_ATGC = pd.read_csv(path_Xtrain, sep=" ", header=None)
    XTest_ATGC = pd.read_csv(path_Xtest, sep=" ", header=None)

    Y = np.squeeze(np.array(YTrain))

    ###################################################################################################
    #                                           SPECTRUM KERNEL                                       #
    ###################################################################################################

    train_features_KF = construct_features(XTrain_ATGC)
    test_features_KF = construct_features(XTest_ATGC)

    full_KF = np.vstack([train_features_KF, test_features_KF])
    mu = np.mean(full_KF, axis=0)
    std = np.std(full_KF, axis=0)

    train_features_KF = train_features_KF - mu
    train_features_KF[:, std != 0] = (
        train_features_KF[:, std != 0] / std[np.newaxis, std != 0]
    )

    test_features_KF = test_features_KF - mu
    test_features_KF[:, std != 0] = (
        test_features_KF[:, std != 0] / std[np.newaxis, std != 0]
    )

    K_SK = np.dot(train_features_KF, train_features_KF.T)
    G_SK = np.dot(test_features_KF, train_features_KF.T)

    ###################################################################################################
    #                                    GLOBAL ALIGNMENT KERNEL                                      #
    ###################################################################################################

    K_AK = construct_KERNELmat(XTrain_ATGC[1:])
    print("constructed K_AK")
    G_AK = construct_Grammat(XTrain_ATGC[1:], XTest_ATGC[1:])
    print("constructed G_AK")

    K = np.exp(K_AK / 120) + K_SK
    G = np.exp(G_AK / 120) + G_SK

    p = fit_SVM_and_predict(K=K, gram=G, Y=Y, C=lambdaa, get_proba=False)
    predictions = (1 + np.sign(p)) / 2

    return predictions


if __name__ == "__main__":

    path_Y_l = ["./data/Ytr0.csv", "./data/Ytr1.csv", "./data/Ytr2.csv"]
    path_Xtrain_l = ["./data/Xtr0.csv", "./data/Xtr1.csv", "./data/Xtr2.csv"]
    path_Xtest_l = ["./data/Xte0.csv", "./data/Xte1.csv", "./data/Xte2.csv"]
    lambdaa_l = [4.64 * 1e-6, 0.046, 1e-4]

    pool = Pool(cpu_count())
    print(f"nbr CPU : {cpu_count()}")

    results = pool.starmap_async(
        train_and_compute_pred,
        zip(path_Y_l, path_Xtrain_l, path_Xtest_l, lambdaa_l),
    ).get()

    dfs = [None, None, None]
    for i in range(3):
        dfs[i] = pd.DataFrame(
            {
                "Id": np.arange(i * 1000, (i + 1) * 1000),
                "Bound": results[i].squeeze().astype(int),
            }
        )
    dfResult = pd.concat(dfs)

    dfResult.to_csv("./data/submissionSVM.csv", index=False)
