import sys
import time

import numpy as np

try:
    import asksvm_utils
    C_FUNCTION = True
except:
    print("Using only Python function", file=sys.stderr)
    C_FUNCTION = False

try:
    import statsmodels.api as sm
    SM = True
except:
    print("WARNING: statsmodel non installed, "
          "stacked generalization not available", file=sys.stderr)
    SM = False

RCOND_LIM = 1e-8
LAMBDA = 1

def computeSupport(K, Y, gamma):
    """ Compute support vector

    Parameters
    ----------
    K : ndarray of shape (n_samples, n_samples)
        Precomputed kernel matrix
   
    Y : array-like of shape (n_samples,)
        Target values in {-1, 1}

    gamma : float
        Parameter used when solving the linear system of AsK problem

    Returns
    -------
    bs : float
        Bias on source

    bt : float
        Bias on target

    weights_s : ndarray of shape (n_samples,)
        Weights on source

    weights_t : ndarray of shape (n_samples,)
        Weights on target    
    """

    n = K.shape[0]
    if C_FUNCTION:
        out = asksvm_utils.computeSupport(K, Y, gamma)
    else:
        A = np.zeros([2*n + 2, 2*n + 2])
        b = np.zeros((2*n + 2))
        H = K * (Y.reshape([-1, 1])@Y.reshape([1, -1]))
    
        b[2:] = 1
    
        A[0, 2:2+n] = A[1, 2+n:] = Y
        A[2:2+n, 0] = A[2+n:, 1] = Y
        A[2:2+n, 2:2+n] = A[2+n:,2+n:] = np.eye(n)/gamma
        A[2:2+n, 2+n:] = H
        A[2+n:, 2:2+n] = H.T

        if 1/np.linalg.cond(A, 1)>RCOND_LIM:
            out = np.linalg.solve(A, b)
        else:
            out = np.linalg.solve(A.T @ A + LAMBDA*np.eye(len(A)), A.T@b)
    return out[0], out[1], out[2+n:]*Y, out[2:2+n]*Y

def compute_metrics(Y_true, Y_preds):
    """ Compute accuracy, micro f1-score and \
        macro f1-score based on Y_true and Y_preds
   
    Parameters
    ----------
    Y_true : array-like of shape (n_samples,)
        True labels
       
    Y_preds : array-like of shape (n_samples,)
        Predicted labels
       
    Returns
    -------
    report : dict
        Dictionnary contaning accuracy, micro f1-score, macro f1-score
        Dictionnary has the following structure:
        {"acc": 0.5,
         "micro-f1": 0.3,
         "macro-f1": 0.5}
    """
    report = dict()
    classes = list(set(Y_true))
    # TP | FP | FN | F1
    results = np.zeros([len(classes), 4])
    for i, c in enumerate(classes):
        results[i, 0] = ((Y_preds==Y_true)*(Y_preds==c)).sum()
        results[i, 1] = ((Y_preds!=Y_true)*(Y_preds==c)).sum()
        results[i, 2] = ((Y_preds!=Y_true)*(Y_true==c)).sum()
        results[i, 3] = results[i, 0] / (results[i, 0] +
                                         (results[i, 1] + results[i, 2])/2)
    report["acc"] = ((Y_preds==Y_true).sum())/len(Y_true)
    report["micro-f1"] = results[:, 0].sum() / (results[:, 0].sum() +
                                                (results[:, 1].sum() +
                                                 results[:, 2].sum())/2)
    report["macro-f1"] = results[:, 3].mean()
    return report

class svm:
    """ AsK-LS algorithm classifier
   
    The implementation is based on "Learning with Asymmetric Kernels: Least
    Squares and Feature Interpretation" from Mingzhen He, Fan He, Lei Shi,
    Xiaolin Huang, Senior Member, IEEE and Johan A.K. Suykens, Fellow, IEEE
   
    This classifier aims to learn by using asymmetric features using
    least-square support vector machine framework
   
    Parameters
    ----------
    gamma : float, default = 1.0
        Regularization parameter
       
    kernel : "precomputed" or callable, default = "precomputed"
        Specifies the kernel type to be used
        If callable the kernel must take as arguments two matrices of shape
        (n_samples_1, n_features), (n_samples_2, n_features) and
        return a kernel matrix of shape (n_samples_1, n_samples_2)
       
    """
    def __init__(self,
                 gamma = 1,
                 kernel = "precomputed"
                 ):
        self.gamma = gamma
        self.kernel = kernel
        self.__isfitted = False

    def fit(self, X, Y):
        """ Fit the SVM according to the given training data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Training vectors, where `n_samples` is the number of samples
            and `n_features` is the number of features
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples)

        Y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            Fitted estimator
        """

        self.__isfitted = False
        self.classes = list(set(Y))
        n_classes = len(self.classes)
        n_train = X.shape[0]
       
        if self.kernel == "precomputed":
            K = X.copy()
        else:
            self.train_data = X
            K = self.kernel(X, X)
       
        if n_classes == 1:
            raise Exception("Only one class appears in labels")
        if n_classes > 2:
            self.weights_s = np.zeros([n_train, n_classes])
            self.weights_t = np.zeros([n_train, n_classes])
            self.bias_s = np.zeros(n_classes)
            self.bias_t = np.zeros(n_classes)
            for i in range(n_classes):
                Y_class = -np.ones(n_train)
                label = self.classes[i]
                Y_class[Y==label] = 1
                (self.bias_s[i],
                 self.bias_t[i],
                 self.weights_s[:, i],
                 self.weights_t[:, i]) = computeSupport(K, Y_class, self.gamma)
        else:
            self.weights_s = np.zeros([n_train, 1])
            self.weights_t = np.zeros([n_train, 1])
            self.bias_s = np.zeros(1)
            self.bias_t = np.zeros(1)
           
            Y_class = -np.ones(n_train)
            label = self.classes[0]
            Y_class[Y==label] = 1
            (self.bias_s[0],
             self.bias_t[0],
             self.weights_s[:, 0],
             self.weights_t[:, 0]) = computeSupport(K, Y_class, self.gamma)
        self.__isfitted = True
        return self
   
    def predict_features(self, X):
        """ Computes fs and ft

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or \
            (n_samples_test, n_samples_train, 2)
            Test vectors
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train)
            Index 0 of the third dimension represents K(X, train_data)
            Index 1 of the third dimension represents K(train_data, X)

        Returns
        -------
        fs : ndarray of shape (n_test, n_classes)
            Order is the same as in self.classes if multiclass.
            If one versus one, shows only the features of first class in
            self.classes

        ft : ndarray of shape (n_test, n_classes)
            Order is the same as in self.classes if multiclass
            If one versus one, shows only the features of first class in
            self.classes
        """
        if not self.__isfitted:
            raise Exception("Model is not fitted yet. Call 'fit' first")
        if self.kernel=="precomputed":
            Ks = X[:, :, 0]
            Kt = X[:, :, 1]
        else:
            Ks = self.kernel(X, self.train_data)
            Kt = self.kernel(self.train_data, X).T
        if (Ks.shape != Kt.shape):
            raise ValueError("Ks and Kt are different shape: "
                             f"{Ks.shape} vs {Kt.shape}")
        if (Ks.shape[1] != self.weights_s.shape[0]):
            raise ValueError("Test kernel matrix should have "
                             f"{self.weights_s.shape[0]} columns "
                             f"but it has {Ks.shape[1]} instead")
        fs = Ks@self.weights_s + self.bias_s
        ft = Kt@self.weights_t + self.bias_t
        return fs, ft
   
    def predict(self, X):
        """ Get classes using sum of fs and ft

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or \
            (n_samples_test, n_samples_train, 2)
            Test vectors
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train)
            Index 0 of the third dimension represents K(X, train_data)
            Index 1 of the third dimension represents K(train_data, X)

        Returns
        -------
        preds : ndarray of shape (n_test,)
            Class labels for sample in X
        """

        fs, ft = self.predict_features(X)
        if len(self.classes)>2:
            pre_preds = np.argmax(fs+ft, axis=1)
        else:
            pre_preds = ((fs+ft)<=0).reshape(-1)
           
        preds = np.zeros(pre_preds.shape)
        for i, label in enumerate(self.classes):
            preds[pre_preds==i] = label
        return preds
   
    def score(self, X, Y):
        """ Return the accuracy on test data and labels

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or \
            (n_samples_test, n_samples_train, 2)
            Test vectors
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train)
            Index 0 of the third dimension represents K(X, train_data)
            Index 1 of the third dimension represents K(train_data, X)
           
        Y : array-like of shape (n_samples_test,)
            True labels of test

        Returns
        -------
        score : float
            Mean accuracy of model with regards to test data and labels
        """
        preds = self.predict(X)
        score = ((preds==Y).sum())/len(Y)
        return score
   
    def metrics(self, X, Y):
        """ Return the accuracy, micro f1-score, macro f1-score on test data
            and labels
           
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or \
            (n_samples_test, n_samples_train, 2)
            Test vectors
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train)
            Index 0 of the third dimension represents K(X, train_data)
            Index 1 of the third dimension represents K(train_data, X)
           
        Y : array-like of shape (n_samples_test,)
            True labels of test
           
        Returns
        -------
        report : dict
            Dictionnary contaning accuracy, micro f1-score, macro f1-score
            Dictionnary has the following structure:
            {"acc": 0.5,
             "micro-f1": 0.3,
             "macro-f1": 0.5}
        """
        preds = self.predict(X)
        return compute_metrics(Y, preds)

def split_nr_fold(Y, nr_fold, random_state = None, stratified = False):
    """ Perform nr_fold split
       
    Parameters
    ----------
    Y : array-like of shape (n_samples,)
        Labels of set (important if stratified)
       
    nr_fold : int
        Number of folds
   
    random_state : int or None, default = None
        Controls the pseudo random number generation used for shuffling

    stratified : bool, default = False
        Stratified K Fold or not
       
    Returns
    -------
    idx : ndarray of shape (n_samples,)
        Shuffled index
       
    folds_start : list of size nr_fold + 1
        Each element of the list is the start of one fold except the last
        element which is n_samples
        Fold i is obtained by folds_start[i]:folds_start[i+1]
    """
    n = Y.shape[0]
    if stratified:
        idx = np.zeros(n, dtype=int)
        classes = list(set(Y))
        classes_idx = dict()
        for c in classes:
            classes_idx[c] = np.random.RandomState(random_state).permutation(
                (Y==c).nonzero()[0])
        start = 0
        folds_start = [0]
        classes_start = dict(zip(classes, np.zeros(len(classes), dtype=int)))
        for i in range(nr_fold):
            for c in classes:
                n_c = len(classes_idx[c])
                size_c = (i+1)*n_c//nr_fold - i*n_c//nr_fold
                idx[start:start+size_c] = classes_idx[c][classes_start[c]:classes_start[c]+size_c]
                start = start + size_c
                classes_start[c] = classes_start[c] + size_c
            folds_start.append(start)
    else:
        folds_start = [i*n//nr_fold for i in range(nr_fold+1)]
        idx = np.random.RandomState(random_state).permutation(n)
    return idx, folds_start

def cross_validation(model,
                     X,
                     Y,
                     nr_fold = 5,
                     random_state = None,
                     silent = False,
                     stratified = False):
    """ Perform cross validation

    Parameters
    ----------
    model : object
        Model used for classification

    X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
        Data vectors, where `n_samples` is the number of samples
        and `n_features` is the number of features
        For model.kernel="precomputed", the expected shape of X is
        (n_samples, n_samples)

    Y : array-like of shape (n_samples,)
        Target values

    nr_fold : int
        Number of folds used in cross validation

    random_state : int or None, default = None
        Controls the pseudo random number generation used for shuffling

    silent : bool, default = False
        Controls if printing time elapsed for each fold or not

    stratified : bool, default = False
        Stratified K Fold or not

    Returns
    -------
    scores : dict
        Dictionnary contaning accuracy, micro f1-score and macro f1-score
        for each fold
        Dictionnary has the following structure:
        {"acc": np.array([acc_fold_1, ..., acc_fold_nr_fold]),
         "micro-f1": np.array([micro-f1_fold_1, ..., micro-f1_fold_nr_fold]),
         "macro-f1": np.array([macro-f1_fold_1, ..., macro-f1_fold_nr_fold]}
                             
    Notes
    -----
    It's better to give a precomputed kernel matrix
    rather than a callable kernel function
    """

    if nr_fold < 2:
        raise ValueError("nr_fold has to be > 2")
    n = X.shape[0]
    if nr_fold > n:
        print("WARNING: Number of folds %d > Number of training data %d, "
              "we'll apply leave-one-out cross validation "
              "(Number of folds = Number of training data)"%(nr_fold, n))
        nr_fold = n
   
    copy_model = model
    idx, folds_start = split_nr_fold(Y, nr_fold, random_state, stratified)
   
    scores = dict()
    scores["acc"] = np.zeros(nr_fold)
    scores["micro-f1"] = np.zeros(nr_fold)
    scores["macro-f1"] = np.zeros(nr_fold)
    for i in range(nr_fold):
        if not silent:
            print("Fold [%d/%d]"%(i+1, nr_fold), end="\t")
            start = time.time()
        val_idx = idx[folds_start[i]:folds_start[i+1]]
        train_idx = np.hstack([idx[:folds_start[i]], idx[folds_start[i+1]:]])
       
        if copy_model.kernel == "precomputed":
            X_train = X[train_idx][:, train_idx]
            X_val = np.stack([X[val_idx][:, train_idx],
                              X[train_idx][:, val_idx].T],
                             axis = 2)
        else:
            X_train = copy_model.kernel(X[train_idx], X[train_idx])
            X_val = np.stack([copy_model.kernel(X[val_idx], X[train_idx]),
                              copy_model.kernel(X[train_idx], X[val_idx]).T],
                             axis = 2)
        Y_train = Y[train_idx]
        Y_val = Y[val_idx]
       
        copy_model.fit(X_train, Y_train)
        dict_metrics = copy_model.metrics(X_val, Y_val)
        scores["acc"][i] = dict_metrics["acc"]
        scores["micro-f1"][i] = dict_metrics["micro-f1"]
        scores["macro-f1"][i] = dict_metrics["macro-f1"]
        if not silent:
            print("Acc: %.4f, Micro-f1: %.4f, "
                  "Macro-f1: %.4f"%(dict_metrics["acc"],
                                    dict_metrics["micro-f1"],
                                    dict_metrics["macro-f1"]))
            print("Time elapsed: %.4fs"%(time.time() - start))
    return scores

def search_gamma_cv(X,
                    Y,
                    gamma_list,
                    kernel = "precomputed",
                    nr_fold = 5,
                    random_state = None,
                    silent = False,
                    stratified = False):
    """ Perfoms cross validation on each value of gamma in gamma_list

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
        Data vectors, where `n_samples` is the number of samples
        and `n_features` is the number of features
        For kernel="precomputed", the expected shape of X is
        (n_samples, n_samples)

    Y : array-like of shape (n_samples,)
        Target values

    gamma_list : list
        Values of gamma that are tested
       
    kernel : "precomputed" or callable, default = "precomputed"
        Specifies the kernel type to be used
        If callable the kernel must take as arguments two matrices of shape
        (n_samples_1, n_features), (n_samples_2, n_features) and
        return a kernel matrix of shape (n_samples_1, n_samples_2)

    nr_fold : int
        Number of folds used in cross validation

    random_state : int or None, default = None
        Controls the pseudo random number generation used for shuffling

    silent : bool, default = False
        Controls if printing or not

    stratified : bool, default = False
        Stratified K Fold or not

    Returns
    -------
    gamma_scores : dict
        Dictionnary where each keys are values of gamma and
        values are output of cross validation
    """
    gamma_scores = dict()
    if random_state is None:
        random_state = np.random.randint(0, 2**32)
    for i, gamma in enumerate(gamma_list):
        if not silent:
            print("%d/%d\t Gamma is set to: %.3f"%(i+1,
                                                   len(gamma_list),
                                                   gamma))
        model = svm(gamma, kernel)
        gamma_scores[gamma] = cross_validation(model,
                                               X,
                                               Y,
                                               nr_fold,
                                               random_state,
                                               silent,
                                               stratified)
    return gamma_scores

def stacked_generalization(model,
                           X_train,
                           Y_train,
                           X_test,
                           Y_test,
                           nr_fold = 5,
                           random_state = None,
                           silent = False,
                           stratified = False):
    """ Perform stacked generalization

    Parameters
    ----------
    model : object
        Model used for classification
       
    X_train : ndarray of shape (n_samples_train, n_features) or \
        (n_samples_train, n_samples_train)
        Training vectors, where `n_samples_train` is the number of samples
        and `n_features` is the number of features
        For model.kernel="precomputed", the expected shape of X is
        (n_samples, n_samples)

    Y_train : array-like of shape (n_samples_train,)
        Target values of training set
       
    X_test : ndarray of shape (n_samples_test, n_features) or \
        (n_samples_test, n_samples_train, 2)
        Test vectors
        For kernel="precomputed", the expected shape of X is
        (n_samples_test, n_samples_train)
        Index 0 of the third dimension represents K(X_test, X_train)
        Index 1 of the third dimension represents K(X_train, X_test)
       
    Y_test : array-like of shape (n_samples_test,)
        Target values of test set

    nr_fold : int
        Number of folds used in cross validation

    random_state : int or None, default = None
        Controls the pseudo random number generation used for shuffling

    silent : bool, default = False
        Controls if printing time elapsed for each fold or not

    stratified : bool, default = False
        Stratified K Fold or not
        
    output : {"class proba", "preds", "metrics"}, default = "class proba"
        Specifies the output

    Returns
    -------
    out : dict
        Dictionnary containing classes, class_proba, preds, metrics
        classes is list of length n_classes
        class_proba is ndarray of shape (n_samples_test, n_classes)
        preds is ndarray of shape (n_samples_test,)
        metrics is dict containing accuracy, micro-f1 score and macro-f1 score,
        it has the following structure:
            {"acc": 0.5,
             "micro-f1": 0.3,
             "macro-f1": 0.5}        
    """
    if not SM:
        raise Exception("statsmodels non installed, "
                        "stacked generalization not available")
    if nr_fold < 2:
        raise ValueError("nr_fold has to be > 2")
    n = X_train.shape[0]
    if nr_fold > n:
        print("WARNING: Number of folds %d > Number of training data %d, "
              "we'll apply leave-one-out cross validation "
              "(Number of folds = Number of training data)"%(nr_fold, n))
        nr_fold = n
   
    copy_model = model
    classes = list(set(Y_train))
    idx, folds_start = split_nr_fold(Y_train,
                                     nr_fold,
                                     random_state,
                                     stratified)
   
    features_train = np.zeros([n, len(classes), 2])
    features_test = np.zeros([len(Y_test), len(classes), 2])
    for i in range(nr_fold):
        if not silent:
            print("Fold [%d/%d]"%(i+1, nr_fold), end="\t")
            start = time.time()
        val_idx = idx[folds_start[i]:folds_start[i+1]]
        train_idx = np.hstack([idx[:folds_start[i]], idx[folds_start[i+1]:]])
       
        if copy_model.kernel == "precomputed":
            X_fold = X_train[train_idx][:, train_idx]
            X_val = np.stack([X_train[val_idx][:, train_idx],
                              X_train[train_idx][:, val_idx].T],
                             axis = 2)
            X_test_fold = X_test[:, train_idx, :]
        else:
            X_fold = copy_model.kernel(X_train[train_idx], X_train[train_idx])
            X_val = np.stack([copy_model.kernel(X_train[train_idx],
                                                X_train[val_idx]),
                              copy_model.kernel(X_train[val_idx],
                                                X_train[train_idx])],
                             axis = 2)
            X_test_fold = np.stack([copy_model.kernel(X_test,
                                                      X_train[train_idx]),
                                    copy_model.kernel(X_train[train_idx],
                                                      X_test).T],
                                   axis = 2)
        Y_fold = Y_train[train_idx]
       
        copy_model.fit(X_fold, Y_fold)
        fs_val, ft_val = copy_model.predict_features(X_val)
        fs_test, ft_test = copy_model.predict_features(X_test_fold)
        features_train[val_idx, :, 0] = fs_val
        features_train[val_idx, :, 1] = ft_val
        features_test[:, :, 0] = features_test[:, :, 0] + fs_test
        features_test[:, :, 1] = features_test[:, :, 1] + ft_test
        if not silent:
            print("Time elapsed: %.4fs"%(time.time() - start))
    features_test = features_test / nr_fold
    class_test_proba = np.zeros([len(Y_test), len(classes)])
    for i, c in enumerate(classes):
        Y_class = np.zeros(len(Y_train))
        Y_class[Y_train==c] = 1
        GLM_model = sm.GLM(Y_class,
                           sm.add_constant(features_train[:, i, :]),
                           family = sm.families.Binomial()).fit()
        class_test_proba[:, i] = GLM_model.predict(sm.add_constant(features_test[:, i, :]))
    pre_preds = np.argmax(class_test_proba, axis=1)
    preds = np.zeros(pre_preds.shape)
    for i, c in enumerate(classes):
        preds[pre_preds == i] = c
    out = dict()
    out["classes"] = classes
    out["class_proba"] = class_test_proba
    out["preds"] = preds
    out["metrics"] = compute_metrics(Y_test, preds)
    return out