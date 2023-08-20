import numpy as np

RCOND_LIM = 1e-8
LAMBDA = 1

def computeSupport(X, Y, gamma):
    """ Compute support vector of primal problem

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features, 2)
        Training set
        Index 0 of the third dimension represents source embedding
        Index 1 of the third dimension represents target embedding
    
    Y : array-like of shape (n_samples,)
        Target values in {-1, 1}

    gamma : float
        Parameter used when solving the linear system of AsK problem

    Returns
    -------
    omega : ndarray of shape (n_features,)
        "Slope" associated with source embedding
    
    nu : ndarray of shape (n_features,)
        "Slope" associated with target embedding
    
    b1 : float
        Bias on source embedding
    
    b2 : float
        Bias on target embedding
    """

    n, d = X.shape[:2]
    X_s = X[:, :, 0]
    X_t = X[:, :, 1]
    
    H = np.zeros([2*d+2, 2*d+2])
    B = np.zeros(2*d+2)
    
    H[0:d, 0:d] = gamma * np.sum([phi_s.reshape([-1, 1]) @ phi_s.reshape([1, -1]) for phi_s in X_s], axis = 0)
    H[d:2*d, d:2*d] = gamma * np.sum([phi_t.reshape([-1, 1]) @ phi_t.reshape([1, -1]) for phi_t in X_t], axis = 0)
    H[0:d, d:2*d] = H[d:2*d, 0:d] = np.eye(d)
    H[0:d, 2*d] = H[2*d, 0:d] = gamma * X_s.sum(axis = 0)
    H[d:2*d, 2*d+1] = H[2*d+1, d:2*d] = gamma * X_t.sum(axis = 0)
    H[2*d, 2*d] = H[2*d+1, 2*d+1] = gamma*n
    
    B[0:d] = -gamma * np.sum(Y * X_s.T, axis = 1)
    B[d:2*d] = -gamma * np.sum(Y * X_t.T, axis = 1)
    B[2*d] = -gamma * np.sum(Y)
    B[2*d + 1] = -gamma * np.sum(Y)

    if 1/np.linalg.cond(H, 1)>RCOND_LIM or True:
        out = -np.linalg.solve(H, B)
    else:
        out = -np.linalg.solve(H.T @ H + LAMBDA*np.eye(len(H)), H.T@B)
    return out[:d], out[d:2*d], out[2*d], out[2*d+1]

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
        results[i, 3] = results[i, 0] / (results[i, 0] + (results[i, 1] + results[i, 2])/2)
    report["acc"] = ((Y_preds==Y_true).sum())/len(Y_true)
    report["micro-f1"] = results[:, 0].sum() / (results[:, 0].sum() + (results[:, 1].sum() + results[:, 2].sum())/2)
    report["macro-f1"] = results[:, 3].mean()
    return report

class svm:
    def __init__(self,
                 gamma = 1,
                 ):
        self.gamma = gamma
        self.__isfitted = False
    
    def fit(self, X, Y):
        """ Fit the SVM according to the given training data

        Parameters
        ----------
       X : ndarray of shape (n_samples, n_features, 2)
           Training set
           Index 0 of the third dimension represents source embedding
           Index 1 of the third dimension represents target embedding
            
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
        n_features = X.shape[1]
        
        if n_classes == 1:
            raise Exception("Only one class appears in labels")
        if n_classes > 2:
            self.omega = np.zeros([n_classes, n_features])
            self.nu = np.zeros([n_classes, n_features])
            self.b1 = np.zeros([n_classes, 1])
            self.b2 = np.zeros([n_classes, 1])
            for i in range(n_classes):
                Y_class = -np.ones(n_train)
                label = self.classes[i]
                Y_class[Y==label] = 1
                self.omega[i], self.nu[i], self.b1[i], self.b2[i] = computeSupport(X, Y_class, self.gamma)
        else:
            self.omega = np.zeros([1, n_features])
            self.nu = np.zeros([1, n_features])
            self.b1 = np.zeros([1, 1])
            self.b2 = np.zeros([1, 1])
            Y_class = -np.ones(n_train)
            label = self.classes[0]
            Y_class[Y==label] = 1
            self.omega[0], self.nu[0], self.b1[0], self.b2[0] = computeSupport(X, Y_class, self.gamma)
        self.__isfitted = True
        return self    
    
    def predict_features(self, X):
        """ Computes fs and ft

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 2)
            Test set
            Index 0 of the third dimension represents source embedding
            Index 1 of the third dimension represents target embedding

        Returns
        -------
        fs : ndarray of shape (n_test, n_classes)
            Order is the same as in self.classes if multiclass.
            If one versus one, shows only the features of first class in self.classes

        ft : ndarray of shape (n_test, n_classes)
            Order is the same as in self.classes if multiclass
            If one versus one, shows only the features of first class in self.classes
        """
        
        if not self.__isfitted:
            raise Exception("Model is not fitted yet. Call 'fit' first")
        X_s = X[:, :, 0]
        X_t = X[:, :, 1]
        if (X_s.shape[1] != self.omega.shape[1]):
            raise ValueError(f"Embedding should have {self.omega.shape[1]} features but it has {X_s.shape[1]} instead")
        fs = self.omega @ X_s.T + self.b1
        ft = self.nu @ X_t.T + self.b2
        return fs.T, ft.T
    
    def predict(self, X):
        """ Get classes using sum of fs and ft

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 2)
            Test set
            Index 0 of the third dimension represents source embedding
            Index 1 of the third dimension represents target embedding

        Returns
        -------
        preds : ndarray of shape (n_test,)
            Class labels for sample in (Ks, Kt)
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
        X : ndarray of shape (n_samples, n_features, 2)
            Test set
            Index 0 of the third dimension represents source embedding
            Index 1 of the third dimension represents target embedding
            
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
        X : ndarray of shape (n_samples, n_features, 2)
            Test set
            Index 0 of the third dimension represents source embedding
            Index 1 of the third dimension represents target embedding
            
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