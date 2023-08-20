import numpy as np
import time
import sys
import scipy.io

import primal

path = "../data/"

# "CLASSIC", "CROSS VALIDATION", "SEARCH GAMMA", "STACKED GENERALIZATION"
mode = "CLASSIC"

gamma = 1
nr_fold = 10
stratified = True
random_state = 10
silent = False
gamma_list = [0.5, 1, 5, 10, 20, 100]
train_ratio = 0.6

try:
    import statsmodels.api as sm
    SM = True
except:
    print("WARNING: statsmodel non installed, "
          "stacked generalization not available", file=sys.stderr)
    SM = False
    
try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    def visualize(features, Y_test, Y_preds):
       # Perform PCA and t-SNE for visualization
        my_pca = PCA(n_components=0.95)
        my_tsne = TSNE(n_components=2)
        test_pca = my_pca.fit_transform(features)
        test_tsne = my_tsne.fit_transform(test_pca)
    
        # Determine the dimensions of test_tsne
        num_data_points, num_classes = test_tsne.shape
    
        # Create scatter plots for true and predicted labels
        fig, (ax_true, ax_preds) = plt.subplots(1, 2, figsize = (10, 4))
        # fig_true, ax_true = plt.subplots()
        # fig_preds, ax_preds = plt.subplots()
    
        # Scatter plot of the dimensions from test_tsne, colored by class labels
        unique_classes = np.unique(Y_test)  # Get unique class labels
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))  # Generate a color map
        for class_label, color in zip(unique_classes, colors):
            verify_true = Y_test == class_label
            verify_preds = Y_preds == class_label
            ax_true.scatter(test_tsne[verify_true, 0], test_tsne[verify_true, 1], color=color, marker = ".")
            ax_preds.scatter(test_tsne[verify_preds, 0], test_tsne[verify_preds, 1], color=color, marker = ".")
        
        # Set axis labels and title
        ax_true.set_title('True labels of test Data: PCA/TSNE')
        ax_preds.set_title('Predicted labels of test Data: PCA/TSNE')
        
        # Remove axis ticks and labels
        ax_true.set_xticklabels([])
        ax_true.set_yticklabels([])
    
        ax_preds.set_xticklabels([])
        ax_preds.set_yticklabels([])
        
        # Show the plot with grid
        ax_true.grid(True)
        ax_preds.grid(True)
        plt.tight_layout()
        plt.show()
    VISUALIZE = True
except:
    VISUALIZE = False

def get_embedding(adj, labels, idx_train):
    """ Computes embedding based on adjacency matrix 
        Source feature i is % of incoming edges pointed from train nodes 
        belonging to label i and target feature i is % of outcoming edges 
        pointing to train nodes belonging to label i
    
    Parameters
    ----------
    adj : ndarray of shape (n_samples, n_samples)
        Adjacency matrix
    
    labels : array-like of shape (n_samples,)
        Target values
        
    idx_train : array-like of shape (n_samples_train,)
        Index of values in train set
    
    Returns
    -------
    phi_X_train : ndarray of shape (n_samples_train, n_classes, 2)
        Train set
        Index 0 of the third dimension represents source embedding
        Index 1 of the third dimension represents target embedding
    
    Y_train : ndarray of shape (n_samples_train,)
        Target values of train set
    
    phi_X_train : ndarray of shape (n_samples_test, n_classes, 2)
        Test set
        Index 0 of the third dimension represents source embedding
        Index 1 of the third dimension represents target embedding
    
    Y_test : ndarray of shape (n_samples_test,)
        Target values of test set
    """
    n = adj.shape[0]
    n_train = len(idx_train)
    n_test = n - n_train
    classes = list(set(labels))
    n_class = len(classes)
    idx_test = np.setdiff1d(np.arange(n), idx_train, assume_unique=True)
    adj_train = adj[idx_train][:, idx_train]
    adj_test_s = adj[idx_test][:, idx_train]
    adj_test_t = adj[idx_train][:, idx_test]
    Y_train = labels[idx_train]
    Y_test = labels[idx_test]

    phi_X_train = np.zeros([n_train, n_class, 2])
    phi_X_test = np.zeros([n_test, n_class, 2])
    for i,c in enumerate(classes):
        phi_X_train[:, i, 0] = adj_train[:, Y_train==c].sum(axis=1)
        phi_X_train[:, i, 1] = adj_train[Y_train==c].sum(axis=0)
        phi_X_test[:, i, 0] = adj_test_s[:, Y_train==c].sum(axis=1)
        phi_X_test[:, i, 1] = adj_test_t[Y_train==c].sum(axis=0)
    phi_X_train = phi_X_train / np.maximum(phi_X_train.sum(axis = 1), 1).reshape([-1, 1, 2])
    phi_X_test = phi_X_test / np.maximum(phi_X_test.sum(axis = 1), 1).reshape([-1, 1, 2])
    return phi_X_train, Y_train, phi_X_test, Y_test

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
            classes_idx[c] = np.random.RandomState(random_state).permutation((Y==c).nonzero()[0])
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
                     adj,
                     labels,
                     nr_fold = 5, 
                     random_state = None, 
                     silent = False, 
                     stratified = False):
    """ Perform cross validation

    Parameters
    ----------
    model : object
        Model used for classification

    adj : ndarray of shape (n_samples, n_samples)
        Adjacency matrix
    
    labels : array-like of shape (n_samples,)
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
        Dictionnary contaning accuracy, micro f1-score, macro f1-score for each fold
        Dictionnary has the following structure:
        {"acc": np.array([acc_fold_1, ..., acc_fold_nr_fold]),
         "micro-f1": np.array([micro-f1_fold_1, ..., micro-f1_fold_nr_fold]),
         "macro-f1": np.array([macro-f1_fold_1, ..., macro-f1_fold_nr_fold]}
    """

    if nr_fold < 2:
        raise ValueError("nr_fold has to be > 2")
    n = adj.shape[0]
    if nr_fold > n:
        print("WARNING: Number of folds %d > Number of training data %d, we'll apply leave-one-out cross validation (Number of folds = Number of training data)"%(nr_fold, n))
        nr_fold = n
    
    copy_model = model
    idx, folds_start = split_nr_fold(labels, nr_fold, random_state, stratified)
    
    scores = dict()
    scores["acc"] = np.zeros(nr_fold)
    scores["micro-f1"] = np.zeros(nr_fold)
    scores["macro-f1"] = np.zeros(nr_fold)
    for i in range(nr_fold):
        if not silent:
            print("Fold [%d/%d]"%(i+1, nr_fold), end="\t")
            start = time.time()
        idx_train = np.hstack([idx[:folds_start[i]], idx[folds_start[i+1]:]])
        phi_X_train, Y_train, phi_X_val, Y_val = get_embedding(adj, labels, idx_train)
        
        copy_model.fit(phi_X_train, Y_train)
        dict_metrics = copy_model.metrics(phi_X_val, Y_val)
        scores["acc"][i] = dict_metrics["acc"]
        scores["micro-f1"][i] = dict_metrics["micro-f1"]
        scores["macro-f1"][i] = dict_metrics["macro-f1"]
        if not silent:
            print("Acc: %.4f, Micro-f1: %.4f, Macro-f1: %.4f"%(dict_metrics["acc"], dict_metrics["micro-f1"], dict_metrics["macro-f1"]))
            print("Time elapsed: %.4fs"%(time.time() - start))
    return scores

def search_gamma_cv(adj,
                    labels,
                    gamma_list, 
                    nr_fold = 5, 
                    random_state = None, 
                    silent = False, 
                    stratified = False):
    """ Perfoms cross validation on each value of gamma in gamma_list

    Parameters
    ----------
    adj : ndarray of shape (n_samples, n_samples)
        Adjacency matrix
    
    labels : array-like of shape (n_samples,)
        Target values

    gamma_list : list
        Values of gamma that are tested

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
        Dictionnary where each keys are values of gamma and values are output of cross validation
    """
    gamma_scores = dict()
    if random_state is None:
        random_state = np.random.randint(0, 2**32, dtype=np.uint32)
    for i, gamma in enumerate(gamma_list):
        if not silent:
            print("%d/%d\t Gamma is set to: %.3f"%(i+1, len(gamma_list), gamma))
        model = primal.svm(gamma)
        gamma_scores[gamma] = cross_validation(model, adj, labels, nr_fold, random_state, silent, stratified)
    return gamma_scores

def stacked_generalization(model,
                           adj,
                           labels,
                           idx_train,
                           nr_fold = 5, 
                           random_state = None, 
                           silent = False, 
                           stratified = False):
    """ Perform stacked generalization

    Parameters
    ----------
    model : object
        Model used for classification

    adj : ndarray of shape (n_samples, n_samples)
        Adjacency matrix
        
    idx_train : array-like of shape (n_samples_train,)
        Index of values in train set
    
    labels : array-like of shape (n_samples,)
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
        raise Exception("statsmodels non installed, stacked generalization not available")
    if nr_fold < 2:
        raise ValueError("nr_fold has to be > 2")
    n_train = idx_train.shape[0]
    if nr_fold > n_train:
        print("WARNING: Number of folds %d > Number of training data %d, we'll apply leave-one-out cross validation (Number of folds = Number of training data)"%(nr_fold, n_train))
        nr_fold = n_train
   
    copy_model = model
    idx_test = np.setdiff1d(np.arange(adj.shape[0]), idx_train)
    Y_train = labels[idx_train]
    adj_train = adj[idx_train][:, idx_train]
    Y_test = labels[idx_test]
    
    classes = list(set(Y_train))
    idx, folds_start = split_nr_fold(Y_train, nr_fold, random_state, stratified)
    
    features_train = np.zeros([n_train, len(classes), 2])
    features_test = np.zeros([len(Y_test), len(classes), 2])
    for i in range(nr_fold):
        if not silent:
            print("Fold [%d/%d]"%(i+1, nr_fold), end="\t")
            start = time.time()
        fold_idx = np.hstack([idx[:folds_start[i]], idx[folds_start[i+1]:]])
        val_idx = idx[folds_start[i]:folds_start[i+1]]
        phi_X_fold, Y_fold, phi_X_val, Y_val = get_embedding(adj_train, Y_train, fold_idx)
        
        # Can be optimized, used to get embedding for test set
        idx_tmp = np.hstack([idx_train[fold_idx], idx_test])
        adj_tmp = adj[idx_tmp][:, idx_tmp]
        Y_tmp = labels[idx_tmp]
        _, _, phi_X_test, Y_tmp = get_embedding(adj_tmp, Y_tmp, np.arange(fold_idx.shape[0]))    
        
        copy_model.fit(phi_X_fold, Y_fold)
        fs_val, ft_val = copy_model.predict_features(phi_X_val)
        fs_test, ft_test = copy_model.predict_features(phi_X_test)
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
        GLM_model = sm.GLM(Y_class, sm.add_constant(features_train[:, i, :]), family = sm.families.Binomial()).fit()
        class_test_proba[:, i] = GLM_model.predict(sm.add_constant(features_test[:, i, :]))
    pre_preds = np.argmax(class_test_proba, axis=1)
    preds = np.zeros(pre_preds.shape)
    for i, c in enumerate(classes):
        preds[pre_preds == i] = c
    out = dict()
    out["classes"] = classes
    out["class_proba"] = class_test_proba
    out["preds"] = preds
    out["metrics"] = primal.compute_metrics(Y_test, preds)
    return out
    
if __name__ == "__main__":
    MATFILE = scipy.io.loadmat(path + "cora.mat")
    G = MATFILE["G"]
    G[ G != 0] = 1
    adj = G - np.eye(G.shape[0])
    labels = MATFILE["label"].reshape(-1)
    print("Suceed loading cora")
    
    print("mode :", mode)
    if mode in ["CLASSIC", "STACKED GENERALIZATION"]:        
        np.random.seed(random_state)
        n = adj.shape[0]
        train_size = int(n*train_ratio)

        idx = np.random.permutation(n)
        idx_train = idx[:train_size]
        
        phi_X_train, Y_train, phi_X_test, Y_test = get_embedding(adj, labels, idx_train)
        model = primal.svm(gamma)
        if mode == "STACKED GENERALIZATION":
            out = stacked_generalization(model,
                                        adj,
                                        labels,
                                        idx_train,
                                        nr_fold,
                                        random_state,
                                        silent,
                                        stratified)
            print(out["metrics"])
            if VISUALIZE:
                visualize(out["class_proba"], Y_test, out["preds"])
        else:
            model.fit(phi_X_train, Y_train)
            score = model.metrics(phi_X_test, Y_test)
            print(score)
            if VISUALIZE:
                fs, ft = model.predict_features(phi_X_test)
                preds = model.predict(phi_X_test)
                visualize(fs + ft, Y_test, preds)
    elif mode == "CROSS VALIDATION":
        model = primal.svm(gamma)
        scores = cross_validation(model, 
                                adj, 
                                labels, 
                                nr_fold = nr_fold, 
                                stratified = stratified, 
                                random_state = random_state, 
                                silent = silent)
        print("Gamma :", gamma)
        print("Acc %.4f, std %.4f"%(scores["acc"].mean(), 
                                    scores["acc"].std()), end="\t")
        print("Micro-F1 %.4f, std %.4f"%(scores["micro-f1"].mean(), 
                                        scores["micro-f1"].std()),
                                        end="\t")
        print("Macro-F1 %.4f, std %.4f"%(scores["macro-f1"].mean(), 
                                        scores["macro-f1"].std()))
    else:
        gamma_scores = search_gamma_cv(adj,
                                    labels,
                                    gamma_list,
                                    nr_fold,
                                    random_state,
                                    silent,
                                    stratified)
        for gamma, scores in gamma_scores.items():
            print("Gamma :", gamma)
            print("Acc %.4f, std %.4f"%(scores["acc"].mean(), 
                                        scores["acc"].std()),
                                        end="\t")
            print("Micro-F1 %.4f, std %.4f"%(scores["micro-f1"].mean(), 
                                            scores["micro-f1"].std()),
                                            end="\t")
            print("Macro-F1 %.4f, std %.4f"%(scores["macro-f1"].mean(), 
                                            scores["macro-f1"].std()))