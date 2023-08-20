import numpy as np
import scipy.io

import asksvm

path = "../data/"

# "cora" or "monks"
data = "monks"

# "CLASSIC", "CROSS VALIDATION", "SEARCH GAMMA", "STACKED GENERALIZATION"
mode = "CLASSIC"

# "T" or "SNE"
kernel_type = "T"

gamma = 1
nr_fold = 10
stratified = True
random_state = 10
silent = False
gamma_list = [0.5, 1, 5, 10, 20, 100]
train_ratio = 0.6
sigma = 1 # Used in UCI

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

def UCI(kernel_type):
    def kernel(Z, kernel_type, sigma = 1):
        if kernel_type == "T":
            def T(X, Y):
                n_X = X.shape[0]
                n_Y = Y.shape[0]
                K = np.zeros([n_X, n_Y])
                for i, x in enumerate(X):
                    tmp = ((1+np.linalg.norm(x-Z, axis=1))**(-1)).sum()
                    for j, y in enumerate(Y):
                        K[i,j] = ((1+np.linalg.norm(x-y))**(-1))/tmp
                return K
            return T
        elif kernel_type == "SNE":
            def SNE(X, Y):
                n_X = X.shape[0]
                n_Y = Y.shape[0]
                K = np.zeros([n_X, n_Y])
                for i, x in enumerate(X):
                    tmp = np.exp(-np.linalg.norm((x - Z), axis=1)/sigma**2).sum()
                    for j, y in enumerate(Y):
                        K[i,j] = np.exp(-np.linalg.norm(x-y)/sigma**2)/tmp
                return K
            return SNE
        
    MATFILE = scipy.io.loadmat(path + "monks_1.mat")
    X_train = MATFILE["X"].astype(int)
    Y_train = MATFILE["Y"].reshape(-1)
    X_test = MATFILE["X_test"].astype(int)
    Y_test = MATFILE["Y_test"].reshape(-1)
    print("Suceed loading monks")

    model = asksvm.svm(gamma, kernel = kernel(X_train, kernel_type, sigma))
    model.fit(X_train, Y_train)
    print(model.metrics(X_test, Y_test))
    if VISUALIZE:
        fs, ft = model.predict_features(X_test)
        preds = model.predict(X_test)
        visualize(fs + ft, Y_test, preds)

def graph(mode):
    global gamma
    MATFILE = scipy.io.loadmat(path + "cora.mat")
    K = MATFILE["G"]
    labels = MATFILE["label"].reshape(-1)
    print("Suceed loading cora")
    
    print("mode :", mode)
    if mode in ["CLASSIC", "STACKED GENERALIZATION"]:
        np.random.seed(random_state)
        n = K.shape[0]
        train_size = int(n*train_ratio)

        idx = np.random.permutation(n)
        idx_train = idx[:train_size]
        idx_test = idx[train_size:]

        K_train = K[idx_train][:, idx_train]
        Y_train = labels[idx_train]
        K_test = np.stack([K[idx_test][:, idx_train],
                        K[idx_train][:, idx_test].T],
                        axis = 2)
        Y_test = labels[idx_test]
        
        model = asksvm.svm(gamma, "precomputed")
        if mode == "STACKED GENERALIZATION":
            out = asksvm.stacked_generalization(model,
                                                K_train,
                                                Y_train,
                                                K_test,
                                                Y_test,
                                                nr_fold,
                                                random_state,
                                                silent,
                                                stratified)
            print(out["metrics"])
            if VISUALIZE:
                visualize(out["class_proba"], Y_test, out["preds"])
        else:
            model.fit(K_train, Y_train)
            score = model.metrics(K_test, Y_test)
            print(score)
            if VISUALIZE:
                fs, ft = model.predict_features(K_test)
                preds = model.predict(K_test)
                visualize(fs + ft, Y_test, preds)
    elif mode == "CROSS VALIDATION":
        model = asksvm.svm(gamma, "precomputed")
        scores = asksvm.cross_validation(model, 
                                    K, 
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
        gamma_scores = asksvm.search_gamma_cv(K,
                                        labels,
                                        gamma_list,
                                        "precomputed",
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

if __name__=="__main__":
    if data == "cora":
        graph(mode)
    elif data == "monks":
        UCI(kernel_type)