import numpy as np
import scipy.io

import asksvm

path = "../data/"

nr_fold = 10
stratified = True
random_state = 10
silent = False
kernel = "precomputed"
gamma = 1
gamma_list = [0.5, 1, 5, 10, 20, 100]
train_ratio = 0.6

if __name__=="__main__":
    MATFILE = scipy.io.loadmat(path + "cora.mat")
    K = MATFILE["G"]
    labels = MATFILE["label"].reshape(-1)
    print("Suceed loading data")

    # Choose one mode 
    # mode = "CLASSIC"
    mode = "STACKED GENERALIZATION"
    # mode = "CROSS VALIDATION"
    # mode = "SEARCH GAMMA"

    print("Mode:", mode)
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
        
        model = asksvm.svm(gamma, kernel)
        if mode == "STACKED GENERALIZATION":
            score = asksvm.stacked_generalization(model,
                                                K_train,
                                                Y_train,
                                                K_test,
                                                Y_test,
                                                nr_fold,
                                                random_state,
                                                silent,
                                                stratified)
            print(score)
        else:
            model.fit(K_train, Y_train)
            score = model.metrics(K_test, Y_test)
            print(score)
    elif mode == "CROSS VALIDATION":
        model = asksvm.svm(gamma, kernel)
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
                                        kernel,
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
