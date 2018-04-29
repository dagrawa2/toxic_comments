import time
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from Utils import pickle

np.random.seed(123)
time_0 = time.time()

X_train = sp.sparse.load_npz("Objects/X_train.npz")  # [1000:1050,:40]
Y_train = np.load("Objects/Y_train.npy")  # [1000:1050]

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.5, random_state=456)
#X_test = sp.sparse.load_npz("Objects/X_train.npz")[3000:3050,:40]
#Y_test = np.load("Objects/Y_train.npy")[3000:3050]

num_tasks = Y_train.shape[1]
time_1 = time.time()
classifier = OneVsRestClassifier(LinearSVC())
classifier.fit(X_train, Y_train)
time_2 = time.time()

results = {}
results.update({"train_time": time_2-time_1})
pred = classifier.predict(X_train)
results.update({"acc": np.array([accuracy_score(Y_train[:,i], pred[:,i]) for i in range(num_tasks)])})
results.update({"acc_mean": np.mean(results["acc"])})
results.update({"roc": np.array([roc_auc_score(Y_train[:,i], pred[:,i]) for i in range(num_tasks)])})
results.update({"roc_mean": np.mean(results["roc"])})
pred = classifier.predict(X_test)
results.update({"val_acc": np.array([accuracy_score(Y_test[:,i], pred[:,i]) for i in range(num_tasks)])})
results.update({"val_acc_mean": np.mean(results["val_acc"])})
results.update({"val_roc": np.array([roc_auc_score(Y_test[:,i], pred[:,i]) for i in range(num_tasks)])})
results.update({"val_roc_mean": np.mean(results["val_roc"])})


#nrows = 10
X_test = sp.sparse.load_npz("Objects/X_test.npz")  # [:nrows,:X_train.shape[1]]
Y_test = classifier.predict(X_test)
submission = pd.read_csv("../Data/sample_submission.csv")  # , nrows=nrows)
submission.iloc[:,1:] = Y_test
submission.to_csv("Objects/submission.csv", index=False)

results.update({"script_time": time.time()-time_0})

pickle.save("Objects/results.dict", results)
