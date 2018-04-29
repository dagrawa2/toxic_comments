import numpy as np
from Utils import pickle

labels = pickle.load("Objects/labels.list")
results = pickle.load("Objects/results.dict")

file = open("Out/results.txt", "w")
file.write("Results\n\n")
file.write("Script time: "+str(np.round(results["script_time"], 5))+" s\n")
file.write("Training time: "+str(np.round(results["train_time"], 5))+" s\n\n")

file.write("Overall accuracy: "+str(np.round(results["acc_mean"], 5))+","+str(np.round(results["val_acc_mean"], 5))+"\n")
file.write("Overall ROC_AUC: "+str(np.round(results["roc_mean"], 5))+","+str(np.round(results["val_roc_mean"], 5))+"\n\n\n")

for i, label in enumerate(labels):
	file.write(label+" accuracy: "+str(np.round(results["acc"][i], 5))+","+str(np.round(results["val_acc"][i], 5))+"\n")
	file.write(label+" ROC_AUC: "+str(np.round(results["roc"][i], 5))+","+str(np.round(results["val_roc"][i], 5))+"\n\n")

file.close()
