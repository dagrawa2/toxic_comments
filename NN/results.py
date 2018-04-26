import numpy as np
from Utils import pickle

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
#labels = pickle.load("Objects/labels.list")
results = pickle.load("Objects/results.dict")

epochs = range(1, len(results["train_time"])+1)

file = open("Out/results.txt", "w")
file.write("Results\n\n")
file.write("Script time: "+str(np.round(results["script_time"], 5))+" s\n")
file.write("Training time: "+str(np.round(results["train_time"][-1], 5))+" s\n\n\n")

file.write("Overall accuracy:\n\n")
file.write("epoch,acc,val_acc\n")
for epoch, acc, val_acc in zip(epochs, results["acc_mean"], results["val_acc_mean"]):
	epoch = str(epoch)+","
	acc = str(np.round(acc, 5))+","
	val_acc = str(np.round(val_acc, 5))+"\n"
	file.write(epoch+acc+val_acc)
file.write("\n")
file.write("Overall ROC_AUC: "+str(np.round(results["roc_mean"], 5))+","+str(np.round(results["val_roc_mean"], 5))+"\n\n\n")

for i, label in enumerate(labels):
	file.write(label+" accuracy:\n\n")
	file.write("epoch,acc,val_acc\n")
	for epoch, acc, val_acc in zip(epochs, results["acc"][i], results["val_acc"][i]):
		epoch = str(epoch)+","
		acc = str(np.round(acc, 5))+","
		val_acc = str(np.round(val_acc, 5))+"\n"
		file.write(epoch+acc+val_acc)
	file.write("\n")
	file.write(label+" ROC_AUC: "+str(np.round(results["roc"][i], 5))+","+str(np.round(results["val_roc"][i], 5))+"\n\n\n")

file.write("---\n\n")
file.write("Most toxic comments for each label:\n\n")

for label, comments in zip(labels, results["top_comments"]):
	file.write(label+":\n\n")
	for i, comment in enumerate(comments):
		file.write("Comment "+str(i)+":\n")
		file.write(comment+"\n\n")

file.close()
