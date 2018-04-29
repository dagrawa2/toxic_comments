import os

dirs = [int(dir) for dir in os.listdir("Results/")]
if len(dirs) > 0:
	next = max(dirs) + 1
else:
	next = 0

path = "Results/" + str(next)

print("Gathering results into "+path)

os.system("mkdir \""+path+"\"")

files = ["Objects/results.dict", "Objects/model.h5", "Objects/submission.csv", "Out/results.txt", "Out/shapes.txt", "model.py"]

for file in files:
	char_len = len(file)
	after_slash = 0
	for i in range(1, char_len+1):
		if file[-i] == "/":
			after_slash = 1-i
			break
	name = file[after_slash:]
	os.system("cp "+file+" "+path+"/"+name)
