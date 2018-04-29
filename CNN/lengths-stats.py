import numpy as np

lengths = np.load("Objects/comment_lengths.npy")

file = open("Out/comment-lengths-stats.txt", "w")
file.write("Comment Length Distribution\n\n")
file.write("number of comments: "+str(len(lengths))+"\n\n")

file.write("min: "+str(np.min(lengths))+"\n")
file.write("median: "+str(np.median(lengths))+"\n")
file.write("max: "+str(np.max(lengths))+"\n")
file.write("mean: "+str(np.mean(lengths))+"\n\n")

percents = np.concatenate(( 5*np.arange(20), np.arange(96, 101) ), axis=0)
percentiles = np.percentile(lengths, percents)
for i in range(len(percents)):
	file.write(str(percents[i])+"th percentile: "+str(percentiles[i])+"\n")

file.close()
