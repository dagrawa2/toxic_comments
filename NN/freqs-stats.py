import numpy as np

freqs = np.load("Objects/word_freqs.npy")

file = open("Out/word-ranks-stats.txt", "w")
file.write("Word Rank Distribution\n\n")

accum = [0]
for freq in freqs:
	sum = accum[-1] + freq
	accum.append(sum)
accum = np.array(accum)

file.write("vocabulary size: "+str(accum[-1])+"\n\n")

percents = np.concatenate(( 5*np.arange(20), np.arange(96, 101) ), axis=0)
props = 100*accum/accum[-1]
i = 0
for j in range(len(props)):
	if props[j] > percents[i]:
		file.write(str(percents[i])+"th percentile: "+str(j-1)+"\n")
		i += 1

file.close()