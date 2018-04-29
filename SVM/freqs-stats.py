import numpy as np

freqs = np.load("Objects/word_freqs.npy")

file = open("Out/word-ranks-stats.txt", "w")
file.write("Word Rank Distribution\n\n")

accum = [0]
min_df = [np.max(freqs[:,1])]
for freq in freqs:
	sum = accum[-1] + freq[0]
	accum.append(sum)
	df = min(min_df[-1], freq[1])
	min_df.append(df)

accum = np.array(accum)
min_df = np.array(min_df)

file.write("vocabulary size: "+str(freqs.shape[0])+"\n\n")

percents = np.concatenate(( 5*np.arange(20), np.arange(96, 101) ), axis=0)
props = 100*accum/accum[-1]
i = 0
for j in range(len(props)):
	if props[j] > percents[i]:
		file.write(str(percents[i])+"th percentile: "+str(j-1)+", "+str(np.round((j-1)/freqs.shape[0], 4))+", "+str(min_df[j-1])+"\n")
		i += 1


# document frequencies

df = freqs[:,1]
df[::-1].sort()

percents = np.concatenate(( 5*np.arange(20), np.arange(96, 101) ), axis=0)
accum = []
for percent in percents[:-1]:
	accum.append(df[int(percent/100*len(df))])

file.write("\n\n")
for percent, freq in zip(percents[:-1], accum):
	file.write(str(percent)+"th percentile: "+str(freq)+"\n")

file.close()