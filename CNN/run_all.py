import os

scripts = ["freqs", "freqs-stats", "tok", "lengths", "lengths-stats", "pad", "shapes", "model", "results"]

for script in scripts:
	print("Running "+script+" .py . . . ")
	os.system("python "+script+".py")
