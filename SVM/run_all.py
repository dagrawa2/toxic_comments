import os

scripts = ["freqs", "freqs-stats", "tok", "shapes", "model", "results"]

for script in scripts:
	print("Running "+script+" .py . . . ")
	os.system("python "+script+".py")
