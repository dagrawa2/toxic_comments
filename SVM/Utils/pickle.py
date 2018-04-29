import pickle

def load(filename):
	with open(filename, "rb") as fp:
		obj = pickle.load(fp)
	return obj

def save(filename, obj):
	with open(filename, "wb") as fp:
		pickle.dump(obj, fp)
