import sys
import pickle
from vocab import Vocab, VocabEntry
import numpy as np

np.random.seed(0)

def main():
	l1 = sys.argv[1]

	embed_size = 300
	embed_file = "./align_embed/data.fast/wiki.%s-en.vec" % (l1)
	weights_file = "./align_embed/data.weights/wiki.%s-en.weights" % (l1)
	vocab_file = "./data/vocab.%s-en.bin" % (l1)
	embed = {}
	
	with open(embed_file, 'r', encoding='utf-8') as file:
		for l in file:
			line = l.strip().split(" ")
			embed[line[0]] = np.array(line[1:]).astype(np.float)
	
	print("loading embed dict done")
	vocab = pickle.load(open(vocab_file, 'rb'))
	vocab_len = len(vocab.src)
	print(vocab_len)

	weights_matrix = np.zeros((vocab_len, embed_size))
	for k, v in vocab.src.word2id.items():
		if k in embed:
			weights_matrix[v] = embed[k]
		else:
			weights_matrix[v] = np.random.normal(scale=0.6, size=(embed_size, ))

	print("creating weights_matrix done")
	pickle.dump(weights_matrix, open(weights_file, 'wb'))


if __name__ == "__main__":
	main()

