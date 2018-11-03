import sys
import pickle
from vocab import Vocab, VocabEntry
import numpy as np

np.random.seed(0)

def main():
	l1 = sys.argv[1]

	embed_size = 300
	src_embed_file = "./align_embed/data.fast/wiki.%s-en.vec" % (l1)
	tgt_embed_file = "./align_embed/data.fast/wiki.en-en.vec"

	weights_file = "./align_embed/data.weights/wiki.%s-en.weights" % (l1)
	vocab_file = "./data/vocab.%s-en.bin" % (l1)
	src_embed = {}
	tgt_embed = {}

	with open(src_embed_file, 'r', encoding='utf-8') as file:
		for l in file:
			line = l.strip().split(" ")
			src_embed[line[0]] = np.array(line[1:]).astype(np.float)
	print("loading src_embed_dict done")

	with open(tgt_embed_file, 'r', encoding='utf-8') as file:
		for l in file:
			line = l.strip().split(" ")
			tgt_embed[line[0]] = np.array(line[1:]).astype(np.float)
	print("loading tgt_embed_dict done")


	vocab = pickle.load(open(vocab_file, 'rb'))

	src_weights_matrix = np.zeros((len(vocab.src), embed_size))
	for k, v in vocab.src.word2id.items():
		if k in src_embed:
			src_weights_matrix[v] = src_embed[k]
		else:
			src_weights_matrix[v] = np.random.normal(scale=0.6, size=(embed_size, ))
	print("creating src_weights_matrix done")


	tgt_weights_matrix = np.zeros((len(vocab.tgt), embed_size))
	for k, v in vocab.tgt.word2id.items():
		if k in tgt_embed:
			tgt_weights_matrix[v] = tgt_embed[k]
		else:
			tgt_weights_matrix[v] = np.random.normal(scale=0.6, size=(embed_size, ))
	print("creating tgt_weights_matrix done")

	weights_matrix = {"src": src_weights_matrix, "tgt": tgt_weights_matrix}
	pickle.dump(weights_matrix, open(weights_file, 'wb'))


if __name__ == "__main__":
	main()


