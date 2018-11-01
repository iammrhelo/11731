import numpy as np
from fasttext import FastVector
import sys

# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=2):
	"""Utility function to normalize the rows of a numpy array."""
	l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
	l2[l2==0] = 1
	return a / np.expand_dims(l2, axis)


def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
	"""
	Source and target dictionaries are the FastVector objects of
	source/target languages. bilingual_dictionary is a list of 
	translation pair tuples [(source_word, target_word), ...].
	"""
	source_matrix = []
	target_matrix = []

	for (source, target) in bilingual_dictionary:
	    if source in source_dictionary and target in target_dictionary:
	        source_matrix.append(source_dictionary[source])
	        target_matrix.append(target_dictionary[target])

	# return training matrices
	return np.array(source_matrix), np.array(target_matrix)


def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
	"""
	Source and target matrices are numpy arrays, shape
	(dictionary_length, embedding_dimension). These contain paired
	word vectors from the bilingual dictionary.
	"""
	# optionally normalize the training vectors
	if normalize_vectors:
	    source_matrix = normalized(source_matrix)
	    target_matrix = normalized(target_matrix)

	# perform the SVD
	product = np.matmul(source_matrix.transpose(), target_matrix)
	U, s, V = np.linalg.svd(product)

	# return orthogonal transformation which aligns source language to the target
	return np.matmul(U, V)


def main():
	print(sys.argv)
	l1 = sys.argv[1]
	typ = sys.argv[2]

	fast_data_path = "./data.fast/"
	bilingual_dict_path = "./data.giza/%s/%s.en-%s.dict.final" % (l1, typ, l1)
	bilingual_dict = []

	with open(bilingual_dict_path, 'r', encoding='utf-8') as file:
		for line in file:
			words = line.strip().split(" ")
			bilingual_dict.append((words[0], words[1]))


	src_embed_file = fast_data_path+'wiki.%s.vec' % (l1)
	tgt_embed_file = fast_data_path+'wiki.en.vec'
	src_embed_transformed_file = fast_data_path+'wiki.%s-en.vec' % (l1)

	src_dict = FastVector(vector_file = src_embed_file)
	tgt_dict = FastVector(vector_file = tgt_embed_file)
	
	print("FastVector Loaded")
	# form the training matrices
	source_matrix, target_matrix = make_training_matrices(src_dict, tgt_dict, bilingual_dict)

	# learn and apply the transformation
	transform = learn_transformation(source_matrix, target_matrix)
	src_dict.apply_transform(transform)
	print("transform done")
	
	src_dict.export(src_embed_transformed_file)


if __name__ == "__main__":
	main()










