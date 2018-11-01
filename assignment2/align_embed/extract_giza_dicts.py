
import io
import sys

transitions = []
src_lang = {}
tgt_lang = {}

print(sys.argv)

l1 = sys.argv[1]
typ = sys.argv[2]
threshold = float(sys.argv[3])

path = "data.giza/%s/" % (l1)

transition_file = path+"%s.en-%s.dict.t3.final" % (typ, l1)
src_vocab_file = path+"%s.en-%s.dict.trn.src.vcb" % (typ, l1)
tgt_vocab_file = path+"%s.en-%s.dict.trn.trg.vcb" % (typ, l1)
dict_file = path+"%s.en-%s.dict.final" % (typ, l1)

with open(src_vocab_file, 'r', encoding='utf-8') as file:
	for line in file:
		words = line.split(" ")
		src_lang[words[0]] = words[1]

with open(tgt_vocab_file, 'r', encoding='utf-8') as file:
	for line in file:
		words = line.split(" ")
		tgt_lang[words[0]] = words[1]

with open(transition_file, 'r', encoding='utf-8') as file:
	for line in file:
		words = line.split(" ")
		if (words[0] in src_lang and words[1] in tgt_lang):
			pair = (src_lang[words[0]], tgt_lang[words[1]])
			prob = float(words[2])

			if prob > threshold:
				transitions.append(pair)

with open(dict_file, 'w', encoding='utf-8') as file:
	for pair in transitions:
		file.write(pair[0]+" "+pair[1]+"\n")

#print(transitions)

