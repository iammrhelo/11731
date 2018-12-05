
lang = 'de-nl'
tgt = lang.split('-')[1]
input_file = '../iwslt2017/normal/test.%s.%s' % (lang, tgt)
output_file = '../iwslt2017/normal/test.%s.%s.txt' % (lang, tgt)

fp = open(input_file, 'r')
wp = open(output_file, 'w')

for line in fp:
	keyword_raw, code, sentence = line.split('\t')
	wp.write(sentence)

wp.close()