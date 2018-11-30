import xml.etree.ElementTree as ET

with open("./normal/train.en-de.en", "r") as fin:
    for line in fin.readlines():
        # Strip endline
        line = line.strip()
        # Split tab
        keyword_raw, code, sentence = line.split('\t')

        keyword_node = ET.fromstring(keyword_raw)

        keywords = keyword_node.text.split(', ')
        assert code in ["en", "de", "nl"]

        words = sentence.split()
