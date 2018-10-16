import re

import os
from subprocess import call
import subprocess


# from itertools import izip
i = 0
for line1, line2 in zip(open('data/valid.de-en.en', 'r'), open('work_dir.concat/decode.dev.beam1.txt', 'r')):
    if i == 6457:
        f = open('gold.txt', 'w')
        g = open('our.txt', 'w')
        f.write(line1)
        g.write(line2)
        gold_len = len(line1.split())
        our_len = len(line2.split())
        f.flush()
        g.flush()
        f.close()
        g.close()
        from subprocess import call
        # p = subprocess.call(["perl multi-bleu.perl data/valid.de-en.en < work_dir.concat/decode.dev.beam1.txt"])
        # output = os.popen("perl multi-bleu.perl /remote/bones/user/dspokoyn/mtclass/assignment1/data/valid.de-en.en < /remote/bones/user/dspokoyn/mtclass/assignment1/work_dir.concat/decode.dev.beam1.txt").read()
        output = os.popen("perl multi-bleu.perl /remote/bones/user/dspokoyn/mtclass/assignment1/gold.txt < /remote/bones/user/dspokoyn/mtclass/assignment1/our.txt").read()
        # print(output, "output")
        # output = output.split(' ').split(',')
        output = re.split(',| ', output)
        print(float(output[2]), gold_len, our_len)
        print(line1)
        print(line2)
        foo
    # else:
    #     continue
    i += 1
    # print(i)

