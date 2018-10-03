import glob
import sys
import os
import re
input_path = sys.argv[1]
def get_number_(s):
        m = re.search("\d+(.)\d+",s)
        return m.group(0)

def get_txt_from(f):

        s = ''
        for i in f.readlines():
                s = s+ i
        w = s.split('\n')
        r1 = float(get_number_(w[2])) * 100
        r2 = float(get_number_(w[7])) * 100
        rl = float(get_number_(w[12])) * 100

        m = "\t%.2f\t%.2f\t%.2f"%(r1,r2,rl) 
        return m

x = glob.glob(os.path.join(input_path,'decode_*'))
x.sort(reverse=True)
s = str(x[0][-5:]) 
print(x)
for i in x:
        f = open(os.path.join(i,'ROUGE_results.txt'),'r')
        s = s + get_txt_from(f)

print(s)













