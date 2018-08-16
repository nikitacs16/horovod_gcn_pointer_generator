import glob
import sys
import os
pattern = sys.argv[1]

def file_to_scores(f):
        new_f = open(f,'r')
        s = ''
        for i in new_f.readlines():
                s = s + i
        j = s.split('\n')
        rouge_1_f = float(j[2].split(':')[1].split()[0]) * 100
        rouge_2_f = float(j[7].split(':')[1].split()[0]) * 100
        rouge_l_f = float(j[12].split(':')[1].split()[0]) * 100

        return rouge_1_f, rouge_2_f, rouge_l_f


rouge_1 = 0
rouge_2 = 0
rouge_l = 0
k = len(glob.glob(pattern + '*'))
for i in glob.glob(pattern + '*'):
        rouge_path = os.path.join(glob.glob(os.path.join(i,'decode_val_*'))[0], 'ROUGE_results.txt')
        r1, r2, rl = file_to_scores(rouge_path)
        rouge_1 += r1
        rouge_2 += r2
        rouge_l += rl

rouge_1 = rouge_1/k
rouge_2 = rouge_2/k
rouge_l = rouge_l/k

with open(pattern, 'w') as f:
        f.write("%s\t%f\t%f\t%f"%(pattern,rouge_1,rouge_2,rouge_l))

print("%s\t%f\t%f\t%f"%(pattern,rouge_1,rouge_2,rouge_l))


