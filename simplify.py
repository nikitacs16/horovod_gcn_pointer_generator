import sys
import glob
import rouge
import bleu
import re
import os

def get_number_(s):
        m = re.search("\d+(.)\d+",s)
        return m.group(0)

def get_metrics(f1,f2):
        ref = []
        decoded = []
        count = 0
        for i, j in zip(sorted(glob.glob(f1)),sorted(glob.glob(f2))):
                ref_tex = ''
                dec_tex = ''
                for k in open(i).readlines():
                        dec_tex = dec_tex + k.strip()
                for l in open(j).readlines():
                        ref_tex = ref_tex + l.strip()
                ref.append(ref_tex)
                decoded.append(dec_tex)
                count = count + 1

        bl = bleu.moses_multi_bleu(decoded,ref)
        x = rouge.rouge(decoded,ref)
        s = "\t%.2f\t%.2f\t%.2f\t%.2f"%(bl,x['rouge_1/f_score']*100,x['rouge_2/f_score']*100,x['rouge_l/f_score']*100)
        print(count)
        return s


input_path = sys.argv[1]

for k in sorted(glob.glob(input_path+'/*fusion*run_2')):
	try:
		x = glob.glob(os.path.join(k,'decode_*'))
	except IndexError:
		print(k)
		continue
	x.sort()
	try:
		s = str(x[0][-5:])
	except IndexError:
		print(x)
		continue
	print(x)
	for i in x:
        	f1 = os.path.join(i,'decoded','*.txt')
        	f2 = os.path.join(i,'reference','*.txt')
        	s = s + get_metrics(f1,f2)

	print(s)


