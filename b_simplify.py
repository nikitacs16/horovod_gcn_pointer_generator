import sys
import glob
import rouge
import bleu
import re
import os
import logging
import pyrouge
import pandas as pd
def rouge_eval(ref_dir, dec_dir):
	"""Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
	r = pyrouge.Rouge155()
  	r.model_filename_pattern = '#ID#_reference.txt'
  	r.system_filename_pattern = '(\d+)_decoded.txt'
  	r.model_dir = ref_dir
  	r.system_dir = dec_dir
  	logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  	rouge_results = r.convert_and_evaluate()
  	w = r.output_to_dict(rouge_results)
  	return w['rouge_1_f_score']*100,w['rouge_2_f_score']*100,w['rouge_l_f_score']*100

#w= rouge_eval('reference/','decoded/')
#print (w)
#print("%.2f\t%.2f\t%.2f\t",w['rouge_1_f_score']*100,w['rouge_2_f_score']*100,w['rouge_l_f_score']*100)


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
rouge_1 = []
rouge_2 = []
rouge_l = []
fname = []
for k in sorted(glob.glob(input_path+'/decode*test*')):
	'''
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
	'''
	r1, r2, r3 = rouge_eval(os.path.join(k,'reference'), os.path.join(k,'decoded'))
	rouge_1.append(r1)
	rouge_2.append(r2)
	rouge_l.append(r3)
	fname.append(k)
	print("%.2f\t%.2f\t%.2f\t",r1,r2,r3)	
        	#f1 = os.path.join(i,'decoded','*.txt')
        	#f2 = os.path.join(i,'reference','*.txt')
        	#s = s + get_metrics(f1,f2)

	#print(s)
d = {'rouge_1': rouge_1, 'rouge_2' : rouge_2, 'rouge_l' :  rouge_l, 'file_name' : fname}
df = pd.DataFrame(d)
df.to_csv('analysis.csv',index=False)

